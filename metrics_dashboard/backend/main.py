# =============================================================================
# PARTE 1: BACKEND FASTAPI - API Profissional
# =============================================================================
# Arquivo: backend/main.py

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import os
import json
import asyncio
import pandas as pd
import numpy as np
from supabase import create_client, Client
import redis.asyncio as redis
from functools import lru_cache
import logging
from contextlib import asynccontextmanager

# Configuração de logging estruturado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Modelos Pydantic (Validação de Dados)
# =============================================================================

class DiagnosisType(str, Enum):
    MELANOMA = "melanoma"
    NEVUS = "nevus"
    BCC = "basal_cell_carcinoma"
    CBK = "pigmented_benign_keratosis"
    DERMATOFIBROMA = "dermatofibroma"
    VASCULAR = "vascular_lesion"
    ACTINIC_KERATOSIS = "actinic_keratosis"
    SCC = "squamous_cell_carcinoma"


class BatchMetrics(BaseModel):
    batch_name: str
    accuracy: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    specificity: float = Field(..., ge=0, le=1)
    precision: float = Field(..., ge=0, le=1)
    f1_score: float = Field(..., ge=0, le=1)
    roc_auc: Optional[float] = Field(None, ge=0, le=1)
    mcc: float = Field(..., ge=-1, le=1)
    created_at: datetime
    total_cases: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


class ImageMetric(BaseModel):
    universal_id: str
    batch_name: str
    output_name: str
    true_label: str
    predicted_label: str
    confidence: float = Field(..., ge=0, le=1)
    is_correct: bool
    predicted_prob: float
    created_at: datetime


class KPIDashboard(BaseModel):
    global_accuracy: float
    melanoma_sensitivity: float
    melanoma_specificity: float
    avg_confidence: float
    critical_misclassification_rate: float
    total_cases_30d: int
    avg_inference_time_ms: float
    nps_clinical: Optional[float] = None
    timestamp: datetime


class DriftAlert(BaseModel):
    alert_id: str
    metric_name: str
    current_value: float
    baseline_value: float
    deviation_percent: float
    severity: str  # "CRITICAL", "WARNING", "INFO"
    detected_at: datetime
    recommended_action: str


class TimeSeriesPoint(BaseModel):
    timestamp: datetime
    value: float
    batch_name: Optional[str] = None


class ConfusionMatrixData(BaseModel):
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    class_labels: List[str]


# =============================================================================
# Configuração e Conexões
# =============================================================================

class Settings:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL", "300"))
    ALERT_THRESHOLD_SENSITIVITY_MELANOMA = float(os.getenv("ALERT_MELANOMA_SENS", "0.95"))
    ALERT_THRESHOLD_ACCURACY = float(os.getenv("ALERT_ACCURACY", "0.85"))
    DRIFT_PSI_THRESHOLD = float(os.getenv("DRIFT_PSI", "0.25"))


settings = Settings()


# =============================================================================
# Gerenciamento de Conexões ( lifespan )
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    app.state.redis = await redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    logger.info("Conexões estabelecidas: Supabase + Redis")

    yield

    # Shutdown
    await app.state.redis.close()
    logger.info("Conexões encerradas")


app = FastAPI(
    title="DermAI Analytics API",
    description="API profissional para análise de performance de IA dermatológica",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://dermai.clinic"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Dependências
# =============================================================================

def get_supabase(request) -> Client:
    return request.app.state.supabase


def get_redis(request) -> redis.Redis:
    return request.app.state.redis


# =============================================================================
# Camada de Cache Inteligente
# =============================================================================

class CacheManager:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.local_cache = {}
        self.local_ttl = {}

    async def get(self, key: str) -> Optional[str]:
        # Tenta Redis primeiro
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Redis miss para {key}: {e}")

        # Fallback para cache local
        if key in self.local_cache:
            if datetime.now() < self.local_ttl.get(key, datetime.min):
                return self.local_cache[key]
            else:
                del self.local_cache[key]

        return None

    async def set(self, key: str, value: Any, ttl: int = None):
        ttl = ttl or settings.CACHE_TTL_SECONDS
        serialized = json.dumps(value, default=str)

        # Salva no Redis
        try:
            await self.redis.setex(key, ttl, serialized)
        except Exception as e:
            logger.warning(f"Falha ao salvar no Redis: {e}")

        # Salva no cache local como backup
        self.local_cache[key] = json.loads(serialized)
        self.local_ttl[key] = datetime.now() + timedelta(seconds=ttl // 2)  # TTL menor para local


# =============================================================================
# Serviço de Dados (Camada de Negócio)
# =============================================================================

class AnalyticsService:
    def __init__(self, supabase: Client, cache: CacheManager):
        self.supabase = supabase
        self.cache = cache

    async def get_kpis(self, days: int = 30) -> KPIDashboard:
        """Calcula KPIs estratégicos com cache"""
        cache_key = f"kpis:{days}:{datetime.now().strftime('%Y%m%d%H')}"
        cached = await self.cache.get(cache_key)
        if cached:
            return KPIDashboard(**cached)

        # Busca dados da view materializada
        response = self.supabase.rpc("get_dashboard_kpis", {"days": days}).execute()
        data = response.data[0] if response.data else {}

        # Busca métricas de performance de batch
        batch_response = self.supabase.table("batch_metrics").select("*").gte(
            "created_at", (datetime.now() - timedelta(days=days)).isoformat()
        ).execute()

        batches = batch_response.data or []
        avg_inference = np.mean([b.get("elapsed_ms", 200) for b in batches]) if batches else 200

        kpi = KPIDashboard(
            global_accuracy=data.get("global_accuracy", 0.0),
            melanoma_sensitivity=data.get("melanoma_sensitivity", 0.0),
            melanoma_specificity=data.get("melanoma_specificity", 0.0),
            avg_confidence=data.get("avg_confidence", 0.0),
            critical_misclassification_rate=data.get("critical_misclassification_rate", 0.0),
            total_cases_30d=data.get("total_cases", 0),
            avg_inference_time_ms=avg_inference,
            timestamp=datetime.now()
        )

        await self.cache.set(cache_key, kpi.dict())
        return kpi

    async def get_drift_analysis(self, window_days: int = 7) -> List[DriftAlert]:
        """Detecta drift usando PSI e comparação de janelas"""
        cache_key = f"drift:{window_days}:{datetime.now().strftime('%Y%m%d')}"
        cached = await self.cache.get(cache_key)
        if cached:
            return [DriftAlert(**a) for a in cached]

        # Busca dados de drift da view materializada
        response = self.supabase.rpc("calculate_drift_metrics", {
            "recent_days": window_days,
            "previous_days": window_days * 2
        }).execute()

        drift_data = response.data or []
        alerts = []

        for metric in drift_data:
            deviation = abs(metric.get("current_value", 0) - metric.get("previous_value", 0))
            deviation_pct = (deviation / metric.get("previous_value", 1)) * 100 if metric.get("previous_value") else 0

            severity = "INFO"
            if deviation_pct > 20:
                severity = "CRITICAL"
            elif deviation_pct > 10:
                severity = "WARNING"

            if severity != "INFO":
                alerts.append(DriftAlert(
                    alert_id=f"drift_{metric['metric_name']}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    metric_name=metric["metric_name"],
                    current_value=metric["current_value"],
                    baseline_value=metric["previous_value"],
                    deviation_percent=deviation_pct,
                    severity=severity,
                    detected_at=datetime.now(),
                    recommended_action=self._get_drift_recommendation(metric["metric_name"], severity)
                ))

        await self.cache.set(cache_key, [a.dict() for a in alerts], ttl=3600)
        return alerts

    def _get_drift_recommendation(self, metric_name: str, severity: str) -> str:
        recommendations = {
            "accuracy": "Verificar qualidade das novas imagens e possível necessidade de retreinamento",
            "confidence_distribution": "Ajustar temperatura do modelo ou revisar calibração",
            "feature_distribution": "Investigar mudança no pipeline de pré-processamento",
            "melanoma_sensitivity": "Priorizar revisão de falsos negativos de melanoma"
        }
        base = recommendations.get(metric_name, "Investigar causa raiz do drift")
        if severity == "CRITICAL":
            return f"URGENTE: {base}. Considerar pausa no deployment."
        return base

    async def get_diagnosis_performance(self) -> List[Dict]:
        """Performance detalhada por tipo de diagnóstico"""
        response = self.supabase.table("dashboard_diagnosis_performance").select("*").execute()
        return response.data or []

    async def get_confusion_matrix(self, batch_name: Optional[str] = None) -> ConfusionMatrixData:
        """Matriz de confusão para análise"""
        query = self.supabase.table("image_metrics").select("true_label", "predicted_label")
        if batch_name:
            query = query.eq("batch_name", batch_name)

        response = query.execute()
        df = pd.DataFrame(response.data or [])

        if df.empty:
            return ConfusionMatrixData(
                true_negatives=0, false_positives=0,
                false_negatives=0, true_positives=0,
                class_labels=["Não Maligno", "Maligno"]
            )

        # Converte para binário
        df['true_bin'] = df['true_label'].str.lower().isin(['melanoma', 'malignant', 'maligno', '1']).astype(int)
        df['pred_bin'] = df['predicted_label'].str.lower().isin(['melanoma', 'malignant', 'maligno', '1']).astype(int)

        cm = pd.crosstab(df['true_bin'], df['pred_bin'])
        tn = cm.get(0, {}).get(0, 0)
        fp = cm.get(0, {}).get(1, 0)
        fn = cm.get(1, {}).get(0, 0)
        tp = cm.get(1, {}).get(1, 0)

        return ConfusionMatrixData(
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            true_positives=int(tp),
            class_labels=["Não Maligno", "Maligno"]
        )

    async def get_time_series(self, metric: str, days: int = 90) -> List[TimeSeriesPoint]:
        """Série temporal de uma métrica específica"""
        response = self.supabase.table("dashboard_kpis_daily").select(
            "date", metric
        ).gte("date", (datetime.now() - timedelta(days=days)).isoformat()).order("date").execute()

        return [
            TimeSeriesPoint(
                timestamp=row["date"],
                value=row.get(metric, 0),
                batch_name=None
            )
            for row in (response.data or [])
        ]

    async def get_cases_for_review(self, priority: str = "high") -> List[Dict]:
        """Casos prioritários para revisão humana"""
        if priority == "high":
            # Falsos negativos de melanoma
            response = self.supabase.table("image_metrics").select("*").eq(
                "output_name", "benign_malignant"
            ).eq("true_label", "melanoma").eq("predicted_label", "nevus").lt(
                "confidence", 0.7
            ).limit(50).execute()
        else:
            # Baixa confiança geral
            response = self.supabase.table("image_metrics").select("*").lt(
                "confidence", 0.5
            ).limit(50).execute()

        return response.data or []


# =============================================================================
# Endpoints da API
# =============================================================================

@app.get("/")
async def root():
    return {"status": "online", "service": "DermAI Analytics API v2.0", "timestamp": datetime.now()}


@app.get("/api/v1/kpis", response_model=KPIDashboard)
async def get_kpis(
        days: int = 30,
        supabase: Client = Depends(get_supabase),
        redis: redis.Redis = Depends(get_redis)
):
    """KPIs estratégicos do dashboard"""
    cache = CacheManager(redis)
    service = AnalyticsService(supabase, cache)
    return await service.get_kpis(days)


@app.get("/api/v1/drift", response_model=List[DriftAlert])
async def get_drift(
        window_days: int = 7,
        supabase: Client = Depends(get_supabase),
        redis: redis.Redis = Depends(get_redis)
):
    """Análise de drift e alertas"""
    cache = CacheManager(redis)
    service = AnalyticsService(supabase, cache)
    return await service.get_drift_analysis(window_days)


@app.get("/api/v1/diagnosis-performance")
async def get_diagnosis_performance(
        supabase: Client = Depends(get_supabase),
        redis: redis.Redis = Depends(get_redis)
):
    """Performance por tipo de diagnóstico"""
    cache = CacheManager(redis)
    service = AnalyticsService(supabase, cache)
    return await service.get_diagnosis_performance()


@app.get("/api/v1/confusion-matrix", response_model=ConfusionMatrixData)
async def get_confusion_matrix(
        batch_name: Optional[str] = None,
        supabase: Client = Depends(get_supabase)
):
    """Matriz de confusão"""
    service = AnalyticsService(supabase, CacheManager(redis))
    return await service.get_confusion_matrix(batch_name)


@app.get("/api/v1/time-series/{metric}")
async def get_time_series(
        metric: str,
        days: int = 90,
        supabase: Client = Depends(get_supabase)
):
    """Série temporal de métricas"""
    service = AnalyticsService(supabase, CacheManager(redis))
    return await service.get_time_series(metric, days)


@app.get("/api/v1/cases-for-review")
async def get_cases_for_review(
        priority: str = "high",
        supabase: Client = Depends(get_supabase)
):
    """Casos para revisão humana prioritária"""
    service = AnalyticsService(supabase, CacheManager(redis))
    return await service.get_cases_for_review(priority)


@app.post("/api/v1/webhooks/alert")
async def receive_alert(
        alert: DriftAlert,
        background_tasks: BackgroundTasks
):
    """Webhook para receber alertas de sistemas externos"""
    # Processa alerta assincronamente
    background_tasks.add_task(process_alert_notification, alert)
    return {"status": "received", "alert_id": alert.alert_id}


async def process_alert_notification(alert: DriftAlert):
    """Envia notificações para Slack/Email"""
    # Implementação real usaria Slack SDK ou SendGrid
    logger.critical(f"ALERTA {alert.severity}: {alert.metric_name} - {alert.deviation_percent:.1f}% de drift")


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
async def health_check(supabase: Client = Depends(get_supabase)):
    try:
        response = supabase.table("image_metrics").select("count", count="exact").limit(1).execute()
        return {
            "status": "healthy",
            "supabase_connected": True,
            "total_records": response.count,
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Supabase indisponível: {str(e)}")



# =============================================================================
# PARTE 3: FRONTEND REACT - Dashboard Profissional
# =============================================================================
# Arquivo: frontend/src/App.tsx (componentes principais)

"""
// Estrutura de componentes React com Tremor (UI) e Recharts/Plotly (gráficos)

"""

# =============================================================================
# PARTE 4: INFRAESTRUTURA E DEPLOYMENT
# =============================================================================
# Arquivo: docker-compose.yml

"""
version: '3.8'

services:
  # Backend FastAPI
  api:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
      - REDIS_URL=redis://redis:6379
      - CACHE_TTL=300
    depends_on:
      - redis
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1.0'
          memory: 512M

  # Redis para cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru

  # Frontend React (servido via nginx)
  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - api
    environment:
      - REACT_APP_API_URL=http://api:8000

  # Worker para processamento de alertas (Celery opcional)
  worker:
    build: ./backend
    command: celery -A tasks worker --loglevel=info
    environment:
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - api

volumes:
  redis_data:
"""

# =============================================================================
# PARTE 5: CONFIGURAÇÃO DE ALERTAS (Slack/Email)
# =============================================================================
# Arquivo: backend/tasks.py (Celery tasks para notificações)

"""
from celery import Celery
from celery.schedules import crontab
import requests
import os

celery_app = Celery('dermai', broker=os.getenv('REDIS_URL', 'redis://localhost:6379'))

@celery_app.task
def check_and_notify_drift():
    """
Verifica
drift
periodicamente
e
envia
alertas
"""
    from main import AnalyticsService, create_client, CacheManager, redis
    import asyncio

    supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
    cache = CacheManager(redis)
    service = AnalyticsService(supabase, cache)

    alerts = asyncio.run(service.get_drift_analysis(7))

    critical_alerts = [a for a in alerts if a.severity == 'CRITICAL']

    if critical_alerts:
        # Envia para Slack
        webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        if webhook_url:
            requests.post(webhook_url, json={
                "text": f"🚨 ALERTA CRÍTICO DE DRIFT\n\n" + 
                        "\n".join([f"• {a.metric_name}: {a.deviation_percent:.1f}% de desvio" 
                                  for a in critical_alerts])
            })

        # Envia email (SendGrid)
        # Implementação similar...

# Agendamento: verifica a cada 10 minutos
celery_app.conf.beat_schedule = {
    'check-drift': {
        'task': 'tasks.check_and_notify_drift',
        'schedule': 600.0,  # 10 minutos em segundos
    },
}
"""

# =============================================================================
# RESUMO DA ARQUITETURA FINAL
# =============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ARQUITETURA DERMAI ANALYTICS v2.0                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────────────────────┐  │
│  │   USUÁRIO   │──────│   CDN/SSL   │──────│  React + Tremor (Vercel)   │  │
│  │  (Médico/   │      │  (CloudFlare│      │  - Dashboard interativo      │  │
│  │   Gestor)   │      │   /AWS CF)  │      │  - Real-time updates         │  │
│  └─────────────┘      └─────────────┘      │  - Export PDF/PPTX           │  │
│                                              └─────────────────────────────┘  │
│                                                        │                      │
│                                              ┌─────────┴─────────┐           │
│                                              │   WebSocket/SSE     │           │
│                                              │   (alertas tempo    │           │
│                                              │    real)            │           │
│                                              └─────────┬─────────┘           │
│                                                        ▼                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         FASTAPI CLUSTER (2+ replicas)                    │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │ │
│  │  │  API inst.1 │  │  API inst.2 │  │  API inst.N │  │  Health     │     │ │
│  │  │  /api/v1/*  │  │  /api/v1/*  │  │  /api/v1/*  │  │  /health    │     │ │
│  │  │  Gunicorn   │  │  Gunicorn   │  │  Gunicorn   │  │  Uvicorn    │     │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────────────┘     │ │
│  │         └─────────────────┴─────────────────┘                            │ │
│  │                           │                                              │ │
│  │  ┌────────────────────────┴────────────────────────┐                      │ │
│  │  │              Cache Layer (Redis)                │                      │ │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │                      │ │
│  │  │  │ KPIs    │  │ Drift   │  │ Sessions/       │  │                      │ │
│  │  │  │ Cache   │  │ Cache   │  │ Rate Limiting   │  │                      │ │
│  │  │  │ (5min)  │  │ (1h)    │  │                 │  │                      │ │
│  │  │  └─────────┘  └─────────┘  └─────────────────┘  │                      │ │
│  │  └──────────────────────────────────────────────────┘                      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│  ┌───────────────────────────┴─────────────────────────────────────────────┐ │
│  │                         SUPABASE (PostgreSQL)                            │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │ │
│  │  │  VIEWS MATERIALIZADAS (refresh a cada 5min via pg_cron)         │    │ │
│  │  │  ┌─────────────────┐ ┌─────────────────────┐ ┌────────────────┐ │    │ │
│  │  │  │ dashboard_kpis  │ │ dashboard_diagnosis │ │ dashboard_drift │ │    │ │
│  │  │  │ _daily          │ │ _performance        │ │ _monitoring    │ │    │ │
│  │  │  └─────────────────┘ └─────────────────────┘ └────────────────┘ │    │ │
│  │  └─────────────────────────────────────────────────────────────────┘    │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │ │
│  │  │  TABELAS DE PRODUÇÃO                                            │    │ │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌───────────────────────────┐  │    │ │
│  │  │  │ image_      │ │ batch_      │ │ Raw data from pipeline    │  │    │ │
│  │  │  │ metrics     │ │ metrics     │ │ (ham10000_*)              │  │    │ │
│  │  │  └─────────────┘ └─────────────┘ └───────────────────────────┘  │    │ │
│  │  └─────────────────────────────────────────────────────────────────┘    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│  ┌───────────────────────────┴─────────────────────────────────────────────┐ │
│  │                         WORKERS (Celery + Redis)                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │ │
│  │  │  TAREFAS ASSÍNCRONAS                                            │    │ │
│  │  │  • Verificação de drift (a cada 10min)                          │    │ │
│  │  │  • Geração de relatórios PDF (sob demanda)                      │    │ │
│  │  │  • Notificações Slack/Email para alertas CRITICAL               │    │ │
│  │  │  • Pre-computação de métricas para dashboards                   │    │ │
│  │  └─────────────────────────────────────────────────────────────────┘    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""
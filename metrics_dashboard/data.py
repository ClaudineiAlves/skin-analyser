# =============================================================================
# GERADOR DE DADOS REALISTAS PARA TESTE DO DASHBOARD DERMAI
# =============================================================================
# Arquivo: data_generator/generate_realistic_data.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum, auto
import json
import uuid
import random
import math
from pathlib import Path
import hashlib
import os


# =============================================================================
# CONFIGURAÇÃO E CONSTANTES
# =============================================================================

class DiagnosisCategory(Enum):
    """Categorias de diagnóstico do dataset HAM10000"""
    MELANOMA = "melanoma"
    NEVUS = "nevus"
    BCC = "basal_cell_carcinoma"
    CBK = "pigmented_benign_keratosis"
    DERMATOFIBROMA = "dermatofibroma"
    VASCULAR = "vascular_lesion"
    ACTINIC_KERATOSIS = "actinic_keratosis"
    SCC = "squamous_cell_carcinoma"


class MalignancyLevel(Enum):
    """Nível de malignidade para classificação binária"""
    BENIGN = "benign"
    MALIGNANT = "malignant"
    INDETERMINATE = "indeterminate"


# Mapeamento de diagnósticos para categorias de malignidade
DIAGNOSIS_TO_MALIGNANCY = {
    DiagnosisCategory.MELANOMA: MalignancyLevel.MALIGNANT,
    DiagnosisCategory.BCC: MalignancyLevel.MALIGNANT,
    DiagnosisCategory.SCC: MalignancyLevel.MALIGNANT,
    DiagnosisCategory.NEVUS: MalignancyLevel.BENIGN,
    DiagnosisCategory.CBK: MalignancyLevel.BENIGN,
    DiagnosisCategory.DERMATOFIBROMA: MalignancyLevel.BENIGN,
    DiagnosisCategory.VASCULAR: MalignancyLevel.BENIGN,
    DiagnosisCategory.ACTINIC_KERATOSIS: MalignancyLevel.INDETERMINATE,
}

# Mapeamento para diagnosis_1 (categoria geral)
DIAGNOSIS_TO_CATEGORY_1 = {
    DiagnosisCategory.MELANOMA: "Malignant",
    DiagnosisCategory.BCC: "Malignant",
    DiagnosisCategory.SCC: "Malignant",
    DiagnosisCategory.NEVUS: "Benign",
    DiagnosisCategory.CBK: "Benign",
    DiagnosisCategory.DERMATOFIBROMA: "Benign",
    DiagnosisCategory.VASCULAR: "Benign",
    DiagnosisCategory.ACTINIC_KERATOSIS: "Indeterminate",
}

# Mapeamento para diagnosis_2 (subcategoria)
DIAGNOSIS_TO_CATEGORY_2 = {
    DiagnosisCategory.MELANOMA: "Malignant melanocytic proliferations (Melanoma)",
    DiagnosisCategory.BCC: "Malignant adnexal epithelial proliferations - Follicular",
    DiagnosisCategory.SCC: "Malignant epidermal proliferations",
    DiagnosisCategory.NEVUS: "Benign melanocytic proliferations",
    DiagnosisCategory.CBK: "Benign epidermal proliferations",
    DiagnosisCategory.DERMATOFIBROMA: "Benign soft tissue proliferations - Fibro-histiocytic",
    DiagnosisCategory.VASCULAR: "Benign soft tissue proliferations - Vascular",
    DiagnosisCategory.ACTINIC_KERATOSIS: "Indeterminate epidermal proliferations",
}

# Prevalências reais aproximadas do dataset HAM10000
REAL_PREVALENCES = {
    DiagnosisCategory.NEVUS: 0.670,  # 67% - Maioria benigna
    DiagnosisCategory.MELANOMA: 0.111,  # 11.1% - Mais comum maligno
    DiagnosisCategory.BCC: 0.052,  # 5.2%
    DiagnosisCategory.CBK: 0.055,  # 5.5%
    DiagnosisCategory.ACTINIC_KERATOSIS: 0.027,  # 2.7%
    DiagnosisCategory.VASCULAR: 0.014,  # 1.4%
    DiagnosisCategory.DERMATOFIBROMA: 0.011,  # 1.1%
    DiagnosisCategory.SCC: 0.009,  # 0.9%
}

# Performance realista por diagnóstico (baseada em literatura médica)
# Formato: (sensibilidade, especificidade, precisão, confiança_média)
REALISTIC_PERFORMANCE = {
    DiagnosisCategory.MELANOMA: (0.91, 0.96, 0.89, 0.82),
    DiagnosisCategory.NEVUS: (0.93, 0.94, 0.96, 0.85),
    DiagnosisCategory.BCC: (0.78, 0.95, 0.76, 0.71),
    DiagnosisCategory.CBK: (0.89, 0.98, 0.88, 0.79),
    DiagnosisCategory.DERMATOFIBROMA: (0.77, 0.99, 0.81, 0.68),
    DiagnosisCategory.VASCULAR: (0.92, 0.99, 0.90, 0.83),
    DiagnosisCategory.ACTINIC_KERATOSIS: (0.74, 0.95, 0.72, 0.65),
    DiagnosisCategory.SCC: (0.80, 0.98, 0.78, 0.73),
}


# =============================================================================
# CLASSES DE CONFIGURAÇÃO
# =============================================================================

@dataclass
class BatchConfig:
    """Configuração para geração de um batch de dados"""
    batch_name: str
    start_date: datetime
    end_date: datetime
    total_images: int
    model_version: str = "v2.1.0"
    backbone_used: str = "efficientnetb0"

    # Parâmetros de drift (para simular degradação ao longo do tempo)
    accuracy_drift: float = 0.0  # Degradação de acurácia (negativo = piora)
    confidence_drift: float = 0.0  # Degradação de confiança

    # Parâmetros de qualidade de dados
    missing_rate: float = 0.02  # Taxa de dados ausentes
    outlier_rate: float = 0.01  # Taxa de outliers


@dataclass
class GeneratorConfig:
    """Configuração global do gerador"""
    output_dir: Path = Path("./generated_test_data")
    random_seed: int = 42
    n_batches: int = 12  # 12 batches = ~3 meses de dados semanais
    images_per_batch_range: Tuple[int, int] = (150, 450)
    start_date: datetime = field(default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))

    # Simulação de eventos
    simulate_drift_event: bool = True  # Simula evento de drift no meio
    simulate_recovery: bool = True  # Simula recuperação após drift


# =============================================================================
# GERADOR DE DADOS SINTÉTICOS REALISTAS
# =============================================================================

class RealisticDataGenerator:
    """
    Gera dados sintéticos que simulam realisticamente:
    - Distribuições de diagnóstico reais (HAM10000)
    - Performance variável por tipo de lesão
    - Drift temporal (degradação/recuperação do modelo)
    - Padrões de confiança calibrados vs. overconfident
    - Correlações entre métricas
    """

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)
        self._setup_distributions()

        # Cria diretório de saída
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _setup_distributions(self):
        """Configura distribuições de probabilidade realistas"""
        # Distribuição de prevalências
        diagnoses = list(DiagnosisCategory)
        prevalences = [REAL_PREVALENCES[d] for d in diagnoses]
        self.diagnosis_distribution = lambda n: self.rng.choice(
            diagnoses, size=n, p=prevalences
        )

        # Distribuição de confiança (beta para forma realista)
        self.confidence_distribution = lambda alpha, beta, n: self.rng.beta(alpha, beta, n)

    def _generate_universal_id(self, index: int, batch_id: str) -> str:
        """Gera ID único no formato ISIC-like"""
        hash_input = f"{batch_id}_{index}_{self.rng.randint(0, 1000000)}"
        hash_digest = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        return f"ISIC_{hash_digest.upper()}"

    def _generate_timestamp(self, start: datetime, end: datetime) -> datetime:
        """Gera timestamp uniformemente distribuído no intervalo"""
        delta = end - start
        random_seconds = self.rng.randint(0, int(delta.total_seconds()))
        return start + timedelta(seconds=random_seconds)

    def _simulate_model_prediction(
            self,
            true_diagnosis: DiagnosisCategory,
            performance_params: Tuple[float, float, float, float],
            drift_factor: float = 0.0
    ) -> Tuple[str, float, bool]:
        """
        Simula predição do modelo com performance realista.

        Returns:
            (predicted_label, confidence, is_correct)
        """
        sensitivity, specificity, precision, avg_confidence = performance_params

        # Aplica drift (degradação)
        effective_sensitivity = max(0.3, min(0.99, sensitivity + drift_factor))
        effective_specificity = max(0.3, min(0.99, specificity + drift_factor))
        effective_confidence = max(0.3, min(0.99, avg_confidence + drift_factor * 0.5))

        is_malignant = DIAGNOSIS_TO_MALIGNANCY[true_diagnosis] == MalignancyLevel.MALIGNANT

        # Simula decisão do modelo baseada em sensibilidade/especificidade
        if is_malignant:
            # Caso positivo real: modelo acerta com probabilidade = sensibilidade
            correct = self.rng.random() < effective_sensitivity
        else:
            # Caso negativo real: modelo acerta com probabilidade = especificidade
            correct = self.rng.random() < effective_specificity

        # Determina predição
        if correct:
            predicted = true_diagnosis
        else:
            # Erro: escolhe diagnóstico confundível
            predicted = self._select_confusable_diagnosis(true_diagnosis)

        # Gera confiança realista
        if correct:
            # Acertos: confiança alta mas com variabilidade
            confidence = self.rng.beta(8, 2) * effective_confidence + (1 - effective_confidence) * 0.5
        else:
            # Erros: confiança mais baixa (mas alguns overconfident)
            if self.rng.random() < 0.15:  # 15% overconfident errors
                confidence = self.rng.beta(7, 3) * 0.9 + 0.1
            else:
                confidence = self.rng.beta(2, 5) * 0.6 + 0.2

        confidence = max(0.01, min(0.99, confidence))

        return predicted.value, float(confidence), correct

    def _select_confusable_diagnosis(self, true: DiagnosisCategory) -> DiagnosisCategory:
        """Seleciona diagnóstico confundível realisticamente"""
        # Matriz de confusão realista (quem costuma ser confundido com quem)
        confusion_pairs = {
            DiagnosisCategory.MELANOMA: [DiagnosisCategory.NEVUS, DiagnosisCategory.CBK],
            DiagnosisCategory.NEVUS: [DiagnosisCategory.MELANOMA, DiagnosisCategory.CBK],
            DiagnosisCategory.BCC: [DiagnosisCategory.ACTINIC_KERATOSIS, DiagnosisCategory.SCC],
            DiagnosisCategory.CBK: [DiagnosisCategory.NEVUS, DiagnosisCategory.MELANOMA],
            DiagnosisCategory.ACTINIC_KERATOSIS: [DiagnosisCategory.BCC, DiagnosisCategory.SCC],
            DiagnosisCategory.SCC: [DiagnosisCategory.BCC, DiagnosisCategory.ACTINIC_KERATOSIS],
            DiagnosisCategory.DERMATOFIBROMA: [DiagnosisCategory.VASCULAR, DiagnosisCategory.NEVUS],
            DiagnosisCategory.VASCULAR: [DiagnosisCategory.DERMATOFIBROMA, DiagnosisCategory.NEVUS],
        }

        candidates = confusion_pairs.get(true, list(DiagnosisCategory))
        weights = [3 if c == candidates[0] else 1 for c in candidates]
        weights = np.array(weights) / sum(weights)

        return self.rng.choice(candidates, p=weights)

    def _generate_augmentation_variants(self, base_confidence: float) -> List[Dict]:
        """Gera variantes de augmentação com consenso realista"""
        n_variants = self.rng.randint(3, 12)
        variants = []

        # Confiança base varia por tipo de augmentação
        aug_effects = {
            'rotation': 0.95, 'flip': 0.98, 'zoom': 0.90,
            'color_jitter': 0.85, 'noise': 0.75, 'blur': 0.70
        }

        for i in range(n_variants):
            aug_type = self.rng.choice(list(aug_effects.keys()))
            effect = aug_effects[aug_type]

            # Confiança da variante = base * efeito + ruído
            var_confidence = base_confidence * effect + self.rng.normal(0, 0.05)
            var_confidence = max(0.01, min(0.99, var_confidence))

            variants.append({
                'variant_id': i,
                'augmentation_type': aug_type,
                'confidence': float(var_confidence),
                'predicted_label': 'melanoma' if var_confidence > 0.7 else 'nevus'  # Simplificado
            })

        return variants

    def generate_batch(self, config: BatchConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Gera dados de um batch completo.

        Returns:
            (df_images, df_batch_metrics)
        """
        print(f"Gerando batch: {config.batch_name} ({config.total_images} imagens)")

        # Determina diagnósticos reais
        true_diagnoses = self.diagnosis_distribution(config.total_images)

        records = []
        for i in range(config.total_images):
            true_diag = true_diagnoses[i]
            perf = REALISTIC_PERFORMANCE[true_diag]

            # Simula predição
            pred_label, confidence, is_correct = self._simulate_model_prediction(
                true_diag, perf, config.accuracy_drift
            )

            # Gera metadados
            universal_id = self._generate_universal_id(i, config.batch_name)
            timestamp = self._generate_timestamp(config.start_date, config.end_date)

            # Determina labels binários
            true_malignant = DIAGNOSIS_TO_MALIGNANCY[true_diag] == MalignancyLevel.MALIGNANT
            pred_malignant = DIAGNOSIS_TO_MALIGNANCY.get(
                DiagnosisCategory(pred_label) if pred_label in [d.value for d in
                                                                DiagnosisCategory] else DiagnosisCategory.NEVUS,
                MalignancyLevel.BENIGN
            ) == MalignancyLevel.MALIGNANT

            # Gera variantes de augmentação
            variants = self._generate_augmentation_variants(confidence)

            # Features extraídas (simuladas)
            shape_features = self._generate_shape_features(true_diag, is_correct)
            color_features = self._generate_color_features(true_diag)
            texture_features = self._generate_texture_features()

            record = {
                # Identificadores
                'universal_id': universal_id,
                'source_image_id': i,
                'batch_name': config.batch_name,
                'created_at': timestamp.isoformat(),

                # Labels verdadeiros
                'true_diagnosis': true_diag.value,
                'true_diagnosis_1': DIAGNOSIS_TO_CATEGORY_1[true_diag],
                'true_diagnosis_2': DIAGNOSIS_TO_CATEGORY_2[true_diag],
                'true_benign_malignant': 'malignant' if true_malignant else 'benign',
                'true_melanocytic': 'melanocytic' if 'melanocytic' in DIAGNOSIS_TO_CATEGORY_2[
                    true_diag].lower() else 'non-melanocytic',

                # Predições do modelo
                'predicted_diagnosis': pred_label,
                'predicted_diagnosis_1': DIAGNOSIS_TO_CATEGORY_1.get(DiagnosisCategory(pred_label), 'Benign'),
                'predicted_diagnosis_2': DIAGNOSIS_TO_CATEGORY_2.get(DiagnosisCategory(pred_label),
                                                                     'Benign melanocytic proliferations'),
                'predicted_benign_malignant': 'malignant' if pred_malignant else 'benign',

                # Métricas de confiança
                'confidence': float(confidence),
                'predicted_prob': float(confidence),  # Probabilidade de classe positiva
                'is_correct': bool(is_correct),

                # Variantes de augmentação (JSON)
                'augmentation_variants': json.dumps(variants),
                'consensus_confidence': float(np.mean([v['confidence'] for v in variants])),
                'consensus_agreement': float(
                    sum(1 for v in variants if v['predicted_label'] == pred_label) / len(variants)),

                # Features extraídas (simuladas)
                'shape_asymmetry': shape_features['asymmetry'],
                'shape_border_irregularity': shape_features['border'],
                'shape_diameter_mm': shape_features['diameter'],
                'color_variance': color_features['variance'],
                'texture_contrast': texture_features['contrast'],

                # Metadados do modelo
                'model_version': config.model_version,
                'backbone_used': config.backbone_used,
                'inference_time_ms': float(self.rng.exponential(50) + 100),  # ~150ms média

                # Flags de qualidade
                'is_outlier': self.rng.random() < config.outlier_rate,
                'has_missing_data': self.rng.random() < config.missing_rate,
            }

            records.append(record)

        df_images = pd.DataFrame(records)

        # Calcula métricas agregadas do batch
        df_batch = self._calculate_batch_metrics(df_images, config)

        return df_images, df_batch

    def _generate_shape_features(self, diagnosis: DiagnosisCategory, is_correct: bool) -> Dict[str, float]:
        """Gera features de forma realistas"""
        base_asymmetry = {
            DiagnosisCategory.MELANOMA: 0.65,
            DiagnosisCategory.NEVUS: 0.30,
            DiagnosisCategory.BCC: 0.55,
        }.get(diagnosis, 0.45)

        # Erros tendem a ter features mais ambíguas
        noise = 0.2 if not is_correct else 0.1

        return {
            'asymmetry': float(np.clip(self.rng.normal(base_asymmetry, noise), 0, 1)),
            'border': float(np.clip(self.rng.normal(0.5, 0.15), 0, 1)),
            'diameter': float(np.clip(self.rng.lognormal(2, 0.5), 2, 20))
        }

    def _generate_color_features(self, diagnosis: DiagnosisCategory) -> Dict[str, float]:
        """Gera features de cor realistas"""
        base_variance = {
            DiagnosisCategory.MELANOMA: 0.75,
            DiagnosisCategory.NEVUS: 0.40,
            DiagnosisCategory.VASCULAR: 0.85,
        }.get(diagnosis, 0.50)

        return {
            'variance': float(np.clip(self.rng.normal(base_variance, 0.1), 0, 1)),
            'dominant_colors': 3 + self.rng.poisson(2)
        }

    def _generate_texture_features(self) -> Dict[str, float]:
        """Gera features de textura"""
        return {
            'contrast': float(self.rng.gamma(2, 0.3)),
            'homogeneity': float(self.rng.beta(3, 2)),
            'energy': float(self.rng.beta(2, 3))
        }

    def _calculate_batch_metrics(self, df: pd.DataFrame, config: BatchConfig) -> pd.DataFrame:
        """Calcula métricas agregadas do batch"""

        # Métricas para benign_malignant (binário)
        y_true = df['true_benign_malignant'].isin(['malignant', 'melanoma', '1']).astype(int)
        y_pred = df['predicted_benign_malignant'].isin(['malignant', 'melanoma', '1']).astype(int)
        y_scores = df['confidence']

        # Calcula componentes da matriz de confusão
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # MCC
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if all([
            (tp + fp) > 0, (tp + fn) > 0, (tn + fp) > 0, (tn + fn) > 0
        ]) else 0
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0

        # ROC-AUC (simplificado)
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, y_scores)
        except:
            auc = 0.5 + (accuracy - 0.5) * 0.8  # Aproximação

        # Métricas por diagnóstico específico
        per_diagnosis = []
        for diag in DiagnosisCategory:
            mask = df['true_diagnosis'] == diag.value
            if mask.sum() == 0:
                continue

            diag_correct = df.loc[mask, 'is_correct'].mean()
            diag_conf = df.loc[mask, 'confidence'].mean()

            per_diagnosis.append({
                'diagnosis': diag.value,
                'count': int(mask.sum()),
                'accuracy': float(diag_correct),
                'avg_confidence': float(diag_conf)
            })

        batch_record = {
            'batch_name': config.batch_name,
            'created_at': config.end_date.isoformat(),
            'model_version': config.model_version,
            'backbone_used': config.backbone_used,

            # Contagens
            'total_images': len(df),
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,

            # Métricas principais
            'accuracy': float(accuracy),
            'recall': float(recall),
            'specificity': float(specificity),
            'precision': float(precision),
            'f1_score': float(f1),
            'roc_auc': float(auc),
            'mcc': float(mcc),

            # Estatísticas de confiança
            'avg_confidence': float(df['confidence'].mean()),
            'std_confidence': float(df['confidence'].std()),
            'min_confidence': float(df['confidence'].min()),
            'max_confidence': float(df['confidence'].max()),

            # Performance por diagnóstico
            'per_diagnosis_metrics': json.dumps(per_diagnosis),

            # Metadados de execução
            'elapsed_ms': float(df['inference_time_ms'].mean()),
            'drift_applied': config.accuracy_drift,
        }

        return pd.DataFrame([batch_record])

    def generate_full_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Gera dataset completo com múltiplos batches e eventos realistas.
        """
        all_images = []
        all_batches = []

        current_date = self.config.start_date

        for batch_idx in range(self.config.n_batches):
            # Determina tamanho do batch
            n_images = self.rng.randint(
                self.config.images_per_batch_range[0],
                self.config.images_per_batch_range[1]
            )

            # Calcula datas
            batch_duration = timedelta(days=7)
            start_date = current_date
            end_date = current_date + batch_duration

            # Calcula drift (simula degradação e recuperação)
            drift = 0.0
            if self.config.simulate_drift_event:
                # Drift significativo no batches 5-7 (simula problema no pipeline)
                if 5 <= batch_idx <= 7:
                    drift = -0.15 - (batch_idx - 5) * 0.05  # Degradação progressiva
                # Recuperação gradual
                elif batch_idx == 8:
                    drift = -0.10
                elif batch_idx == 9:
                    drift = -0.05
                elif batch_idx >= 10:
                    drift = 0.0  # Recuperação completa

            batch_config = BatchConfig(
                batch_name=f"HAM10000_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}",
                start_date=start_date,
                end_date=end_date,
                total_images=n_images,
                model_version="v2.1.0" if batch_idx < 8 else "v2.2.0-hotfix",
                backbone_used=self.rng.choice([
                    "efficientnetb0", "resnet50", "densenet121"
                ]),
                accuracy_drift=drift,
                confidence_drift=drift * 0.5
            )

            df_img, df_batch = self.generate_batch(batch_config)
            all_images.append(df_img)
            all_batches.append(df_batch)

            current_date = end_date + timedelta(hours=1)

        # Concatena todos os batches
        df_all_images = pd.concat(all_images, ignore_index=True)
        df_all_batches = pd.concat(all_batches, ignore_index=True)

        # Adiciona colunas para compatibilidade com dashboard
        df_all_images = self._format_for_dashboard(df_all_images)

        return df_all_images, df_all_batches

    def _format_for_dashboard(self, df: pd.DataFrame) -> pd.DataFrame:
        """Formata DataFrame para compatibilidade com colunas do dashboard"""

        # Mapeia para nomes de colunas esperados pelo dashboard
        column_mapping = {
            'true_diagnosis': 'true_label',
            'predicted_diagnosis': 'predicted_label',
            'true_benign_malignant': 'real_diagnoses',
            'predicted_benign_malignant': 'ia_diagnoses',
            'confidence': 'confidence',  # Já está correto
            'predicted_prob': 'predicted_prob',  # Já está correto
            'is_correct': 'is_correct',  # Já está correto
            'created_at': 'created_at',  # Já está correto
            'batch_name': 'batch_name',  # Já está correto
            'universal_id': 'universal_id',  # Já está correto
        }

        # Cria colunas derivadas para o dashboard
        df['output_name'] = 'benign_malignant'  # Output principal

        # Converte labels para formato binário do dashboard
        df['real_diagnoses_bin'] = df['true_benign_malignant'].apply(
            lambda x: 1 if str(x).lower() in ['malignant', 'melanoma', 'maligno', '1', 'true'] else 0
        )
        df['ia_diagnoses_bin'] = df['predicted_benign_malignant'].apply(
            lambda x: 1 if str(x).lower() in ['malignant', 'melanoma', 'maligno', '1', 'true'] else 0
        )

        # Gera coluna de timestamp como datetime
        df['timestamp'] = pd.to_datetime(df['created_at'])

        # Adiciona coluna de probabilidade predita (para ROC)
        # Probabilidade "calibrada" baseada na confiança
        df['predicted_prob'] = df.apply(
            lambda row: row['confidence'] if row['ia_diagnoses_bin'] == 1 else 1 - row['confidence'],
            axis=1
        )

        # Adiciona ruído realisticamente calibrado
        df['predicted_prob'] = df['predicted_prob'].apply(
            lambda p: np.clip(p + np.random.normal(0, 0.05), 0.01, 0.99)
        )

        return df

    def save_to_files(self, df_images: pd.DataFrame, df_batches: pd.DataFrame):
        """Salva dados gerados em múltiplos formatos"""
        output = self.config.output_dir

        # CSV principal
        df_images.to_csv(output / "image_metrics.csv", index=False)
        df_batches.to_csv(output / "batch_metrics.csv", index=False)

        # JSON para API
        df_images.to_json(output / "image_metrics.json", orient="records", date_format="iso")
        df_batches.to_json(output / "batch_metrics.json", orient="records", date_format="iso")

        # Parquet para análise eficiente
        df_images.to_parquet(output / "image_metrics.parquet", index=False)
        df_batches.to_parquet(output / "batch_metrics.parquet", index=False)

        # SQL de inserção para Supabase
        self._generate_sql_inserts(df_images, df_batches, output)

        # Relatório de qualidade
        self._generate_quality_report(df_images, df_batches, output)

        print(f"\n✅ Dados salvos em: {output.absolute()}")
        print(f"   - image_metrics.csv: {len(df_images)} registros")
        print(f"   - batch_metrics.csv: {len(df_batches)} registros")

    def _generate_sql_inserts(self, df_img: pd.DataFrame, df_batch: pd.DataFrame, output: Path):
        """Gera SQL INSERT para popular Supabase diretamente"""

        # Simplifica DataFrame para SQL (remove JSON complexo)
        df_sql = df_img[[
            'universal_id', 'batch_name', 'output_name',
            'true_label', 'predicted_label', 'confidence',
            'predicted_prob', 'is_correct', 'created_at'
        ]].copy()

        sql_lines = ["-- Generated SQL for Supabase insertion", ""]
        sql_lines.append("BEGIN;")

        # Inserts para image_metrics
        sql_lines.append("\n-- Insert into image_metrics")
        for _, row in df_sql.iterrows():
            sql_lines.append(f"""
                INSERT INTO image_metrics (
                    universal_id, batch_name, output_name,
                    true_label, predicted_label, confidence,
                    predicted_prob, is_correct, created_at
                ) VALUES (
                    '{row['universal_id']}', '{row['batch_name']}', '{row['output_name']}',
                    '{row['true_label']}', '{row['predicted_label']}', {row['confidence']:.6f},
                    {row['predicted_prob']:.6f}, {str(row['is_correct']).lower()}, '{row['created_at']}'
                ) ON CONFLICT DO NOTHING;
            """)

        # Inserts para batch_metrics
        sql_lines.append("\n-- Insert into batch_metrics")
        for _, row in df_batch.iterrows():
            sql_lines.append(f"""
                INSERT INTO batch_metrics (
                    batch_name, created_at, accuracy, recall, specificity,
                    precision, f1_score, roc_auc, mcc, total_images
                ) VALUES (
                    '{row['batch_name']}', '{row['created_at']}', {row['accuracy']:.6f},
                    {row['recall']:.6f}, {row['specificity']:.6f}, {row['precision']:.6f},
                    {row['f1_score']:.6f}, {row['roc_auc']:.6f}, {row['mcc']:.6f},
                    {row['total_images']}
                ) ON CONFLICT DO NOTHING;
            """)

        sql_lines.append("COMMIT;")

        with open(output / "insert_data.sql", "w", encoding="utf-8") as f:
            f.write("\n".join(sql_lines))

    def _generate_quality_report(self, df_img: pd.DataFrame, df_batch: pd.DataFrame, output: Path):
        """Gera relatório de qualidade dos dados"""

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_images": len(df_img),
            "total_batches": len(df_batch),
            "date_range": {
                "start": df_img['created_at'].min(),
                "end": df_img['created_at'].max()
            },
            "diagnosis_distribution": df_img['true_diagnosis'].value_counts().to_dict(),
            "malignancy_distribution": df_img['true_benign_malignant'].value_counts().to_dict(),
            "overall_accuracy": float(df_img['is_correct'].mean()),
            "melanoma_sensitivity": float(
                df_img[(df_img['true_diagnosis'] == 'melanoma') & (df_img['predicted_diagnosis'] == 'melanoma')].shape[
                    0] /
                df_img[df_img['true_diagnosis'] == 'melanoma'].shape[0]
            ) if df_img[df_img['true_diagnosis'] == 'melanoma'].shape[0] > 0 else 0,
            "batch_drift_detected": bool((df_batch['drift_applied'] < -0.1).any()),
            "avg_inference_time_ms": float(df_img['inference_time_ms'].mean()),
        }

        with open(output / "quality_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)


# =============================================================================
# SCRIPT DE EXECUÇÃO
# =============================================================================

def main():
    """Executa geração de dados de teste"""

    print("=" * 70)
    print("GERADOR DE DADOS REALISTAS - DERMAI ANALYTICS")
    print("=" * 70)

    # Configuração
    config = GeneratorConfig(
        output_dir=Path("./generated_test_data"),
        random_seed=42,
        n_batches=15,  # ~3.5 meses de dados
        images_per_batch_range=(200, 500),
        start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        simulate_drift_event=True,
        simulate_recovery=True
    )

    # Gera dados
    generator = RealisticDataGenerator(config)
    df_images, df_batches = generator.generate_full_dataset()

    # Salva arquivos
    generator.save_to_files(df_images, df_batches)

    # Preview
    print("\n" + "=" * 70)
    print("PREVIEW DOS DADOS GERADOS")
    print("=" * 70)

    print("\n--- Primeiros registros de image_metrics ---")
    print(df_images[[
        'universal_id', 'batch_name', 'true_diagnosis',
        'predicted_diagnosis', 'confidence', 'is_correct'
    ]].head(10).to_string())

    print("\n--- Métricas agregadas por batch ---")
    print(df_batches[[
        'batch_name', 'total_images', 'accuracy', 'recall',
        'specificity', 'f1_score', 'drift_applied'
    ]].to_string())

    print("\n--- Distribuição de diagnósticos ---")
    print(df_images['true_diagnosis'].value_counts())

    print("\n--- Evolução temporal da acurácia ---")
    print(df_batches[['batch_name', 'accuracy', 'drift_applied']].to_string())

    print("\n" + "=" * 70)
    print("GERAÇÃO CONCLUÍDA COM SUCESSO")
    print("=" * 70)


if __name__ == "__main__":
    main()
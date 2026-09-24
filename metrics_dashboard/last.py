import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
from supabase import create_client, Client
from dotenv import load_dotenv
from pathlib import Path
import logging
import os

# =========================================
# region Constantes e Configurações
# =========================================
LOG_DIR = Path("z-other/logs")
POS_LABELS = {"melanoma", "maligno", "cancer", "câncer", "positivo", "malign", "1", "true"}
METRIC_COLS = ["accuracy", "recall", "specificity", "precision", "f1_score", "roc_auc", "mcc"]
# endregion

# =========================================
# region Logs
# =========================================
LOG_DIR.mkdir(parents=True, exist_ok=True)

def define_logs(name: str, filename: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(LOG_DIR / filename, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
    return logger

logger_success = define_logs("success", "upload_success.log")
logger_error   = define_logs("error",   "upload_error.log")
# endregion

# =========================================
# region Conexão e Dados (COM CORREÇÃO)
# =========================================
@st.cache_resource(ttl=600)
def init_supabase_client() -> Client | None:
    env_path = Path("link_supabase.env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not all([url, key]):
        st.error("Variáveis de ambiente do Supabase URL/Key não encontradas.")
        logger_error.error("Missing Supabase URL or Key environment variables.")
        return None
        
    try:
        return create_client(url, key)
    except Exception as e:
        logger_error.error(f"Erro ao criar cliente Supabase: {e}")
        st.error(f"Erro ao conectar com o Supabase: {e}")
        return None

# ===== INÍCIO DA ATUALIZAÇÃO =====
@st.cache_data(ttl=300)
def fetch_data(_supabase: Client) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        # Consulta 1: Busca os dados agregados diretamente da tabela de batches.
        # Esta é a fonte principal para os gráficos de evolução e comparação.
        response_batches = _supabase.table("metrics_by_batch_2").select("*").order("created_at", desc=True).execute()
        if not response_batches.data:
            st.warning("Nenhum dado de batch foi encontrado no Supabase.")
            return pd.DataFrame(), pd.DataFrame()

        df_batches = pd.DataFrame(response_batches.data)
        # Converte as colunas de métricas para numérico e data para datetime
        for col in METRIC_COLS:
            if col in df_batches.columns:
                df_batches[col] = pd.to_numeric(df_batches[col], errors="coerce")
        if "created_at" in df_batches.columns:
            df_batches["created_at"] = pd.to_datetime(df_batches["created_at"])


        # Consulta 2: Busca os dados individuais de cada imagem, juntando o nome do batch.
        response_images = _supabase.table("metrics_by_images_2").select("*, batch_id!inner(batch_name)").execute()
        if not response_images.data:
            st.warning("Nenhum dado de imagem foi encontrado no Supabase.")
            return pd.DataFrame(), df_batches # Retorna os batches mesmo se não houver imagens

        df_images_raw = pd.json_normalize(response_images.data, sep="_")
        
        # Renomeia as colunas para o formato esperado
        image_cols = {
            "batch_id_batch_name": "batch_name",
            "real_diagnoses": "real_diagnoses",
            "ia_diagnoses": "ia_diagnoses",
            "metric_result": "metric_result",
            "predicted_prob": "predicted_prob",
            "created_at": "created_at"
        }
        df_images = df_images_raw.rename(columns=image_cols)
        
        # Seleciona apenas as colunas necessárias e faz as conversões de tipo
        df_images = df_images[list(image_cols.values())]
        df_images["predicted_prob"] = pd.to_numeric(df_images["predicted_prob"], errors="coerce")
        df_images["created_at"] = pd.to_datetime(df_images["created_at"])
        
        logger_success.info(f"{len(df_images)} imagens | {len(df_batches)} batches carregados.")
        return df_images, df_batches

    except Exception as e:
        logger_error.error(f"Erro ao buscar dados: {e}")
        st.error(f"Erro ao buscar dados do Supabase: {e}")
        return pd.DataFrame(), pd.DataFrame()
# ===== FIM DA ATUALIZAÇÃO =====
# endregion

# =========================================
# region Utils
# =========================================
def to_binary(s: str | int) -> int:
    return 1 if str(s).strip().lower() in POS_LABELS else 0
# endregion

# =========================================
# region Plots
# =========================================
def plot_confusion_matrix(y_true, y_pred, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    z_text = [[f"TN: {cm[0,0]}", f"FP: {cm[0,1]}"], [f"FN: {cm[1,0]}", f"TP: {cm[1,1]}"]]

    fig = px.imshow(cm, text_auto=True, labels=dict(x="Diagnóstico Previsto", y="Diagnóstico Real"),
                      x=["Negativo", "Positivo"], y=["Negativo", "Positivo"],
                      title=title, color_continuous_scale="Purples", template="plotly_dark")
    fig.update_traces(text=z_text, texttemplate="%{text}")
    st.plotly_chart(fig, use_container_width=True)

def plot_roc_curve(y_true, y_scores, title: str):
    if y_true.nunique() < 2:
        st.info(f"Dados insuficientes para gerar a curva ROC (necessita de ambas as classes).")
        return

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Modelo (AUC = {auc_score:.3f})', line=dict(color='cyan', width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aleatório', line=dict(color='gray', dash='dash')))
    fig.update_layout(title=title, xaxis_title="Taxa de Falsos Positivos", yaxis_title="Taxa de Verdadeiros Positivos",
                      xaxis=dict(constrain='domain'), yaxis=dict(scaleanchor='x', scaleratio=1), template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def plot_metric_bars(df_batches: pd.DataFrame, selected_batches: list[str]):
    df_filtered = df_batches[df_batches["batch_name"].isin(selected_batches)]
    num_cols = 2
    cols = st.columns(num_cols)
    
    for i, metric in enumerate(METRIC_COLS):
        if metric not in df_filtered.columns or df_filtered[metric].isnull().all():
            continue
        with cols[i % num_cols]:
            df_sorted = df_filtered.sort_values(by=metric, ascending=False)
            fig = px.bar(df_sorted, x="batch_name", y=metric, text_auto=".3f", 
                         title=f"Comparativo de {metric.replace('_', ' ').title()}",
                         template="plotly_dark", 
                         color="batch_name",
                         color_discrete_sequence=px.colors.qualitative.Vivid,
                         hover_data={"batch_name": True, metric: ':.3f'})
            fig.update_layout(yaxis_range=[0, 1.05], xaxis_title=None, yaxis_title="Valor", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def plot_metrics_evolution(df_batches: pd.DataFrame, selected_batches: list[str]):
    st.markdown("### 📈 Evolução do Desempenho ao Longo do Tempo")
    # Ordena por 'created_at' em ordem crescente para o gráfico de linha
    df_plot = df_batches[df_batches["batch_name"].isin(selected_batches)].sort_values("created_at", ascending=True)

    if df_plot.empty or len(df_plot) < 2:
        st.info("Selecione pelo menos dois batches para ver a evolução.")
        return
        
    existing_metrics = [metric for metric in METRIC_COLS if metric in df_plot.columns and not df_plot[metric].isnull().all()]

    if not existing_metrics:
        st.warning("Nenhuma métrica válida foi encontrada nos dados dos batches selecionados para gerar o gráfico.")
        return
        
    df_melted = df_plot.melt(
        id_vars=["created_at", "batch_name"], 
        value_vars=existing_metrics,
        var_name="Métrica", 
        value_name="Valor"
    )

    if df_melted.empty:
        st.warning("Não foi possível gerar o gráfico de evolução com os dados disponíveis.")
        return

    fig = px.line(
        df_melted, 
        x="created_at", 
        y="Valor", 
        color="Métrica", 
        markers=True,
        title="Evolução das Métricas dos Batches Selecionados",
        labels={"created_at": "Data de Criação", "Valor": "Performance"},
        hover_data={"batch_name": True, "Valor": ":.3f"},
        template="plotly_dark"
    )
    fig.update_layout(yaxis_range=[-0.05, 1.05])
    st.plotly_chart(fig, use_container_width=True)
# endregion

# =========================================
# region Glossário (COM TEXTO COMPLETO)
# =========================================
def render_glossario_tab():
    st.header("📚 Glossário de Métricas de Classificação")
    st.markdown("Entenda o que cada métrica significa, como é calculada e como interpretá-la.")
    
    st.subheader("A Base: Matriz de Confusão")
    st.markdown("""
    A maioria das métricas deriva de quatro valores fundamentais da Matriz de Confusão, que compara os valores reais com as previsões do modelo. Para o exemplo de um diagnóstico de câncer:
    """)
    with st.expander("Clique para ver os componentes da Matriz de Confusão"):
        st.markdown("- **Verdadeiro Positivo (VP ou TP):** O paciente *tem* câncer e o modelo *previu* corretamente que ele tem câncer.")
        st.markdown("- **Verdadeiro Negativo (VN ou TN):** O paciente *não tem* câncer e o modelo *previu* corretamente que ele não tem.")
        st.markdown("- **Falso Positivo (FP ou Erro Tipo I):** O paciente *não tem* câncer, mas o modelo *previu* que ele tem. (Alarme falso)")
        st.markdown("- **Falso Negativo (FN ou Erro Tipo II):** O paciente *tem* câncer, mas o modelo *previu* que ele não tem. (Erro perigoso)")

    st.markdown("---")
    
    metricas = {
        "Acurácia (Accuracy)": {
            "formula": r"Acurácia = \frac{VP + VN}{VP + VN + FP + FN}",
            "descricao": "Mede a proporção de previsões corretas (positivas e negativas) em relação ao total de previsões. É a métrica mais intuitiva.",
            "interpretacao": "Um valor de 0.90 significa que 90% de todas as previsões foram corretas. **Cuidado:** A acurácia pode ser enganosa em cenários com dados desbalanceados. Se 99% dos pacientes são saudáveis, um modelo que sempre prevê 'saudável' terá 99% de acurácia, mas será inútil para detectar a doença."
        },
        "Precisão (Precision)": {
            "formula": r"Precisão = \frac{VP}{VP + FP}",
            "descricao": "De todas as vezes que o modelo previu 'Positivo', quantas ele acertou?",
            "interpretacao": "Mede a confiabilidade da previsão positiva. Uma precisão de 0.85 significa que quando o modelo diagnostica um paciente com câncer, ele está correto 85% das vezes. É crucial quando o custo de um **Falso Positivo** é alto (ex: tratamentos desnecessários, pânico)."
        },
        "Recall (Sensibilidade ou Revocação)": {
            "formula": r"Recall = \frac{VP}{VP + FN}",
            "descricao": "De todos os casos que eram realmente 'Positivos', quantos o modelo conseguiu identificar?",
            "interpretacao": "Mede a capacidade do modelo de encontrar todos os casos positivos. Um recall de 0.95 significa que o modelo identificou 95% de todos os pacientes que realmente tinham câncer. É a métrica mais importante quando o custo de um **Falso Negativo** é altíssimo (ex: não diagnosticar uma doença grave)."
        },
        "Especificidade (Specificity)": {
            "formula": r"Especificidade = \frac{VN}{VN + FP}",
            "descricao": "De todos os casos que eram realmente 'Negativos', quantos o modelo conseguiu identificar?",
            "interpretacao": "É o 'Recall' para a classe negativa. Mede a capacidade do modelo de identificar corretamente os casos saudáveis. Uma alta especificidade é importante para evitar alarmes falsos em programas de triagem."
        },
        "F1-Score": {
            "formula": r"F1 = 2 \times \frac{Precisão \times Recall}{Precisão + Recall}",
            "descricao": "É a média harmônica entre Precisão e Recall. Cria um equilíbrio entre as duas métricas.",
            "interpretacao": "O F1-Score é útil quando você precisa de um balanço entre minimizar os Falsos Positivos e os Falsos Negativos, e não pode priorizar um em detrimento do outro. Varia de 0 a 1, onde 1 é o melhor valor."
        },
        "AUC-ROC": {
            "formula": "Não possui uma fórmula simples; é a 'Área Sob a Curva ROC'.",
            "descricao": "A Curva ROC plota a taxa de Verdadeiros Positivos (Recall) contra a taxa de Falsos Positivos em vários limiares de decisão. A AUC é a área sob essa curva.",
            "interpretacao": "Mede a capacidade geral do modelo de distinguir entre as classes positiva e negativa. Uma AUC de 1.0 significa um classificador perfeito. Uma AUC de 0.5 representa um modelo que acerta tanto quanto um palpite aleatório. É uma ótima métrica para avaliar o desempenho geral, especialmente em dados desbalanceados."
        },
        "MCC (Coeficiente de Correlação de Matthews)": {
            "formula": r"MCC = \frac{VP \times VN - FP \times FN}{\sqrt{(VP+FP)(VP+FN)(VN+FP)(VN+FN)}}",
            "descricao": "Mede a qualidade da classificação binária, considerando todos os quatro valores da matriz de confusão. É considerada uma das métricas mais robustas e balanceadas.",
            "interpretacao": "O valor varia de -1 a +1. **+1** indica uma previsão perfeita. **0** representa uma previsão não melhor que aleatória. **-1** indica uma total discordância entre previsão e realidade. É altamente recomendada para conjuntos de dados desbalanceados."
        }
    }
    
    for metrica, detalhes in metricas.items():
        st.subheader(metrica)
        st.markdown(f"**O que é?**\n{detalhes['descricao']}")
        st.markdown("**Como se calcula?**")
        st.latex(detalhes['formula'])
        st.markdown(f"**Como interpretar?**\n{detalhes['interpretacao']}")
        st.markdown("---")
# endregion

# =========================================
# region Dashboard
# =========================================
def run_dashboard():
    st.set_page_config(page_title="Dashboard de Métricas", layout="wide", initial_sidebar_state="expanded")
    st.title("📊 Dashboard de Desempenho de Classificação")

    supabase = init_supabase_client()
    if not supabase: st.stop()

    df_images, df_batches = fetch_data(supabase)
    if df_batches.empty: 
        st.warning("Não foi possível carregar os dados dos batches. O dashboard não pode continuar.")
        st.stop()

    # --- BARRA LATERAL (FILTROS) ---
    st.sidebar.header("⚙️ Filtros")
    # Garante que os batches são ordenados pela data de criação mais recente na sidebar
    available_batches = df_batches.sort_values("created_at", ascending=False)["batch_name"].dropna().unique().tolist()
    
    select_all = st.sidebar.checkbox("Selecionar Todos os Batches", value=True)
    if select_all:
        selected_batches = st.sidebar.multiselect("Batches para Análise:", options=available_batches, default=available_batches)
    else:
        default_selection = available_batches[:min(5, len(available_batches))] if available_batches else []
        selected_batches = st.sidebar.multiselect("Batches para Análise:", options=available_batches, default=default_selection)
    
    if not selected_batches:
        st.warning("Por favor, selecione pelo menos um batch na barra lateral.")
        st.stop()

    # --- KPIs Interativos ---
    kpi_options = ["Média Geral"] + selected_batches
    selection = st.selectbox("Selecione a visão para as métricas em destaque:", options=kpi_options, label_visibility="collapsed")
    
    data_source = None
    title = ""
    df_kpi_selected = df_batches[df_batches["batch_name"].isin(selected_batches)]
    
    available_kpi_metrics = [m for m in METRIC_COLS if m in df_kpi_selected.columns]

    if selection == "Média Geral":
        title = "Resumo Médio dos Batches Selecionados"
        if not df_kpi_selected.empty:
            data_source = df_kpi_selected[available_kpi_metrics].mean()
    else:
        title = f"Métricas para o Batch: {selection}"
        batch_data = df_kpi_selected[df_kpi_selected["batch_name"] == selection]
        if not batch_data.empty:
            data_source = batch_data[available_kpi_metrics].iloc[0]

    st.markdown(f"#### {title}")
    
    if data_source is not None and not data_source.empty:
        cols_to_display = [col for col in data_source.index if pd.notna(data_source[col])]
        if cols_to_display:
            cols = st.columns(len(cols_to_display))
            for i, metric in enumerate(cols_to_display):
                value = data_source[metric]
                cols[i].metric(label=metric.replace("_", " ").title(), value=f"{value:.3f}")
    
    st.markdown("---")

    # --- TABS ---
    tabs_list = ["📊 **Visão Agregada**", "⚖️ Comparativo entre Batches", "📈 Evolução no Tempo", "🔎 Análise por Batch Individual", "📚 Glossário de Métricas"]
    tab_agg, tab_bars, tab_lines, tab_ind, tab_glossario = st.tabs(tabs_list)

    with tab_agg:
        st.header("Análise Agregada dos Batches Selecionados")
        if df_images.empty:
            st.warning("Não há dados de imagem disponíveis para a análise agregada.")
        else:
            df_filtered_images = df_images[df_images['batch_name'].isin(selected_batches)]
            st.markdown(f"Esta visão combina todas as **{len(df_filtered_images)} imagens** dos **{len(selected_batches)} batches** selecionados.")
            
            df_agg = df_filtered_images.dropna(subset=['real_diagnoses', 'predicted_prob', 'ia_diagnoses'])
            
            if df_agg.empty:
                st.warning("Não há dados de imagem suficientes nos batches selecionados para a análise agregada.")
            else:
                y_true_agg = df_agg['real_diagnoses'].map(to_binary)
                y_pred_agg = df_agg['ia_diagnoses'].map(to_binary)
                y_scores_agg = df_agg['predicted_prob']

                col1, col2 = st.columns([1, 1.5])
                with col1:
                    plot_confusion_matrix(y_true_agg, y_pred_agg, title="Matriz de Confusão Agregada")
                with col2:
                    plot_roc_curve(y_true_agg, y_scores_agg, title="Curva ROC Agregada")

                st.markdown("#### Relatório de Classificação Agregado")
                report = classification_report(y_true_agg, y_pred_agg, target_names=["Negativo (Classe 0)", "Positivo (Classe 1)"])
                st.code(report, language='text')

    with tab_bars:
        st.header("Comparativo de Desempenho entre Batches Individuais")
        plot_metric_bars(df_batches, selected_batches)

    with tab_lines:
        st.header("Evolução do Desempenho dos Batches ao Longo do Tempo")
        plot_metrics_evolution(df_batches, selected_batches)

    with tab_ind:
        st.header("Análise Detalhada de um Batch Específico")
        batch_to_inspect = st.selectbox("Selecione um batch para inspecionar:", options=selected_batches)
        
        if batch_to_inspect:
            if df_images.empty:
                 st.warning(f"Não há dados de imagem disponíveis para inspecionar o batch '{batch_to_inspect}'.")
            else:
                df_inspect = df_images[df_images['batch_name'] == batch_to_inspect].dropna(subset=['real_diagnoses', 'predicted_prob', 'ia_diagnoses'])

                if df_inspect.empty:
                    st.warning(f"Não há dados de imagem suficientes no batch '{batch_to_inspect}' para análise.")
                else:
                    y_true_ind = df_inspect['real_diagnoses'].map(to_binary)
                    y_pred_ind = df_inspect['ia_diagnoses'].map(to_binary)
                    y_scores_ind = df_inspect['predicted_prob']
                    
                    c1, c2 = st.columns([1, 1.5])
                    with c1:
                        plot_confusion_matrix(y_true_ind, y_pred_ind, title=f"Matriz de Confusão — {batch_to_inspect}")
                    with c2:
                        plot_roc_curve(y_true_ind, y_scores_ind, title=f"Curva ROC — {batch_to_inspect}")

                    st.markdown("#### Amostra de Dados do Batch")
                    st.dataframe(df_inspect[["real_diagnoses", "ia_diagnoses", "predicted_prob", "metric_result"]].head(20), use_container_width=True)
    
    with tab_glossario:
        render_glossario_tab()

if __name__ == "__main__":
    run_dashboard()
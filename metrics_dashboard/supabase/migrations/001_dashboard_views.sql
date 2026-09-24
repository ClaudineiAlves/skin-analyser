

-- # =============================================================================
-- # PARTE 2: VIEWS MATERIALIZADAS SUPABASE (SQL)
-- # =============================================================================
-- Arquivo: supabase/migrations/001_dashboard_views.sql

-- View: KPIs diários agregados
CREATE MATERIALIZED VIEW dashboard_kpis_daily AS
WITH daily_stats AS (
    SELECT
        DATE_TRUNC('day', created_at) as date,
        COUNT(DISTINCT universal_id) as total_cases,
        AVG(is_correct::int) as global_accuracy,
        AVG(confidence) as avg_confidence,
        AVG(CASE WHEN true_label = 'melanoma' AND predicted_label = 'melanoma' THEN 1.0
                 WHEN true_label = 'melanoma' THEN 0.0 END) as melanoma_sensitivity,
        AVG(CASE WHEN true_label != 'melanoma' AND predicted_label != 'melanoma' THEN 1.0
                 WHEN true_label != 'melanoma' THEN 0.0 END) as melanoma_specificity,
        AVG(CASE WHEN true_label IN ('melanoma', 'squamous cell carcinoma', 'basal cell carcinoma')
                  AND predicted_label IN ('nevus', 'pigmented benign keratosis', 'dermatofibroma')
             THEN 1.0 ELSE 0.0 END) as critical_misclassification_rate
    FROM image_metrics
    WHERE created_at >= NOW() - INTERVAL '90 days'
    GROUP BY DATE_TRUNC('day', created_at)
)
SELECT * FROM daily_stats
ORDER BY date DESC;

-- Índice para performance
CREATE UNIQUE INDEX idx_dashboard_kpis_daily_date ON dashboard_kpis_daily(date);
CREATE INDEX idx_dashboard_kpis_daily_date_brin ON dashboard_kpis_daily USING BRIN(date);

-- View: Performance por diagnóstico
CREATE MATERIALIZED VIEW dashboard_diagnosis_performance AS
SELECT
    output_name,
    true_label,
    predicted_label,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence,
    AVG(is_correct::int) as accuracy,
    -- Precisão para esta classe
    SUM(CASE WHEN predicted_label = true_label THEN 1 ELSE 0 END)::float /
        NULLIF(SUM(CASE WHEN predicted_label = true_label THEN 1 ELSE 0 END), 0) as precision_for_class,
    -- Recall para esta classe
    SUM(CASE WHEN true_label = predicted_label THEN 1 ELSE 0 END)::float /
        NULLIF(COUNT(*), 0) as recall_for_class
FROM image_metrics
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY output_name, true_label, predicted_label;

-- Índice composto
CREATE UNIQUE INDEX idx_diag_perf_composite ON dashboard_diagnosis_performance(output_name, true_label, predicted_label);

-- View: Drift monitoring
CREATE MATERIALIZED VIEW dashboard_drift_monitoring AS
WITH
recent_window AS (
    SELECT * FROM image_metrics
    WHERE created_at >= NOW() - INTERVAL '7 days'
),
previous_window AS (
    SELECT * FROM image_metrics
    WHERE created_at >= NOW() - INTERVAL '14 days'
    AND created_at < NOW() - INTERVAL '7 days'
),
recent_stats AS (
    SELECT
        AVG(confidence) as avg_confidence,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY confidence) as median_confidence,
        STDDEV(confidence) as std_confidence
    FROM recent_window
),
previous_stats AS (
    SELECT
        AVG(confidence) as avg_confidence,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY confidence) as median_confidence,
        STDDEV(confidence) as std_confidence
    FROM previous_window
)
SELECT
    'confidence_mean' as metric_name,
    r.avg_confidence as current_value,
    p.avg_confidence as previous_value,
    ABS(r.avg_confidence - p.avg_confidence) / NULLIF(p.avg_confidence, 0) as relative_change
FROM recent_stats r, previous_stats p
UNION ALL
SELECT
    'confidence_std',
    r.std_confidence,
    p.std_confidence,
    ABS(r.std_confidence - p.std_confidence) / NULLIF(p.std_confidence, 0)
FROM recent_stats r, previous_stats p;

-- Função RPC para KPIs (usada pela API)
CREATE OR REPLACE FUNCTION get_dashboard_kpis(days_param int)
RETURNS TABLE (
    global_accuracy float,
    melanoma_sensitivity float,
    melanoma_specificity float,
    avg_confidence float,
    critical_misclassification_rate float,
    total_cases bigint
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        AVG(d.global_accuracy),
        AVG(d.melanoma_sensitivity),
        AVG(d.melanoma_specificity),
        AVG(d.avg_confidence),
        AVG(d.critical_misclassification_rate),
        SUM(d.total_cases)::bigint
    FROM dashboard_kpis_daily d
    WHERE d.date >= NOW() - (days_param || ' days')::interval;
END;
$$ LANGUAGE plpgsql;

-- Função RPC para drift
CREATE OR REPLACE FUNCTION calculate_drift_metrics(
    recent_days int,
    previous_days int
)
RETURNS TABLE (
    metric_name text,
    current_value float,
    previous_value float
) AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM dashboard_drift_monitoring;
END;
$$ LANGUAGE plpgsql;

-- Refresh automático a cada 5 minutos (usando pg_cron ou trigger)
SELECT cron.schedule('refresh-dashboard-views', '*/5 * * * *',
    'REFRESH MATERIALIZED VIEW CONCURRENTLY dashboard_kpis_daily; REFRESH MATERIALIZED VIEW CONCURRENTLY dashboard_diagnosis_performance;'
);

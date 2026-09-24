
import React, { useState } from 'react';
import {
  Card, Title, Text, Metric, Flex, Grid, Col, BarList,
  DonutChart, AreaChart, LineChart, BarChart, Legend,
  Color, Button, Badge, Tab, TabGroup, TabList, TabPanel,
  Select, SelectItem, DateRangePicker
} from '@tremor/react';
import { 
  AlertTriangle, CheckCircle, TrendingUp, TrendingDown,
  Activity, Brain, Clock, Shield, Eye, Filter
} from 'lucide-react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format, subDays } from 'date-fns';
import { ptBR } from 'date-fns/locale';

// =============================================================================
// Tipos TypeScript (mapeando os modelos Pydantic)
// =============================================================================

interface KPIDashboard {
  global_accuracy: number;
  melanoma_sensitivity: number;
  melanoma_specificity: number;
  avg_confidence: number;
  critical_misclassification_rate: number;
  total_cases_30d: number;
  avg_inference_time_ms: number;
  nps_clinical?: number;
  timestamp: string;
}

interface DriftAlert {
  alert_id: string;
  metric_name: string;
  current_value: number;
  baseline_value: number;
  deviation_percent: number;
  severity: 'CRITICAL' | 'WARNING' | 'INFO';
  detected_at: string;
  recommended_action: string;
}

interface DiagnosisPerformance {
  output_name: string;
  true_label: string;
  predicted_label: string;
  count: number;
  avg_confidence: number;
  accuracy: number;
  precision_for_class: number;
  recall_for_class: number;
}

interface ConfusionMatrixData {
  true_negatives: number;
  false_positives: number;
  false_negatives: number;
  true_positives: number;
  class_labels: string[];
}

interface TimeSeriesPoint {
  timestamp: string;
  value: number;
  batch_name?: string;
}

// =============================================================================
// Serviço de API (React Query + Axios)
// =============================================================================

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const fetchKPIs = async (days: number): Promise<KPIDashboard> => {
  const res = await fetch(`${API_BASE}/api/v1/kpis?days=${days}`);
  if (!res.ok) throw new Error('Falha ao carregar KPIs');
  return res.json();
};

const fetchDrift = async (windowDays: number): Promise<DriftAlert[]> => {
  const res = await fetch(`${API_BASE}/api/v1/drift?window_days=${windowDays}`);
  if (!res.ok) throw new Error('Falha ao carregar drift');
  return res.json();
};

const fetchDiagnosisPerformance = async (): Promise<DiagnosisPerformance[]> => {
  const res = await fetch(`${API_BASE}/api/v1/diagnosis-performance`);
  if (!res.ok) throw new Error('Falha ao carregar performance');
  return res.json();
};

const fetchConfusionMatrix = async (batchName?: string): Promise<ConfusionMatrixData> => {
  const url = batchName 
    ? `${API_BASE}/api/v1/confusion-matrix?batch_name=${batchName}`
    : `${API_BASE}/api/v1/confusion-matrix`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Falha ao carregar matriz');
  return res.json();
};

const fetchTimeSeries = async (metric: string, days: number): Promise<TimeSeriesPoint[]> => {
  const res = await fetch(`${API_BASE}/api/v1/time-series/${metric}?days=${days}`);
  if (!res.ok) throw new Error('Falha ao carregar série temporal');
  return res.json();
};

// =============================================================================
// Componente: KPI Cards (Topo do Dashboard)
// =============================================================================

const KPICards: React.FC<{ data: KPIDashboard }> = ({ data }) => {
  const kpiConfig = [
    {
      title: "Taxa de Acerto Global",
      value: data.global_accuracy,
      format: "percentage",
      target: 0.90,
      icon: CheckCircle,
      color: "emerald" as Color,
      description: "De todas as previsões, quantas o modelo acertou"
    },
    {
      title: "Sensibilidade a Melanoma",
      value: data.melanoma_sensitivity,
      format: "percentage",
      target: 0.95,
      icon: Shield,
      color: "rose" as Color,
      description: "De todos os melanomas reais, quantos foram detectados"
    },
    {
      title: "Especificidade Global",
      value: data.melanoma_specificity,
      format: "percentage",
      target: 0.95,
      icon: Activity,
      color: "blue" as Color,
      description: "De todos os casos benignos, quantos foram corretamente identificados"
    },
    {
      title: "Confiança Média",
      value: data.avg_confidence,
      format: "percentage",
      target: 0.80,
      icon: Brain,
      color: "amber" as Color,
      description: "Nível médio de certeza do modelo nas previsões"
    },
    {
      title: "Taxa de Erros Críticos",
      value: data.critical_misclassification_rate,
      format: "percentage",
      target: 0.05,
      icon: AlertTriangle,
      color: "red" as Color,
      description: "Malignos classificados como benignos (FN perigosos)",
      inverse: true // Quanto menor, melhor
    },
    {
      title: "Tempo Médio de Inferência",
      value: data.avg_inference_time_ms,
      format: "milliseconds",
      target: 200,
      icon: Clock,
      color: "indigo" as Color,
      description: "Tempo médio de processamento por imagem"
    }
  ];

  return (
    <Grid numItemsMd={2} numItemsLg={3} className="gap-6">
      {kpiConfig.map((kpi) => {
        const isGood = kpi.inverse 
          ? kpi.value <= kpi.target 
          : kpi.value >= kpi.target;
        const Icon = kpi.icon;

        return (
          <Card key={kpi.title} decoration="top" decorationColor={kpi.color}>
            <Flex justifyContent="start" className="space-x-4">
              <Icon className={`w-8 h-8 text-${kpi.color}-500`} />
              <div className="truncate">
                <Text className="truncate">{kpi.title}</Text>
                <Metric className="truncate">
                  {kpi.format === 'percentage' 
                    ? `${(kpi.value * 100).toFixed(1)}%`
                    : `${kpi.value.toFixed(0)}ms`
                  }
                </Metric>
              </div>
            </Flex>
            <Flex className="mt-4 space-x-2">
              <Badge color={isGood ? "emerald" : "red"} size="sm">
                {isGood ? "✓ Meta atingida" : "⚠ Abaixo da meta"}
              </Badge>
              <Text className="text-xs text-gray-500">{kpi.description}</Text>
            </Flex>
            {/* Mini sparkline */}
            <div className="mt-3 h-8">
              <MiniSparkline 
                data={generateMockSparkline(kpi.value, kpi.target)} 
                color={kpi.color}
                target={kpi.target}
              />
            </div>
          </Card>
        );
      })}
    </Grid>
  );
};

// =============================================================================
// Componente: Alertas de Drift
// =============================================================================

const DriftAlerts: React.FC<{ alerts: DriftAlert[] }> = ({ alerts }) => {
  if (alerts.length === 0) {
    return (
      <Card>
        <Flex justifyContent="start" className="space-x-2">
          <CheckCircle className="w-5 h-5 text-emerald-500" />
          <Text>Sistema estável. Nenhum drift detectado nas últimas 7 dias.</Text>
        </Flex>
      </Card>
    );
  }

  return (
    <div className="space-y-3">
      {alerts.map((alert) => (
        <Card key={alert.alert_id} decoration="left" 
              decorationColor={alert.severity === 'CRITICAL' ? 'red' : 'amber'}>
          <Flex justifyContent="between" alignItems="start">
            <div>
              <Flex className="space-x-2">
                <AlertTriangle className={`w-5 h-5 ${
                  alert.severity === 'CRITICAL' ? 'text-red-500' : 'text-amber-500'
                }`} />
                <Title className="text-sm">
                  {alert.severity === 'CRITICAL' ? 'CRÍTICO' : 'AVISO'}: {alert.metric_name}
                </Title>
              </Flex>
              <Text className="mt-1">
                Desvio de {alert.deviation_percent.toFixed(1)}% 
                (atual: {alert.current_value.toFixed(3)}, 
                 baseline: {alert.baseline_value.toFixed(3)})
              </Text>
              <Text className="text-xs text-gray-500 mt-1">
                Ação recomendada: {alert.recommended_action}
              </Text>
            </div>
            <Button size="xs" variant="secondary" color="gray">
              Investigar
            </Button>
          </Flex>
        </Card>
      ))}
    </div>
  );
};

// =============================================================================
// Componente: Tabela de Performance por Diagnóstico
// =============================================================================

const DiagnosisPerformanceTable: React.FC<{ data: DiagnosisPerformance[] }> = ({ data }) => {
  // Agrega por true_label (diagnóstico real)
  const aggregated = React.useMemo(() => {
    const byDiagnosis: Record<string, {
      total_cases: number;
      correct: number;
      precision_sum: number;
      recall_sum: number;
      confidence_sum: number;
    }> = {};

    data.forEach(row => {
      if (!byDiagnosis[row.true_label]) {
        byDiagnosis[row.true_label] = {
          total_cases: 0, correct: 0, 
          precision_sum: 0, recall_sum: 0, confidence_sum: 0
        };
      }
      const d = byDiagnosis[row.true_label];
      d.total_cases += row.count;
      d.correct += row.count * row.accuracy;
      d.precision_sum += row.precision_for_class * row.count;
      d.recall_sum += row.recall_for_class * row.count;
      d.confidence_sum += row.avg_confidence * row.count;
    });

    return Object.entries(byDiagnosis).map(([label, stats]) => ({
      label,
      total_cases: stats.total_cases,
      accuracy: stats.correct / stats.total_cases,
      precision: stats.precision_sum / stats.total_cases,
      recall: stats.recall_sum / stats.total_cases,
      avg_confidence: stats.confidence_sum / stats.total_cases,
      status: stats.recall_sum / stats.total_cases >= 0.90 ? 'good' : 
              stats.recall_sum / stats.total_cases >= 0.80 ? 'warning' : 'critical'
    }));
  }, [data]);

  const statusColors = {
    good: 'emerald',
    warning: 'amber',
    critical: 'red'
  };

  return (
    <Card>
      <Title>Performance por Tipo de Lesão</Title>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead className="text-xs text-gray-700 uppercase bg-gray-50">
            <tr>
              <th className="px-4 py-3">Diagnóstico</th>
              <th className="px-4 py-3">Casos</th>
              <th className="px-4 py-3">Acurácia</th>
              <th className="px-4 py-3">Precisão</th>
              <th className="px-4 py-3">Sensibilidade</th>
              <th className="px-4 py-3">Confiança Média</th>
              <th className="px-4 py-3">Status</th>
            </tr>
          </thead>
          <tbody>
            {aggregated.map((row) => (
              <tr key={row.label} className="border-b hover:bg-gray-50">
                <td className="px-4 py-3 font-medium">{row.label}</td>
                <td className="px-4 py-3">{row.total_cases.toLocaleString()}</td>
                <td className="px-4 py-3">
                  <Badge color={row.accuracy >= 0.90 ? 'emerald' : row.accuracy >= 0.80 ? 'amber' : 'red'}>
                    {(row.accuracy * 100).toFixed(1)}%
                  </Badge>
                </td>
                <td className="px-4 py-3">{(row.precision * 100).toFixed(1)}%</td>
                <td className="px-4 py-3">{(row.recall * 100).toFixed(1)}%</td>
                <td className="px-4 py-3">{(row.avg_confidence * 100).toFixed(1)}%</td>
                <td className="px-4 py-3">
                  <Badge color={statusColors[row.status] as Color}>
                    {row.status === 'good' ? '✓ Bom' : 
                     row.status === 'warning' ? '⚠ Regular' : '❌ Crítico'}
                  </Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
};

// =============================================================================
// Componente: Matriz de Confusão Interativa (Plotly via React)
// =============================================================================

const ConfusionMatrixHeatmap: React.FC<{ data: ConfusionMatrixData }> = ({ data }) => {
  const matrix = [
    [data.true_negatives, data.false_positives],
    [data.false_negatives, data.true_positives]
  ];

  const annotations = [
    { text: `VN: ${data.true_negatives}`, x: 0, y: 0 },
    { text: `FP: ${data.false_positives}`, x: 1, y: 0 },
    { text: `FN: ${data.false_negatives}`, x: 0, y: 1 },
    { text: `VP: ${data.true_positives}`, x: 1, y: 1 }
  ];

  // Usando Recharts para heatmap simplificado (ou Plotly se preferir)
  return (
    <Card>
      <Title>Matriz de Confusão</Title>
      <div className="mt-4 flex justify-center">
        <div className="grid grid-cols-2 gap-1 w-64">
          {matrix.flat().map((val, idx) => (
            <div 
              key={idx}
              className={`p-4 text-center rounded-lg ${
                idx === 0 ? 'bg-emerald-100 text-emerald-800' :
                idx === 3 ? 'bg-emerald-100 text-emerald-800' :
                'bg-red-100 text-red-800'
              }`}
            >
              <div className="text-2xl font-bold">{val}</div>
              <div className="text-xs">
                {idx === 0 ? 'Verdadeiro Negativo' :
                 idx === 1 ? 'Falso Positivo' :
                 idx === 2 ? 'Falso Negativo' : 'Verdadeiro Positivo'}
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
        <div>
          <Text className="text-gray-500">Taxa de Falsos Positivos</Text>
          <Metric>{(data.false_positives / (data.false_positives + data.true_negatives) * 100).toFixed(1)}%</Metric>
        </div>
        <div>
          <Text className="text-gray-500">Taxa de Falsos Negativos</Text>
          <Metric>{(data.false_negatives / (data.false_negatives + data.true_positives) * 100).toFixed(1)}%</Metric>
        </div>
      </div>
    </Card>
  );
};

// =============================================================================
// Componente: Gráfico de Evolução Temporal
// =============================================================================

const MetricsEvolutionChart: React.FC<{ metric: string; days: number }> = ({ metric, days }) => {
  const { data, isLoading } = useQuery(
    ['timeSeries', metric, days],
    () => fetchTimeSeries(metric, days),
    { refetchInterval: 300000 } // Refresh a cada 5 minutos
  );

  if (isLoading || !data) return <Card><Text>Carregando...</Text></Card>;

  const chartData = data.map(d => ({
    date: format(new Date(d.timestamp), 'dd/MM'),
    value: d.value,
    // Adiciona banda de confiança
    upper: d.value * 1.05,
    lower: d.value * 0.95
  }));

  return (
    <Card>
      <Title>Evolução de {metric.replace('_', ' ').toUpperCase()}</Title>
      <AreaChart
        className="h-72 mt-4"
        data={chartData}
        index="date"
        categories={["value"]}
        colors={["indigo"]}
        showLegend={false}
        showGridLines={false}
        yAxisWidth={40}
      />
    </Card>
  );
};

// =============================================================================
// Componente Principal: App
// =============================================================================

const App: React.FC = () => {
  const [selectedDays, setSelectedDays] = useState(30);
  const [activeTab, setActiveTab] = useState(0);

  const queryClient = useQueryClient();

  // Queries com React Query (cache, stale-while-revalidate, background refresh)
  const { data: kpis, isLoading: kpisLoading } = useQuery(
    ['kpis', selectedDays],
    () => fetchKPIs(selectedDays),
    { 
      refetchInterval: 300000, // 5 minutos
      staleTime: 60000, // 1 minuto
      cacheTime: 3600000 // 1 hora
    }
  );

  const { data: driftAlerts } = useQuery(
    ['drift', 7],
    () => fetchDrift(7),
    { refetchInterval: 600000 } // 10 minutos
  );

  const { data: diagnosisPerf } = useQuery(
    ['diagnosisPerf'],
    fetchDiagnosisPerformance,
    { refetchInterval: 300000 }
  );

  const { data: confusionMatrix } = useQuery(
    ['confusionMatrix'],
    () => fetchConfusionMatrix(),
    { refetchInterval: 300000 }
  );

  // Mutação para refresh manual
  const refreshMutation = useMutation(
    () => Promise.all([
      queryClient.invalidateQueries(['kpis']),
      queryClient.invalidateQueries(['drift']),
      queryClient.invalidateQueries(['diagnosisPerf']),
      queryClient.invalidateQueries(['confusionMatrix'])
    ])
  );

  if (kpisLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-indigo-600" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      {/* Header */}
      <div className="mb-8">
        <Flex justifyContent="between" alignItems="center">
          <div>
            <Title className="text-2xl">🔬 DermAI Analytics</Title>
            <Text className="text-gray-500">
              Dashboard de Performance de IA Dermatológica
            </Text>
          </div>
          <div className="flex items-center space-x-4">
            <Select 
              value={selectedDays.toString()} 
              onValueChange={(v) => setSelectedDays(Number(v))}
            >
              <SelectItem value="7">Últimos 7 dias</SelectItem>
              <SelectItem value="30">Últimos 30 dias</SelectItem>
              <SelectItem value="90">Últimos 90 dias</SelectItem>
            </Select>
            <Button 
              icon={Filter} 
              variant="secondary"
              onClick={() => refreshMutation.mutate()}
              loading={refreshMutation.isLoading}
            >
              Atualizar Dados
            </Button>
          </div>
        </Flex>
        <Text className="text-xs text-gray-400 mt-1">
          Última atualização: {kpis ? format(new Date(kpis.timestamp), 'dd/MM/yyyy HH:mm') : '-'}
        </Text>
      </div>

      {/* Alertas de Drift */}
      {driftAlerts && driftAlerts.length > 0 && (
        <div className="mb-6">
          <DriftAlerts alerts={driftAlerts} />
        </div>
      )}

      {/* KPI Cards */}
      {kpis && <KPICards data={kpis} />}

      {/* Tabs de Análise */}
      <div className="mt-8">
        <TabGroup index={activeTab} onIndexChange={setActiveTab}>
          <TabList className="mt-4">
            <Tab icon={Activity}>Visão Geral</Tab>
            <Tab icon={Brain}>Performance por Diagnóstico</Tab>
            <Tab icon={Eye}>Análise de Erros</Tab>
            <Tab icon={TrendingUp}>Evolução Temporal</Tab>
          </TabList>

          <TabPanel>
            <Grid numItemsMd={2} className="gap-6 mt-6">
              <Col>
                {confusionMatrix && <ConfusionMatrixHeatmap data={confusionMatrix} />}
              </Col>
              <Col>
                <MetricsEvolutionChart metric="global_accuracy" days={selectedDays} />
              </Col>
            </Grid>
          </TabPanel>

          <TabPanel>
            <div className="mt-6">
              {diagnosisPerf && <DiagnosisPerformanceTable data={diagnosisPerf} />}
            </div>
          </TabPanel>

          <TabPanel>
            <Card className="mt-6">
              <Title>Análise de Erros para Revisão</Title>
              <Text className="mt-2">
                Casos prioritários identificados automaticamente pelo sistema de monitoramento.
              </Text>
              {/* Aqui entraria a tabela de casos para revisão, buscada de /api/v1/cases-for-review */}
            </Card>
          </TabPanel>

          <TabPanel>
            <Grid numItemsMd={2} className="gap-6 mt-6">
              <MetricsEvolutionChart metric="melanoma_sensitivity" days={selectedDays} />
              <MetricsEvolutionChart metric="avg_confidence" days={selectedDays} />
            </Grid>
          </TabPanel>
        </TabGroup>
      </div>
    </div>
  );
};

// =============================================================================
// Utilitários
// =============================================================================

const MiniSparkline: React.FC<{ data: number[]; color: string; target: number }> = ({ 
  data, color, target 
}) => {
  // Implementação simplificada - em produção usar Recharts Sparkline
  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;

  return (
    <svg viewBox="0 0 100 30" className="w-full h-full">
      <line x1="0" y1={30 - ((target - min) / range) * 30} 
            x2="100" y2={30 - ((target - min) / range) * 30} 
            stroke="gray" strokeDasharray="2" strokeWidth="0.5" />
      <polyline
        fill="none"
        stroke={color}
        strokeWidth="2"
        points={data.map((d, i) => 
          `${(i / (data.length - 1)) * 100},${30 - ((d - min) / range) * 30}`
        ).join(' ')}
      />
    </svg>
  );
};

const generateMockSparkline = (current: number, target: number): number[] => {
  // Gera dados simulados para sparkline
  return Array.from({ length: 20 }, (_, i) => {
    const noise = (Math.random() - 0.5) * 0.1;
    const trend = (i / 20) * (current - target) * 0.3;
    return Math.max(0, Math.min(1, target + trend + noise));
  });
};

export default App;
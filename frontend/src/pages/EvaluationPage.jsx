import React, { useState, useEffect } from 'react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Award, BookOpen, FileText, CheckCircle2, Info } from 'lucide-react';
import { getEvaluationSummary } from '../services/api';

export default function EvaluationPage() {
  const [evalData, setEvalData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setIsLoading(true);
        const data = await getEvaluationSummary();
        setEvalData(data);
      } catch (err) {
        console.error('Failed to load evaluation summary:', err);
        setError('Could not load evaluation data. Ensure scripts/run_evaluation.py has been executed.');
      } finally {
        setIsLoading(false);
      }
    }
    fetchData();
  }, []);

  if (isLoading) {
    return (
      <div className="state-center">
        <div className="spinner"></div>
        <p>Loading Information Retrieval benchmark and performance metrics...</p>
      </div>
    );
  }

  if (error || !evalData) {
    return (
      <div className="state-center">
        <p style={{ color: 'var(--accent-rose)' }}>{error}</p>
      </div>
    );
  }

  const { metrics_comparison = [], precision_recall_curves = {}, benchmark_queries = [] } = evalData;

  // Format data for Recharts Bar Chart
  const barChartData = metrics_comparison.map((m) => ({
    name: m.model_name,
    'Precision@10': Number(m.precision_at_10),
    'Recall@10': Number(m.recall),
    MAP: Number(m.map),
    'nDCG@10': Number(m.ndcg_at_10),
  }));

  // Format data for Recharts 11-point Precision-Recall Line Chart
  const recallLevels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  const lineChartData = recallLevels.map((r, idx) => {
    const point = { recall: r };
    Object.entries(precision_recall_curves).forEach(([modelName, curve]) => {
      if (curve && curve[idx]) {
        point[modelName] = curve[idx].precision;
      }
    });
    return point;
  });

  // Identify top score in each column — only compare available models to avoid
  // highlighting unavailable (0.0) rows as "best" when all values are tied at zero
  const availableModels = metrics_comparison.filter((m) => m.is_available);
  const maxP = availableModels.length > 0 ? Math.max(...availableModels.map((m) => m.precision_at_10)) : -1;
  const maxR = availableModels.length > 0 ? Math.max(...availableModels.map((m) => m.recall)) : -1;
  const maxMap = availableModels.length > 0 ? Math.max(...availableModels.map((m) => m.map)) : -1;
  const maxNdcg = availableModels.length > 0 ? Math.max(...availableModels.map((m) => m.ndcg_at_10)) : -1;

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">Model Performance & Evaluation Benchmark</h2>
        <p className="page-description">
          Empirical evaluation across all 4 Information Retrieval models on a fixed, reproducible
          benchmark of 17 test queries with actual Book-Crossing ground truth relevance.
        </p>
      </div>

      <div className="note-box">
        <strong>Evaluation Protocol & Reproducibility: </strong>
        {evalData.protocol_description}
      </div>

      {/* 1. Comparison Table */}
      <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '0.75rem', color: '#fff' }}>
        Performance Comparison Table
      </h3>
      <div className="table-container">
        <table className="custom-table">
          <thead>
            <tr>
              <th>Retrieval Model</th>
              <th>Precision@10</th>
              <th>Recall@10</th>
              <th>MAP</th>
              <th>nDCG@10</th>
              <th>Avg Latency (ms)</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {metrics_comparison.map((m) => (
              <tr key={m.model_name}>
                <td style={{ fontWeight: 600 }}>{m.model_name}</td>
                <td className={m.precision_at_10 === maxP ? 'best-val' : ''}>
                  {m.precision_at_10.toFixed(4)}
                </td>
                <td className={m.recall === maxR ? 'best-val' : ''}>
                  {m.recall.toFixed(4)}
                </td>
                <td className={m.map === maxMap ? 'best-val' : ''}>
                  {m.map.toFixed(4)}
                </td>
                <td className={m.ndcg_at_10 === maxNdcg ? 'best-val' : ''}>
                  {m.ndcg_at_10.toFixed(4)}
                </td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>{m.avg_latency_ms.toFixed(1)} ms</td>
                <td>
                  <span
                    style={{
                      fontSize: '0.75rem',
                      padding: '0.2rem 0.5rem',
                      borderRadius: 4,
                      background: m.is_available ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)',
                      color: m.is_available ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                    }}
                  >
                    {m.is_available ? 'Active' : 'Unavailable'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 2. Charts Grid */}
      <div className="charts-grid">
        {/* Metric Comparison Bar Chart */}
        <div className="chart-card">
          <h4 className="chart-title">Retrieval Metrics Comparison</h4>
          <p className="chart-subtitle">Direct comparison of P@10, Recall, MAP, and nDCG@10 across models</p>
          <div style={{ width: '100%', height: 320 }}>
            <ResponsiveContainer>
              <BarChart data={barChartData} margin={{ top: 10, right: 20, left: -10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="name" stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <YAxis stroke="#9ca3af" domain={[0, 1]} tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151', color: '#fff' }}
                />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 10 }} />
                <Bar dataKey="Precision@10" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Recall@10" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="MAP" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                <Bar dataKey="nDCG@10" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Precision-Recall Curves */}
        <div className="chart-card">
          <h4 className="chart-title">11-Point Precision-Recall Curves</h4>
          <p className="chart-subtitle">Standard interpolated precision at recall levels [0.0 ... 1.0]</p>
          <div style={{ width: '100%', height: 320 }}>
            <ResponsiveContainer>
              <LineChart data={lineChartData} margin={{ top: 10, right: 20, left: -10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="recall"
                  stroke="#9ca3af"
                  label={{ value: 'Recall', position: 'insideBottom', offset: -10, fill: '#9ca3af', fontSize: 11 }}
                  tick={{ fontSize: 11 }}
                />
                <YAxis
                  stroke="#9ca3af"
                  domain={[0, 1.05]}
                  label={{ value: 'Precision', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 11 }}
                  tick={{ fontSize: 11 }}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151', color: '#fff' }}
                />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 10 }} />
                <Line type="monotone" dataKey="TF-IDF" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="BM25" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="Sentence Transformer" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="Hybrid RRF" stroke="#8b5cf6" strokeWidth={3} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* 3. Metrics Definition Section */}
      <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1rem', color: '#fff' }}>
        Algorithmic Formulations & Metric Definitions
      </h3>
      <div className="stats-cards-grid">
        <div className="metric-card">
          <span className="metric-label">Precision@10</span>
          <p style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: 1.4 }}>
            Proportion of top-10 retrieved books that are relevant:
          </p>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--accent-cyan)', marginTop: '0.5rem' }}>
            P@10 = |Rel ∩ Ret@10| / 10
          </div>
        </div>

        <div className="metric-card">
          <span className="metric-label">Recall@10</span>
          <p style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: 1.4 }}>
            Fraction of total ground-truth relevant books retrieved in top-10:
          </p>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--accent-emerald)', marginTop: '0.5rem' }}>
            R@10 = |Rel ∩ Ret@10| / |Rel|
          </div>
        </div>

        <div className="metric-card">
          <span className="metric-label">Mean Average Precision (MAP)</span>
          <p style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: 1.4 }}>
            Macro-average of Average Precision across all benchmark test queries:
          </p>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--accent-amber)', marginTop: '0.5rem' }}>
            MAP = (1/|Q|) ∑ (1/|R_q|) ∑ P@k · rel(k)
          </div>
        </div>

        <div className="metric-card">
          <span className="metric-label">nDCG@10</span>
          <p style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: 1.4 }}>
            Normalized Discounted Cumulative Gain with logarithmic position penalty:
          </p>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--accent-indigo)', marginTop: '0.5rem' }}>
            nDCG@10 = DCG@10 / IDCG@10
          </div>
        </div>
      </div>

      {/* 4. Benchmark Query Suite Table */}
      <h3 style={{ fontSize: '1.25rem', fontWeight: 700, margin: '1.5rem 0 0.75rem 0', color: '#fff' }}>
        Benchmark Test Query Suite ({benchmark_queries.length} Queries)
      </h3>
      <div className="table-container">
        <table className="custom-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Test Query</th>
              <th>Category</th>
              <th>Relevant Books in Catalog</th>
              <th>Sample ISBNs</th>
            </tr>
          </thead>
          <tbody>
            {benchmark_queries.map((q) => (
              <tr key={q.query_id}>
                <td style={{ fontFamily: 'var(--font-mono)' }}>{q.query_id}</td>
                <td style={{ fontWeight: 600 }}>{q.query_text}</td>
                <td>
                  <span
                    style={{
                      background: 'var(--bg-card)',
                      border: '1px solid var(--border-subtle)',
                      padding: '0.2rem 0.5rem',
                      borderRadius: 4,
                      fontSize: '0.75rem',
                    }}
                  >
                    {q.category}
                  </span>
                </td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>{q.num_relevant_docs}</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  {q.sample_relevant_isbns.join(', ')}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="note-box" style={{ borderColor: 'var(--accent-cyan)', marginTop: '1.5rem' }}>
        <strong>Dataset Limitations Disclosure: </strong>
        {evalData.limitations_notes}
      </div>
    </div>
  );
}

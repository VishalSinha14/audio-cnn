import { useState, useEffect } from 'react';
import { getTrainingMetrics, type MetricPoint } from '../api';

// Simple inline SVG chart component (no external dependency needed)
function LineChart({ data, color, label, formatValue }: {
  data: MetricPoint[];
  color: string;
  label: string;
  formatValue?: (v: number) => string;
}) {
  if (data.length === 0) {
    return (
      <div className="empty-state" style={{ padding: '2rem' }}>
        <div>No {label} data available</div>
      </div>
    );
  }

  const width = 600;
  const height = 200;
  const padding = { top: 20, right: 20, bottom: 30, left: 50 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  const minX = Math.min(...data.map(d => d.step));
  const maxX = Math.max(...data.map(d => d.step));
  const minY = Math.min(...data.map(d => d.value));
  const maxY = Math.max(...data.map(d => d.value));
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const padY = rangeY * 0.1;

  const scaleX = (v: number) => padding.left + ((v - minX) / rangeX) * chartW;
  const scaleY = (v: number) => padding.top + chartH - ((v - (minY - padY)) / (rangeY + 2 * padY)) * chartH;

  const pathD = data.map((d, i) =>
    `${i === 0 ? 'M' : 'L'} ${scaleX(d.step).toFixed(1)} ${scaleY(d.value).toFixed(1)}`
  ).join(' ');

  // Area fill path
  const areaD = pathD +
    ` L ${scaleX(data[data.length - 1].step).toFixed(1)} ${scaleY(minY - padY).toFixed(1)}` +
    ` L ${scaleX(data[0].step).toFixed(1)} ${scaleY(minY - padY).toFixed(1)} Z`;

  // Y-axis ticks
  const yTicks = 5;
  const yTickValues = Array.from({ length: yTicks }, (_, i) =>
    minY - padY + (rangeY + 2 * padY) * (i / (yTicks - 1))
  );

  const fmt = formatValue || ((v: number) => v.toFixed(2));

  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto' }}>
      {/* Grid lines */}
      {yTickValues.map((v, i) => (
        <g key={i}>
          <line
            x1={padding.left} y1={scaleY(v)}
            x2={width - padding.right} y2={scaleY(v)}
            stroke="rgba(255,255,255,0.05)" strokeWidth="1"
          />
          <text
            x={padding.left - 8} y={scaleY(v) + 4}
            textAnchor="end" fill="rgba(255,255,255,0.3)"
            fontSize="10" fontFamily="Inter"
          >
            {fmt(v)}
          </text>
        </g>
      ))}

      {/* Area fill */}
      <path d={areaD} fill={color} opacity="0.08" />

      {/* Line */}
      <path d={pathD} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />

      {/* Last point dot */}
      {data.length > 0 && (
        <circle
          cx={scaleX(data[data.length - 1].step)}
          cy={scaleY(data[data.length - 1].value)}
          r="4" fill={color}
        />
      )}

      {/* X-axis label */}
      <text
        x={width / 2} y={height - 5}
        textAnchor="middle" fill="rgba(255,255,255,0.3)"
        fontSize="10" fontFamily="Inter"
      >
        Epoch
      </text>
    </svg>
  );
}

export default function TrainingDashboard() {
  const [metrics, setMetrics] = useState<{
    train_loss: MetricPoint[];
    val_loss: MetricPoint[];
    val_accuracy: MetricPoint[];
    learning_rate: MetricPoint[];
  } | null>(null);
  const [hasData, setHasData] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchMetrics = async () => {
    try {
      const res = await getTrainingMetrics();
      setMetrics(res.metrics);
      setHasData(res.has_data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load metrics');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  if (loading) {
    return (
      <div>
        <div className="page-header">
          <h1 className="page-title">Training Metrics</h1>
          <p className="page-subtitle">Model training history and performance</p>
        </div>
        <div className="card">
          <div className="loading-spinner" />
          <div className="loading-text">Loading metrics...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <div className="page-header">
          <h1 className="page-title">Training Metrics</h1>
          <p className="page-subtitle">Model training history and performance</p>
        </div>
        <div className="card">
          <div className="empty-state">
            <div className="empty-state-icon">⚠️</div>
            <div>{error}</div>
            <div style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              Make sure the backend server is running (python main.py)
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!hasData || !metrics) {
    return (
      <div>
        <div className="page-header">
          <h1 className="page-title">Training Metrics</h1>
          <p className="page-subtitle">Model training history and performance</p>
        </div>
        <div className="card">
          <div className="empty-state">
            <div className="empty-state-icon">📉</div>
            <div>No training data found</div>
            <div style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              Run <code>python train.py</code> to train the model and generate metrics
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Compute summary stats
  const lastTrainLoss = metrics.train_loss[metrics.train_loss.length - 1]?.value;
  const lastValLoss = metrics.val_loss[metrics.val_loss.length - 1]?.value;
  const bestAccuracy = Math.max(...metrics.val_accuracy.map(m => m.value), 0);
  const totalEpochs = Math.max(
    ...metrics.train_loss.map(m => m.step),
    ...metrics.val_loss.map(m => m.step),
    0
  ) + 1;

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Training Metrics</h1>
        <p className="page-subtitle">Model training history and performance</p>
      </div>

      {/* Summary stats */}
      <div className="metrics-stat-grid">
        <div className="metric-stat">
          <div className="metric-stat-label">Epochs</div>
          <div className="metric-stat-value">{totalEpochs}</div>
        </div>
        <div className="metric-stat">
          <div className="metric-stat-label">Best Accuracy</div>
          <div className="metric-stat-value">{bestAccuracy.toFixed(1)}%</div>
        </div>
        <div className="metric-stat">
          <div className="metric-stat-label">Train Loss</div>
          <div className="metric-stat-value">{lastTrainLoss?.toFixed(3) ?? '—'}</div>
        </div>
        <div className="metric-stat">
          <div className="metric-stat-label">Val Loss</div>
          <div className="metric-stat-value">{lastValLoss?.toFixed(3) ?? '—'}</div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid-2">
        <div className="card">
          <div className="card-header">
            <div className="card-title"><span>📉</span> Training Loss</div>
          </div>
          <LineChart data={metrics.train_loss} color="#6366f1" label="training loss" />
        </div>
        <div className="card">
          <div className="card-header">
            <div className="card-title"><span>📉</span> Validation Loss</div>
          </div>
          <LineChart data={metrics.val_loss} color="#8b5cf6" label="validation loss" />
        </div>
        <div className="card grid-full">
          <div className="card-header">
            <div className="card-title"><span>📈</span> Validation Accuracy</div>
          </div>
          <LineChart
            data={metrics.val_accuracy}
            color="#22c55e"
            label="accuracy"
            formatValue={(v) => `${v.toFixed(1)}%`}
          />
        </div>
      </div>

      <div style={{ marginTop: 'var(--space-md)', textAlign: 'center' }}>
        <button className="btn btn-secondary" onClick={fetchMetrics}>
          🔄 Refresh Metrics
        </button>
      </div>
    </div>
  );
}

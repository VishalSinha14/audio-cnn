import type { Prediction } from '../api';

interface ClassificationResultsProps {
  predictions: Prediction[];
  audioInfo?: {
    filename: string;
    duration: number;
  };
}

export default function ClassificationResults({ predictions, audioInfo }: ClassificationResultsProps) {
  if (predictions.length === 0) {
    return (
      <div className="card">
        <div className="card-header">
          <div className="card-title"><span>🏷️</span> Classification</div>
        </div>
        <div className="empty-state">
          <div className="empty-state-icon">📊</div>
          <div>Upload or record audio to see predictions</div>
        </div>
      </div>
    );
  }

  const maxConfidence = Math.max(...predictions.map(p => p.confidence));

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title"><span>🏷️</span> Classification</div>
        {audioInfo && (
          <span className="card-badge">{audioInfo.duration.toFixed(1)}s</span>
        )}
      </div>

      <div className="results-list" key={predictions.map(p => p.class).join(',')}>
        {predictions.map((pred, i) => (
          <div className="result-item" key={pred.class}>
            <span className="result-rank">#{i + 1}</span>
            <span className="result-class">{pred.class}</span>
            <div className="result-bar-container">
              <div
                className="result-bar"
                style={{ width: `${(pred.confidence / maxConfidence) * 100}%` }}
              />
            </div>
            <span className="result-confidence">
              {(pred.confidence * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

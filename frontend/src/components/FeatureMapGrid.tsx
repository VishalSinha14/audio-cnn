import { useRef, useEffect } from 'react';
import type { FeatureMapData } from '../api';

interface FeatureMapGridProps {
  data: FeatureMapData | null;
}

function viridisColor(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.min(1, Math.max(0,
    t < 0.25 ? 0.267 + t * 4 * 0.01 :
    t < 0.5 ? 0.277 + (t - 0.25) * 4 * 0.05 :
    t < 0.75 ? 0.327 + (t - 0.5) * 4 * 0.45 :
    0.777 + (t - 0.75) * 4 * 0.22
  )));
  const g = Math.round(255 * Math.min(1, Math.max(0,
    t < 0.25 ? 0.004 + t * 4 * 0.17 :
    t < 0.5 ? 0.174 + (t - 0.25) * 4 * 0.25 :
    t < 0.75 ? 0.424 + (t - 0.5) * 4 * 0.27 :
    0.694 + (t - 0.75) * 4 * 0.24
  )));
  const b = Math.round(255 * Math.min(1, Math.max(0,
    t < 0.25 ? 0.329 + t * 4 * 0.13 :
    t < 0.5 ? 0.459 - (t - 0.25) * 4 * 0.05 :
    t < 0.75 ? 0.409 - (t - 0.5) * 4 * 0.23 :
    0.179 - (t - 0.75) * 4 * 0.15
  )));
  return [r, g, b];
}

function FeatureMapCanvas({ values, shape }: { values: number[][]; shape: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const [height, width] = shape;
    canvas.width = width;
    canvas.height = height;

    const imageData = ctx.createImageData(width, height);

    let min = Infinity, max = -Infinity;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const val = values[y]?.[x] ?? 0;
        if (val < min) min = val;
        if (val > max) max = val;
      }
    }

    const range = max - min || 1;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const t = ((values[y]?.[x] ?? 0) - min) / range;
        const [r, g, b] = viridisColor(t);
        const idx = (y * width + x) * 4;
        imageData.data[idx] = r;
        imageData.data[idx + 1] = g;
        imageData.data[idx + 2] = b;
        imageData.data[idx + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [values, shape]);

  return (
    <canvas
      ref={canvasRef}
      className="viz-canvas"
      style={{ imageRendering: 'pixelated' }}
    />
  );
}

export default function FeatureMapGrid({ data }: FeatureMapGridProps) {
  if (!data || Object.keys(data).length === 0) {
    return (
      <div className="card">
        <div className="card-header">
          <div className="card-title"><span>🧠</span> Feature Maps</div>
        </div>
        <div className="empty-state">
          <div className="empty-state-icon">🧠</div>
          <div>No feature map data yet</div>
        </div>
      </div>
    );
  }

  const entries = Object.entries(data);

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title"><span>🧠</span> Feature Maps</div>
        <span className="card-badge">{entries.length} layers</span>
      </div>
      <div className="feature-map-grid">
        {entries.map(([name, fm]) => (
          <div className="feature-map-item" key={name}>
            <div className="feature-map-label">
              {name} <span style={{ opacity: 0.5 }}>({fm.shape[0]}×{fm.shape[1]})</span>
            </div>
            <FeatureMapCanvas values={fm.values} shape={fm.shape} />
          </div>
        ))}
      </div>
    </div>
  );
}

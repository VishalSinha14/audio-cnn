import { useRef, useEffect } from 'react';
import type { SpectrogramData } from '../api';

interface SpectrogramViewProps {
  data: SpectrogramData | null;
  title?: string;
}

// Inferno-like colormap for spectrograms
function infernoColor(t: number): [number, number, number] {
  // Simplified inferno colormap
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.min(1, Math.max(0,
    t < 0.25 ? t * 4 * 0.18 :
    t < 0.5 ? 0.18 + (t - 0.25) * 4 * 0.62 :
    t < 0.75 ? 0.8 + (t - 0.5) * 4 * 0.2 :
    1.0 - (t - 0.75) * 4 * 0.05
  )));
  const g = Math.round(255 * Math.min(1, Math.max(0,
    t < 0.25 ? t * 4 * 0.02 :
    t < 0.5 ? 0.02 + (t - 0.25) * 4 * 0.15 :
    t < 0.75 ? 0.17 + (t - 0.5) * 4 * 0.55 :
    0.72 + (t - 0.75) * 4 * 0.27
  )));
  const b = Math.round(255 * Math.min(1, Math.max(0,
    t < 0.25 ? 0.015 + t * 4 * 0.42 :
    t < 0.5 ? 0.435 - (t - 0.25) * 4 * 0.2 :
    t < 0.75 ? 0.235 - (t - 0.5) * 4 * 0.18 :
    0.055 + (t - 0.75) * 4 * 0.9
  )));
  return [r, g, b];
}

export default function SpectrogramView({ data, title = 'Mel Spectrogram' }: SpectrogramViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!data || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const [height, width] = data.shape;
    canvas.width = width;
    canvas.height = height;

    const imageData = ctx.createImageData(width, height);

    // Find min/max for normalization
    let min = Infinity, max = -Infinity;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const val = data.values[y][x];
        if (val < min) min = val;
        if (val > max) max = val;
      }
    }

    const range = max - min || 1;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const t = (data.values[y][x] - min) / range;
        const [r, g, b] = infernoColor(t);
        const flippedY = height - 1 - y; // flip vertically (low freq at bottom)
        const idx = (flippedY * width + x) * 4;
        imageData.data[idx] = r;
        imageData.data[idx + 1] = g;
        imageData.data[idx + 2] = b;
        imageData.data[idx + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [data]);

  if (!data) {
    return (
      <div className="card">
        <div className="card-header">
          <div className="card-title"><span>📊</span> {title}</div>
        </div>
        <div className="empty-state">
          <div className="empty-state-icon">🎵</div>
          <div>No spectrogram data yet</div>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title"><span>📊</span> {title}</div>
        <span className="card-badge">{data.shape[0]}×{data.shape[1]}</span>
      </div>
      <div className="viz-canvas-container">
        <canvas
          ref={canvasRef}
          className="viz-canvas"
          style={{ imageRendering: 'pixelated' }}
        />
        <span className="viz-label">Frequency ↑ | Time →</span>
      </div>
    </div>
  );
}

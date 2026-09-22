import { useRef, useEffect } from 'react';
import type { WaveformData } from '../api';

interface WaveformViewProps {
  data: WaveformData | null;
}

export default function WaveformView({ data }: WaveformViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!data || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const values = data.values;
    const midY = h / 2;

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Background grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    for (let y = 0; y < h; y += h / 4) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Center line
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(w, midY);
    ctx.stroke();

    // Waveform
    const gradient = ctx.createLinearGradient(0, 0, w, 0);
    gradient.addColorStop(0, '#6366f1');
    gradient.addColorStop(0.5, '#8b5cf6');
    gradient.addColorStop(1, '#06b6d4');

    ctx.strokeStyle = gradient;
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    const step = w / values.length;

    for (let i = 0; i < values.length; i++) {
      const x = i * step;
      const y = midY - values[i] * midY * 0.85;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    ctx.stroke();

    // Filled waveform (subtle)
    ctx.globalAlpha = 0.08;
    ctx.fillStyle = gradient;
    ctx.lineTo(w, midY);
    ctx.lineTo(0, midY);
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1;

  }, [data]);

  if (!data) {
    return (
      <div className="card">
        <div className="card-header">
          <div className="card-title"><span>〰️</span> Waveform</div>
        </div>
        <div className="empty-state">
          <div className="empty-state-icon">〰️</div>
          <div>No waveform data yet</div>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title"><span>〰️</span> Waveform</div>
        <span className="card-badge">{data.duration.toFixed(1)}s • {data.sample_rate}Hz</span>
      </div>
      <div className="viz-canvas-container" style={{ height: '120px' }}>
        <canvas
          ref={canvasRef}
          className="viz-canvas"
          style={{ width: '100%', height: '120px' }}
        />
      </div>
    </div>
  );
}

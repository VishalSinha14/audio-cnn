import { useState, useEffect, useCallback } from 'react';
import AudioUploader from './components/AudioUploader';
import ClassificationResults from './components/ClassificationResults';
import SpectrogramView from './components/SpectrogramView';
import WaveformView from './components/WaveformView';
import FeatureMapGrid from './components/FeatureMapGrid';
import TrainingDashboard from './components/TrainingDashboard';
import {
  classifyAudio,
  getHealth,
  type ClassifyResponse,
  type HealthResponse,
} from './api';

type Page = 'classify' | 'training';

function App() {
  const [page, setPage] = useState<Page>('classify');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ClassifyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const handleFileSelected = useCallback(async (file: File | Blob) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await classifyAudio(file);
      setResult(response);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Classification failed');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const isOnline = health?.status === 'ok';

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <nav className="sidebar">
        <div className="sidebar-logo">
          <div className="sidebar-logo-icon">🔊</div>
          <span className="sidebar-logo-text">Audio CNN</span>
        </div>

        <div className="sidebar-section-label">Analysis</div>
        <div
          className={`sidebar-nav-item ${page === 'classify' ? 'active' : ''}`}
          onClick={() => setPage('classify')}
          id="nav-classify"
        >
          <span className="sidebar-nav-icon">🎵</span>
          Classify Audio
        </div>

        <div className="sidebar-section-label">Training</div>
        <div
          className={`sidebar-nav-item ${page === 'training' ? 'active' : ''}`}
          onClick={() => setPage('training')}
          id="nav-training"
        >
          <span className="sidebar-nav-icon">📈</span>
          Training Metrics
        </div>

        {/* Status */}
        <div className="sidebar-status">
          <div>
            <span className={`sidebar-status-dot ${isOnline ? 'online' : 'offline'}`} />
            <span style={{ fontWeight: 500 }}>
              {isOnline ? 'Backend Online' : 'Backend Offline'}
            </span>
          </div>
          {health?.cuda_available && (
            <div style={{ marginTop: '0.4rem', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
              🚀 {health.gpu}
            </div>
          )}
          {health && !health.checkpoint_exists && (
            <div style={{ marginTop: '0.4rem', color: 'var(--accent-warning)', fontSize: '0.75rem' }}>
              ⚠️ No trained model found
            </div>
          )}
        </div>
      </nav>

      {/* Main content */}
      <main className="main-content">
        {page === 'classify' && (
          <ClassifyPage
            onFileSelected={handleFileSelected}
            isLoading={isLoading}
            result={result}
            error={error}
          />
        )}
        {page === 'training' && <TrainingDashboard />}
      </main>
    </div>
  );
}

// ─── Classify Page ──────────────────────────────────────────

interface ClassifyPageProps {
  onFileSelected: (file: File | Blob) => void;
  isLoading: boolean;
  result: ClassifyResponse | null;
  error: string | null;
}

function ClassifyPage({ onFileSelected, isLoading, result, error }: ClassifyPageProps) {
  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Classify Audio</h1>
        <p className="page-subtitle">
          Upload or record audio to classify environmental sounds using the CNN model
        </p>
      </div>

      {/* Upload + Results row */}
      <div className="grid-2">
        <AudioUploader onFileSelected={onFileSelected} isLoading={isLoading} />
        <ClassificationResults
          predictions={result?.predictions ?? []}
          audioInfo={result?.audio_info}
        />
      </div>

      {error && (
        <div className="card" style={{ marginTop: 'var(--space-lg)', borderColor: 'var(--accent-danger)' }}>
          <div style={{ color: 'var(--accent-danger)', fontWeight: 500 }}>
            ⚠️ {error}
          </div>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.25rem' }}>
            Make sure the backend is running: <code>python main.py</code>
          </div>
        </div>
      )}

      {/* Visualizations */}
      {result && (
        <>
          <div className="grid-2" style={{ marginTop: 'var(--space-lg)' }}>
            <WaveformView data={result.waveform} />
            <SpectrogramView data={result.input_spectrogram} />
          </div>

          <div style={{ marginTop: 'var(--space-lg)' }}>
            <FeatureMapGrid data={result.feature_maps} />
          </div>
        </>
      )}
    </div>
  );
}

export default App;

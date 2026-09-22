const API_BASE = 'http://localhost:8000';

export interface Prediction {
  class: string;
  confidence: number;
}

export interface SpectrogramData {
  shape: number[];
  values: number[][];
}

export interface FeatureMapData {
  [key: string]: {
    shape: number[];
    values: number[][];
  };
}

export interface WaveformData {
  values: number[];
  sample_rate: number;
  duration: number;
}

export interface AudioInfo {
  filename: string;
  original_sample_rate: number;
  duration: number;
}

export interface ClassifyResponse {
  predictions: Prediction[];
  input_spectrogram: SpectrogramData;
  feature_maps: FeatureMapData;
  waveform: WaveformData;
  audio_info: AudioInfo;
}

export interface HealthResponse {
  status: string;
  cuda_available: boolean;
  gpu: string | null;
  model_loaded: boolean;
  checkpoint_exists: boolean;
}

export interface ModelInfoResponse {
  num_classes: number;
  classes: string[];
  total_params: number;
  trainable_params: number;
  best_accuracy: number;
  best_epoch: number;
  device: string;
}

export interface MetricPoint {
  step: number;
  value: number;
}

export interface TrainingMetricsResponse {
  has_data: boolean;
  metrics: {
    train_loss: MetricPoint[];
    val_loss: MetricPoint[];
    val_accuracy: MetricPoint[];
    learning_rate: MetricPoint[];
  };
}

export async function classifyAudio(file: File | Blob): Promise<ClassifyResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/api/classify`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Classification failed');
  }

  return res.json();
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error('Backend not reachable');
  return res.json();
}

export async function getModelInfo(): Promise<ModelInfoResponse> {
  const res = await fetch(`${API_BASE}/api/model-info`);
  if (!res.ok) throw new Error('Could not fetch model info');
  return res.json();
}

export async function getTrainingMetrics(): Promise<TrainingMetricsResponse> {
  const res = await fetch(`${API_BASE}/api/training-metrics`);
  if (!res.ok) throw new Error('Could not fetch training metrics');
  return res.json();
}

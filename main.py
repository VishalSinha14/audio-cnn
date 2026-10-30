"""
FastAPI backend for Audio-CNN inference and training metrics.

Usage:
    python main.py

Serves at http://localhost:8000
API docs at http://localhost:8000/docs
"""

import io
import os
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
import soundfile as sf
import librosa
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model import AudioCNN

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

CHECKPOINT_PATH = Path("./checkpoints/best_model.pth")
TENSORBOARD_DIR = Path("./runs")

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
F_MIN = 0
F_MAX = 11025

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────

app = FastAPI(
    title="Audio-CNN API",
    description="Environmental Sound Classification using a ResNet-style CNN",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Audio processor
# ──────────────────────────────────────────────

class AudioProcessor:
    def __init__(self, device):
        self.device = device
        self.transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
                f_min=F_MIN,
                f_max=F_MAX
            ),
            T.AmplitudeToDB()
        ).to(device)

    def process(self, audio_data: np.ndarray):
        waveform = torch.from_numpy(audio_data).float().to(self.device)
        waveform = waveform.unsqueeze(0)
        spectrogram = self.transform(waveform)
        return spectrogram.unsqueeze(0)


# ──────────────────────────────────────────────
# Model loading (lazy, on first request)
# ──────────────────────────────────────────────

_model = None
_classes = None
_device = None
_processor = None


def get_model():
    global _model, _classes, _device, _processor

    if _model is not None:
        return _model, _classes, _device, _processor

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not CHECKPOINT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model checkpoint not found at {CHECKPOINT_PATH}. Run `python train.py` first."
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=_device)
    _classes = checkpoint['classes']

    _model = AudioCNN(num_classes=len(_classes))
    _model.load_state_dict(checkpoint['model_state_dict'])
    _model.to(_device)
    _model.eval()

    _processor = AudioProcessor(_device)

    gpu_info = torch.cuda.get_device_name(0) if _device.type == 'cuda' else 'CPU'
    print(f"[OK] Model loaded on {gpu_info}")
    print(f"  Classes: {len(_classes)}, Epoch: {checkpoint.get('epoch', '?')}, Accuracy: {checkpoint.get('accuracy', '?'):.2f}%")

    return _model, _classes, _device, _processor


# ──────────────────────────────────────────────
# TensorBoard log parser
# ──────────────────────────────────────────────

def parse_tensorboard_logs(log_dir: Path):
    """Parse TensorBoard event files and return metrics as dicts."""
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    tag_map = {
        "Loss/Train": "train_loss",
        "Loss/Validation": "val_loss",
        "Accuracy/Validation": "val_accuracy",
        "Learning_Rate": "learning_rate",
    }

    # Find the latest run directory
    if not log_dir.exists():
        return metrics

    run_dirs = sorted(log_dir.iterdir(), reverse=True)
    if not run_dirs:
        return metrics

    latest_run = run_dirs[0]

    # Find event files
    event_files = list(latest_run.glob("events.out.tfevents.*"))
    if not event_files:
        return metrics

    try:
        # Minimal TensorBoard event file parser
        for event_file in event_files:
            _parse_event_file(event_file, metrics, tag_map)
    except Exception as e:
        print(f"Warning: Could not parse TensorBoard logs: {e}")

    # Sort by step
    for key in metrics:
        metrics[key].sort(key=lambda x: x["step"])

    return metrics


def _parse_event_file(filepath: Path, metrics: dict, tag_map: dict):
    """Parse a single TensorBoard event file."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        ea = EventAccumulator(str(filepath.parent))
        ea.Reload()

        for tag in ea.Tags().get("scalars", []):
            if tag in tag_map:
                key = tag_map[tag]
                for event in ea.Scalars(tag):
                    metrics[key].append({
                        "step": event.step,
                        "value": event.value,
                    })
    except ImportError:
        # Fallback: no tensorboard package, return empty
        pass


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Health check."""
    cuda_available = torch.cuda.is_available()
    return {
        "status": "ok",
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "model_loaded": _model is not None,
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
    }


@app.get("/api/model-info")
async def model_info():
    """Return model architecture info and class list."""
    model, classes, device, _ = get_model()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    return {
        "num_classes": len(classes),
        "classes": classes,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_accuracy": checkpoint.get("accuracy"),
        "best_epoch": checkpoint.get("epoch"),
        "device": str(device),
    }


@app.post("/api/classify")
async def classify(file: UploadFile = File(...)):
    """
    Classify an audio file.
    Returns predictions, spectrogram, feature maps, and waveform data.
    """
    model, classes, device, processor = get_model()

    # Read audio
    try:
        audio_bytes = await file.read()
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read audio file: {e}")

    # Convert to mono
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)


    # Resample to the model's training sample rate
    original_sr = sample_rate

    if sample_rate != SAMPLE_RATE:
        audio_data = librosa.resample(
            y=audio_data,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE
        )
        sample_rate = SAMPLE_RATE


    # Generate spectrogram and classify
    spectrogram = processor.process(audio_data)

    with torch.no_grad():
        output, feature_maps = model(spectrogram, return_feature_maps=True)

        output = torch.nan_to_num(output)
        probabilities = torch.softmax(output, dim=1)
        top_k = min(5, len(classes))
        top_probs, top_indices = torch.topk(probabilities[0], top_k)

        predictions = [
            {"class": classes[idx.item()], "confidence": round(prob.item(), 4)}
            for prob, idx in zip(top_probs, top_indices)
        ]

        # Feature maps — aggregate channels and downsample for JSON transfer
        viz_data = {}
        for name, tensor in feature_maps.items():
            if tensor.dim() == 4:
                aggregated = torch.mean(tensor, dim=1).squeeze(0)
                arr = aggregated.cpu().numpy()
                arr = np.nan_to_num(arr)
                # Downsample large feature maps for transfer
                max_dim = 64
                if arr.shape[0] > max_dim or arr.shape[1] > max_dim:
                    from PIL import Image
                    img = Image.fromarray(arr)
                    img = img.resize((min(arr.shape[1], max_dim), min(arr.shape[0], max_dim)))
                    arr = np.array(img)
                viz_data[name] = {
                    "shape": list(arr.shape),
                    "values": arr.tolist()
                }

        # Spectrogram data
        spec_np = spectrogram.squeeze(0).squeeze(0).cpu().numpy()
        spec_np = np.nan_to_num(spec_np)

        # Waveform — downsample for display
        max_samples = 8000
        if len(audio_data) > max_samples:
            step = len(audio_data) // max_samples
            waveform_data = audio_data[::step].tolist()
        else:
            waveform_data = audio_data.tolist()

    return {
        "predictions": predictions,
        "input_spectrogram": {
            "shape": list(spec_np.shape),
            "values": spec_np.tolist()
        },
        "feature_maps": viz_data,
        "waveform": {
            "values": waveform_data,
            "sample_rate": sample_rate,
            "duration": round(len(audio_data) / sample_rate, 2)
        },
        "audio_info": {
            "filename": file.filename,
            "original_sample_rate": original_sr,
            "duration": round(len(audio_data) / sample_rate, 2),
        }
    }


@app.get("/api/training-metrics")
async def training_metrics():
    """Return training metrics from TensorBoard logs."""
    metrics = parse_tensorboard_logs(TENSORBOARD_DIR)
    has_data = any(len(v) > 0 for v in metrics.values())
    return {
        "has_data": has_data,
        "metrics": metrics
    }


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Audio-CNN API Server")
    print("=" * 50)
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Checkpoint: {CHECKPOINT_PATH} ({'found' if CHECKPOINT_PATH.exists() else 'NOT FOUND'})")
    print(f"  API docs: http://localhost:8000/docs")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000)
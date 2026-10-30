# Audio-CNN — Environmental Sound Classification

A full-stack deep learning project for classifying environmental sounds using a **ResNet-34 style CNN** trained on the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset (50 classes, 2000 audio clips).

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  React + TypeScript Frontend (Vite, port 5173)       │
│  - Audio upload & microphone recording               │
│  - Waveform & spectrogram visualization              │
│  - CNN feature map heatmaps                          │
│  - Training metrics dashboard                        │
└──────────────────┬───────────────────────────────────┘
                   │ HTTP API
┌──────────────────▼───────────────────────────────────┐
│  FastAPI Backend (Python, port 8000)                  │
│  - /api/classify — audio classification               │
│  - /api/model-info — model architecture info          │
│  - /api/training-metrics — TensorBoard log parsing    │
│  - /api/health — system status                        │
└──────────────────┬───────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────┐
│  AudioCNN Model (PyTorch + CUDA)                      │
│  - ResNet-34 adapted for mel spectrograms             │
│  - 50-class environmental sound classification        │
│  - ~21.3M parameters                                  │
└──────────────────────────────────────────────────────┘
```

## Results

| Metric       | Value   |
|-------------|---------|
| Accuracy    | 84.75%  |
| Macro F1    | 0.8415  |
| Weighted F1 | 0.8415  |
| Best Epoch  | 88/100  |

## Prerequisites

- Python 3.10+
- Node.js 18+
- NVIDIA GPU with CUDA support (recommended)

## Setup

### 1. Clone & create virtual environment

```bash
git clone https://github.com/<your-username>/audio-cnn.git
cd audio-cnn
python -m venv .venv
```

### 2. Activate virtual environment

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

### 3. Install PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

> For CPU-only: `pip install torch torchvision torchaudio`

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

## Usage

### Train the model

```bash
python train.py
```

The script will:
- Automatically download the ESC-50 dataset (~600 MB)
- Train for 100 epochs with mixup augmentation, label smoothing, and OneCycleLR
- Save the best model to `checkpoints/best_model.pth`
- Log metrics to `runs/` (viewable with `tensorboard --logdir runs`)

### Run the full-stack app

**Terminal 1 — Backend:**
```bash
# .\.venv\Scripts\python.exe main.py
python main.py
# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
# Open http://localhost:5173
```

### CLI inference

```bash
python inference.py path/to/audio.wav
```

### Evaluate the trained model

```bash
python evaluate.py
```

Outputs confusion matrix, classification report, and per-class accuracy to `evaluation/`.

## Project Structure

```
audio-cnn/
├── model.py              # ResNet-34 AudioCNN architecture
├── train.py              # Local GPU training script (ESC-50)
├── main.py               # FastAPI backend server
├── inference.py          # CLI inference script
├── evaluate.py           # Model evaluation with metrics & plots
├── requirements.txt      # Python dependencies
├── frontend/             # React + TypeScript + Vite
│   ├── src/
│   │   ├── App.tsx                        # Main app with sidebar navigation
│   │   ├── api.ts                         # Type-safe API client
│   │   ├── index.css                      # Design system & styles
│   │   └── components/
│   │       ├── AudioUploader.tsx          # Drag & drop + mic recording
│   │       ├── ClassificationResults.tsx  # Top-K prediction display
│   │       ├── WaveformView.tsx           # Audio waveform canvas
│   │       ├── SpectrogramView.tsx        # Mel spectrogram heatmap
│   │       ├── FeatureMapGrid.tsx         # CNN layer activations
│   │       └── TrainingDashboard.tsx      # Loss/accuracy charts
│   ├── package.json
│   └── vite.config.ts
├── checkpoints/          # Trained model weights (gitignored)
├── data/                 # ESC-50 dataset (gitignored)
├── runs/                 # TensorBoard logs (gitignored)
└── evaluation/           # Evaluation outputs (gitignored)
```

## Tech Stack

- **Model**: PyTorch, torchaudio
- **Audio processing**: librosa, soundfile
- **Backend**: FastAPI, uvicorn
- **Frontend**: React, TypeScript, Vite
- **Training monitoring**: TensorBoard
- **Evaluation**: scikit-learn, matplotlib

## License

MIT

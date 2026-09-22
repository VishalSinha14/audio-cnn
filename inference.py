"""
Local inference script for AudioCNN.
Usage:
    python inference.py <audio_file.wav>
    python inference.py chirpingbirds.wav
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
import soundfile as sf
import librosa

from model import AudioCNN

CHECKPOINT_PATH = Path("./checkpoints/best_model.pth")

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
F_MIN = 0
F_MAX = 11025


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

    def process(self, audio_data):
        waveform = torch.from_numpy(audio_data).float().to(self.device)
        waveform = waveform.unsqueeze(0)   # add channel dim
        spectrogram = self.transform(waveform)
        return spectrogram.unsqueeze(0)    # add batch dim


def load_model(device):
    """Load the trained model from checkpoint."""
    if not CHECKPOINT_PATH.exists():
        print(f"✗ Checkpoint not found at {CHECKPOINT_PATH}")
        print("  Run `python train.py` first to train the model.")
        sys.exit(1)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    classes = checkpoint['classes']
    epoch = checkpoint.get('epoch', '?')
    accuracy = checkpoint.get('accuracy', '?')

    model = AudioCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded (epoch {epoch}, val accuracy: {accuracy:.2f}%)")
    print(f"  Classes: {len(classes)}")
    return model, classes


def classify_audio(audio_path: str):
    """Classify an audio file and print results."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"✗ Audio file not found: {audio_path}")
        sys.exit(1)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model, classes = load_model(device)
    processor = AudioProcessor(device)

    # Load audio
    audio_data, sample_rate = sf.read(str(audio_path), dtype="float32")

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)



    original_sr = sample_rate

    if sample_rate != SAMPLE_RATE:
        audio_data = librosa.resample(
            y=audio_data,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE
        )
        sample_rate = SAMPLE_RATE


    duration = len(audio_data) / SAMPLE_RATE


    print(
    f"Audio: {audio_path.name} "
    f"({duration:.1f}s, {original_sr}Hz → {sample_rate}Hz)")
    
    # Inference
    spectrogram = processor.process(audio_data)

    with torch.no_grad():
        output = model(spectrogram)
        output = torch.nan_to_num(output)
        probabilities = torch.softmax(output, dim=1)
        top5_probs, top5_indices = torch.topk(probabilities[0], min(5, len(classes)))

    # Results
    print(f"\n{'─'*40}")
    print(f"  Top predictions:")
    print(f"{'─'*40}")
    for rank, (prob, idx) in enumerate(zip(top5_probs, top5_indices), 1):
        class_name = classes[idx.item()]
        confidence = prob.item()
        bar_len = int(confidence * 30)
        bar = '█' * bar_len + '░' * (30 - bar_len)
        print(f"  {rank}. {class_name:<20s} [{bar}] {confidence:6.2%}")
    print(f"{'─'*40}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <audio_file.wav>")
        print("Example: python inference.py chirpingbirds.wav")
        sys.exit(1)

    classify_audio(sys.argv[1])

"""
Local GPU training script for AudioCNN on ESC-50 dataset.
Usage: python train.py
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn as nn
import torchaudio.transforms as T
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import soundfile as sf

from model import AudioCNN

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

DATA_DIR = Path("./data/ESC-50-master")
CHECKPOINT_DIR = Path("./checkpoints")
LOG_DIR = Path("./runs")

BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.0005
MAX_LR = 0.002
WEIGHT_DECAY = 0.01
MIXUP_PROBABILITY = 0.3      # apply mixup to 30% of batches
MIXUP_ALPHA = 0.2
LABEL_SMOOTHING = 0.1

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
F_MIN = 0
F_MAX = 11025


# ──────────────────────────────────────────────
# Dataset Download
# ──────────────────────────────────────────────

def download_esc50(data_dir: Path):
    """Download and extract the ESC-50 dataset if not already present."""
    if (data_dir / "meta" / "esc50.csv").exists():
        print(f"✓ ESC-50 dataset already exists at {data_dir}")
        return

    zip_path = data_dir.parent / "esc50.zip"
    data_dir.parent.mkdir(parents=True, exist_ok=True)

    url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    print(f"Downloading ESC-50 dataset from {url} ...")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar_len = 40
            filled = int(bar_len * pct / 100)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"\r  [{bar}] {pct:.1f}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end='', flush=True)

    urllib.request.urlretrieve(url, zip_path, reporthook=progress_hook)
    print()  # newline after progress bar

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(data_dir.parent)

    print(f"✓ ESC-50 dataset extracted to {data_dir}")
    zip_path.unlink()  # clean up the zip file


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        if split == 'train':
            self.metadata = self.metadata[self.metadata['fold'] != 5]
        else:
            self.metadata = self.metadata[self.metadata['fold'] == 5]

        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(
            self.class_to_idx)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row['filename']

        # Load audio using SoundFile instead of TorchCodec
        audio, sample_rate = sf.read(audio_path, dtype="float32")

        # Convert NumPy array → PyTorch tensor
        waveform = torch.from_numpy(audio)

        # Convert to [channels, samples]
        if waveform.ndim == 1:
            # Mono: [samples] → [1, samples]
            waveform = waveform.unsqueeze(0)
        else:
            # Stereo/multi-channel: [samples, channels] → [channels, samples]
            waveform = waveform.transpose(0, 1)

        # Convert multi-channel audio to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Apply Mel Spectrogram / other transform
        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform

        return spectrogram, row['label']


# ──────────────────────────────────────────────
# Mixup Augmentation
# ──────────────────────────────────────────────

def mixup_data(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train():
    # ── Device setup ────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🚀 Using GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    else:
        print("⚠️  No CUDA GPU detected — training on CPU (this will be very slow)")

    # ── Download dataset ────────────────────
    download_esc50(DATA_DIR)

    # ── Transforms ──────────────────────────
    train_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX
        ),
        T.AmplitudeToDB(),
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80)
    )

    val_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX
        ),
        T.AmplitudeToDB()
    )

    # ── Datasets & DataLoaders ──────────────
    metadata_file = DATA_DIR / "meta" / "esc50.csv"

    train_dataset = ESC50Dataset(
        data_dir=DATA_DIR, metadata_file=metadata_file,
        split="train", transform=train_transform
    )
    val_dataset = ESC50Dataset(
        data_dir=DATA_DIR, metadata_file=metadata_file,
        split="test", transform=val_transform
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )

    # ── Model ───────────────────────────────
    model = AudioCNN(num_classes=len(train_dataset.classes))
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ── Training setup ──────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

    # ── TensorBoard ─────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_DIR / f"run_{timestamp}"
    writer = SummaryWriter(str(log_dir))
    print(f"📊 TensorBoard logs: {log_dir}")
    print(f"   Run: tensorboard --logdir {LOG_DIR}")

    # ── Training loop ───────────────────────
    best_accuracy = 0.0
    print(f"\n{'='*60}")
    print(f"Starting training for {NUM_EPOCHS} epochs")
    print(f"{'='*60}\n")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
            train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}',
            bar_format='{l_bar}{bar:30}{r_bar}'
        )

        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)

            if np.random.random() > (1 - MIXUP_PROBABILITY):
                data, target_a, target_b, lam = mixup_data(data, target)
                output = model(data)
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            else:
                output = model(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # ── Validation ──────────────────────
        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)

        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)

        print(
            f'Epoch {epoch+1:3d}/{NUM_EPOCHS} │ '
            f'Train Loss: {avg_epoch_loss:.4f} │ '
            f'Val Loss: {avg_val_loss:.4f} │ '
            f'Accuracy: {accuracy:.2f}%'
        )

        # ── Save best model ─────────────────
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch,
                'classes': train_dataset.classes
            }, checkpoint_path)
            print(f'  ✓ New best model saved: {accuracy:.2f}% → {checkpoint_path}')

    writer.close()
    print(f'\n{"="*60}')
    print(f'Training completed! Best accuracy: {best_accuracy:.2f}%')
    print(f'Best model saved to: {CHECKPOINT_DIR / "best_model.pth"}')
    print(f'{"="*60}')


if __name__ == "__main__":
    train()
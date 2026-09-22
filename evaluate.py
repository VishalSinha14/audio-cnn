"""
Evaluate the trained AudioCNN on the ESC-50 validation fold.

Uses the SAME validation preprocessing as train.py:
- SoundFile audio loading
- no resampling during dataset evaluation
- MelSpectrogram:
    sample_rate=22050
    n_fft=1024
    hop_length=512
    n_mels=128
    f_min=0
    f_max=11025
- AmplitudeToDB

Outputs:
evaluation/
    confusion_matrix.png
    classification_report.csv
    per_class_accuracy.csv
    evaluation_summary.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from model import AudioCNN


# ============================================================
# Configuration — intentionally matches train.py
# ============================================================

DATA_DIR = Path("./data/ESC-50-master")
METADATA_FILE = DATA_DIR / "meta" / "esc50.csv"
CHECKPOINT_PATH = Path("./checkpoints/best_model.pth")
OUTPUT_DIR = Path("./evaluation")

BATCH_SIZE = 32
NUM_WORKERS = 2

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
F_MIN = 0
F_MAX = 11025


# ============================================================
# Dataset
# ============================================================

class ESC50EvaluationDataset(Dataset):
    """
    ESC-50 fold 5 validation set.

    This deliberately mirrors the validation preprocessing in train.py.
    """

    def __init__(self, data_dir, metadata_file, transform=None):
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.metadata = self.metadata[self.metadata["fold"] == 5].copy()
        self.transform = transform

        self.classes = sorted(self.metadata["category"].unique())
        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        self.metadata["label"] = self.metadata["category"].map(
            self.class_to_idx
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row["filename"]

        audio, sample_rate = sf.read(audio_path, dtype="float32")

        waveform = torch.from_numpy(audio)

        # [samples] -> [1, samples]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            # [samples, channels] -> [channels, samples]
            waveform = waveform.transpose(0, 1)

        # Convert to mono, matching train.py
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.transform is not None:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform

        return spectrogram, int(row["label"])


# ============================================================
# Main evaluation
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("AudioCNN — ESC-50 Evaluation")
    print("=" * 70)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # Exact validation transform from train.py
    # --------------------------------------------------------

    val_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
        ),
        T.AmplitudeToDB(),
    )

    # --------------------------------------------------------
    # Dataset / DataLoader
    # --------------------------------------------------------

    dataset = ESC50EvaluationDataset(
        data_dir=DATA_DIR,
        metadata_file=METADATA_FILE,
        transform=val_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Validation samples: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    classes = checkpoint["classes"]
    checkpoint_accuracy = checkpoint.get("accuracy", None)
    checkpoint_epoch = checkpoint.get("epoch", None)

    print(f"Checkpoint epoch: {checkpoint_epoch}")
    if checkpoint_accuracy is not None:
        print(f"Checkpoint validation accuracy: {checkpoint_accuracy:.2f}%")

    # Verify class order
    if list(classes) != list(dataset.classes):
        raise RuntimeError(
            "Class ordering in checkpoint and evaluation dataset differs.\n"
            f"Checkpoint: {classes}\n"
            f"Dataset:    {dataset.classes}"
        )

    model = AudioCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader, start=1):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            outputs = model(data)
            predictions = torch.argmax(outputs, dim=1)

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

            print(
                f"\rEvaluating batch {batch_idx}/{len(loader)}",
                end="",
                flush=True,
            )

    print()

    y_true = np.asarray(all_targets)
    y_pred = np.asarray(all_predictions)

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y_true, y_pred)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()

    # --------------------------------------------------------
    # Per-class accuracy
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(classes)),
    )

    per_class_rows = []

    for i, class_name in enumerate(classes):
        total = cm[i].sum()
        correct = cm[i, i]
        class_accuracy = correct / total if total else 0.0

        per_class_rows.append(
            {
                "class": class_name,
                "correct": int(correct),
                "total": int(total),
                "accuracy": class_accuracy,
                "precision": report_df.loc[class_name, "precision"],
                "recall": report_df.loc[class_name, "recall"],
                "f1_score": report_df.loc[class_name, "f1-score"],
            }
        )

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df = per_class_df.sort_values(
        "accuracy",
        ascending=False,
    )

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    report_df.to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    per_class_df.to_csv(
        OUTPUT_DIR / "per_class_accuracy.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Confusion matrix plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(22, 20),
    )

    image = ax.imshow(cm, interpolation="nearest")

    ax.set_title(
        f"ESC-50 Confusion Matrix\nAccuracy: {accuracy:.2%}"
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(
        classes,
        rotation=90,
        fontsize=7,
    )
    ax.set_yticklabels(
        classes,
        fontsize=7,
    )

    fig.colorbar(image, ax=ax)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Find strongest class confusions
    # --------------------------------------------------------

    confusion_pairs = []

    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append(
                    (
                        int(cm[i, j]),
                        classes[i],
                        classes[j],
                    )
                )

    confusion_pairs.sort(reverse=True)

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = OUTPUT_DIR / "evaluation_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("AudioCNN ESC-50 Evaluation\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Device: {device}\n")
        if device.type == "cuda":
            f.write(
                f"GPU: {torch.cuda.get_device_name(0)}\n"
            )

        f.write(f"Validation samples: {len(dataset)}\n")
        f.write(f"Classes: {len(classes)}\n")

        if checkpoint_epoch is not None:
            f.write(
                f"Checkpoint epoch: {checkpoint_epoch}\n"
            )

        if checkpoint_accuracy is not None:
            f.write(
                f"Checkpoint recorded accuracy: "
                f"{checkpoint_accuracy:.2f}%\n"
            )

        f.write(
            f"Recomputed accuracy: {accuracy:.4%}\n"
        )

        f.write("\n")
        f.write("Overall metrics\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"Accuracy:  {report_df.loc['accuracy', 'f1-score']:.4%}\n"
        )
        f.write(
            f"Macro F1:  {report_df.loc['macro avg', 'f1-score']:.4f}\n"
        )
        f.write(
            f"Weighted F1: {report_df.loc['weighted avg', 'f1-score']:.4f}\n"
        )

        f.write("\n")
        f.write("Strongest confusion pairs\n")
        f.write("-" * 60 + "\n")

        for count, true_class, predicted_class in confusion_pairs[:15]:
            f.write(
                f"{true_class:25s} -> "
                f"{predicted_class:25s}: {count}\n"
            )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(f"Accuracy : {accuracy:.2%}")
    print(
        f"Macro F1 : "
        f"{report_df.loc['macro avg', 'f1-score']:.4f}"
    )
    print(
        f"Weighted F1 : "
        f"{report_df.loc['weighted avg', 'f1-score']:.4f}"
    )

    print("\nTop 10 class accuracies:")
    print(
        per_class_df[
            ["class", "correct", "total", "accuracy"]
        ].head(10).to_string(index=False)
    )

    print("\nStrongest confusions:")
    for count, true_class, predicted_class in confusion_pairs[:10]:
        print(
            f"  {true_class} -> {predicted_class}: {count}"
        )

    print("\nSaved files:")
    print(f"  {OUTPUT_DIR / 'confusion_matrix.png'}")
    print(f"  {OUTPUT_DIR / 'classification_report.csv'}")
    print(f"  {OUTPUT_DIR / 'per_class_accuracy.csv'}")
    print(f"  {OUTPUT_DIR / 'evaluation_summary.txt'}")


if __name__ == "__main__":
    main()

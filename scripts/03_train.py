#!/usr/bin/env python3
"""
Step 4: Train microWakeWord model.

Usage (from ua_wakeword/):
  source env-mww/bin/activate
  python scripts/03_train.py --word "агент" --samples-dir generated_samples/styletts2

Configuration: matches basic_training_notebook.ipynb from kahrendt/microWakeWord
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--word", type=str, default="агент",
                    help="Wake word name (used for output naming)")
parser.add_argument("--samples-dir", type=str, default="generated_samples/styletts2",
                    help="Directory with positive WAV samples (16kHz)")
parser.add_argument("--aug-dir", type=str, default="augmentation_data",
                    help="Augmentation data directory")
parser.add_argument("--output-dir", type=str, default="training_output",
                    help="Output directory for model files")
parser.add_argument("--steps", type=int, default=20000,
                    help="Training steps (default: 20000)")
parser.add_argument("--batch-size", type=int, default=128)
args = parser.parse_args()

samples_dir = Path(args.samples_dir)
aug_dir = Path(args.aug_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Validation checks
n_samples = len(list(samples_dir.glob("*.wav")))
if n_samples < 10:
    print(f"[ERROR] Only {n_samples} samples found in {samples_dir}")
    print("Run first: python scripts/01_generate_samples.py")
    sys.exit(1)

print(f"[INFO] Found {n_samples} positive samples")
print(f"[INFO] Wake word: '{args.word}'")
print(f"[INFO] Output: {output_dir}")

# ── Setup ──────────────────────────────────────────────────────────────────────
import numpy as np
import tensorflow as tf
print(f"[INFO] TensorFlow: {tf.__version__}")

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"[INFO] GPU: {gpus if gpus else 'CPU only'}")

import microwakeword
from microwakeword.data_processor import DataProcessor
from microwakeword import train as mww_train

# ── Configuration (from basic_training_notebook.ipynb) ────────────────────────
config = {
    "target_word": args.word,
    "batch_size": args.batch_size,
    "steps": args.steps,
    "spectrogram_length": 198,        # number of 10ms frames (~2 sec)
    "time_shift_ms": 100,
    "sample_rate": 16000,
    "clip_duration_ms": 1980,
    "window_size_ms": 30,
    "window_stride_ms": 10,
    "feature_bin_count": 40,          # 40-feature melspectrogram
    "learning_rate_step_boundaries": [10000, 15000, 18000],
    "learning_rate_schedule_values": [0.001, 0.0005, 0.0002, 0.0001],
    "positive_class_weight": 25,      # important for rare class
    "data": {
        "wanted_words": [args.word],
        "positive": str(samples_dir.resolve()),
        "speech": str((aug_dir / "negative_speech" / "speech").resolve()),
        "dinner_party": str((aug_dir / "negative_speech" / "dinner_party").resolve()),
        "no_speech": str((aug_dir / "negative_speech" / "no_speech").resolve()),
        "rirs": str((aug_dir / "mit_rirs").resolve()),
        "noise": str((aug_dir / "audioset_16k").resolve())
                 if (aug_dir / "audioset_16k").exists()
                 else str((aug_dir / "fma").resolve()),
        "music": str((aug_dir / "fma").resolve()),
    },
}

# Save config for reproducibility
config_path = output_dir / "training_config.yaml"
with open(config_path, "w") as f:
    yaml.dump(config, f, allow_unicode=True)
print(f"[INFO] Config saved: {config_path}")

# ── Augmentation config ────────────────────────────────────────────────────────
augmentation_config = {
    "rirs_dir": config["data"]["rirs"],
    "noise_dir": config["data"]["noise"],
    "music_dir": config["data"]["music"],
    "noise_prob": 0.8,
    "music_prob": 0.3,
    "rir_prob": 0.7,
    "min_snr_db": 5,
    "max_snr_db": 20,
}

# ── Data ───────────────────────────────────────────────────────────────────────
print("\n[INFO] Initializing DataProcessor...")
data_processor = DataProcessor(config)

print("[INFO] Building training set...")
train_fingerprints, train_ground_truth, _ = data_processor.get_data(
    "training",
    batch_size=config["batch_size"],
    features_length=config["spectrogram_length"],
    truncation_strategy="random",
)
print(f"  train: {train_fingerprints.shape}")

val_fingerprints, val_ground_truth, _ = data_processor.get_data(
    "validation",
    batch_size=config["batch_size"],
    features_length=config["spectrogram_length"],
    truncation_strategy="truncate_start",
)
print(f"  val:   {val_fingerprints.shape}")

# ── Model ──────────────────────────────────────────────────────────────────────
print("\n[INFO] Building model...")
from microwakeword.model import MicroWakeWordModel

model = MicroWakeWordModel(config).get_model()
model.summary()

# ── Training ───────────────────────────────────────────────────────────────────
print(f"\n[INFO] Training ({args.steps} steps)...")
history = mww_train.train(
    config=config,
    data_processor=data_processor,
    model=model,
    train_set="training",
    test_set="validation",
    augmentation_config=augmentation_config,
)

# ── Save ───────────────────────────────────────────────────────────────────────
model_safe_name = args.word.replace(" ", "_")
saved_model_path = output_dir / f"{model_safe_name}_model"
model.save(str(saved_model_path))
print(f"\n[INFO] Model saved: {saved_model_path}")

# ── Convert to TFLite ──────────────────────────────────────────────────────────
print("[INFO] Converting to TFLite (INT8 quantization)...")
from microwakeword import model_conversion

tflite_path = output_dir / f"{model_safe_name}.tflite"
model_conversion.convert_to_tflite(
    model=model,
    config=config,
    data_processor=data_processor,
    output_path=str(tflite_path),
)

tflite_size = tflite_path.stat().st_size / 1024
print(f"[DONE] TFLite model: {tflite_path} ({tflite_size:.1f} KB)")
print(f"[DONE] Done! Copy {tflite_path} to your Raspberry Pi.")

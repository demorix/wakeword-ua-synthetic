#!/usr/bin/env python3
"""
Step 6: Inference and testing of a trained microWakeWord model on an audio file.

Designed for validation on PC/server or directly on Raspberry Pi.

Usage:
  source env-mww/bin/activate
  python scripts/05_test_inference.py --audio generated_samples/styletts2/00000.wav
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import resampy

from microwakeword.inference import Model

parser = argparse.ArgumentParser(description="Test microWakeWord streaming TFLite model")
parser.add_argument("--model", default="trained_models/agent_ukrainian_wakeword.tflite", help="Path to .tflite model")
parser.add_argument("--audio", required=True, help="Path to input .wav audio")
parser.add_argument("--cutoff", type=float, default=0.95, help="Detection probability cutoff (default: 0.95)")
parser.add_argument("--smooth", type=int, default=3, help="Sliding window smoothing average (default: 3)")
args = parser.parse_args()

model_path = Path(args.model)
audio_path = Path(args.audio)

if not model_path.exists():
    print(f"[ERROR] Model not found: {model_path}")
    sys.exit(1)

if not audio_path.exists():
    print(f"[ERROR] Audio file not found: {audio_path}")
    sys.exit(1)

# Load audio
data, sr = sf.read(str(audio_path))
if data.ndim > 1:
    data = data.mean(axis=1)
if sr != 16000:
    data = resampy.resample(data.astype(np.float32), sr, 16000)

audio_int16 = (data * 32767).clip(-32768, 32767).astype(np.int16)

# Initialize model with stride=3 (matches trained MixedNet architecture)
model = Model(str(model_path), stride=3)

# Run model over audio (step_ms=10)
predictions = model.predict_clip(audio_int16, step_ms=10)

scores = [float(p) for p in predictions]
if args.smooth > 1 and len(scores) >= args.smooth:
    scores = list(np.convolve(scores, np.ones(args.smooth) / args.smooth, mode="valid"))

max_score = max(scores) if scores else 0.0

detected = any(s >= args.cutoff for s in scores)
if detected:
    detect_idx = next(i for i, s in enumerate(scores) if s >= args.cutoff)
    time_sec = (detect_idx * 3 * 10) / 1000.0  # stride * step_ms
    print(f"🎯 [DETECTION] Wake word detected at {time_sec:.2f}s! (score: {scores[detect_idx]*100:.1f}%)")

print(f"\n[INFO] Result: {'✅ DETECTED' if detected else '❌ NOT DETECTED'}")
print(f"[INFO] Max model score: {max_score*100:.1f}% (activation threshold: {args.cutoff*100:.1f}%)")

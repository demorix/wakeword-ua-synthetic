#!/usr/bin/env python3
"""
Step 6.1: Real-time wake word detection via microphone (Live Microphone Detection).

Works on both PC/server and Raspberry Pi.

Usage:
  source env-mww/bin/activate
  python scripts/06_live_microphone.py [--device 0] [--cutoff 0.95]
"""

import argparse
import collections
import sys
import time
from pathlib import Path
import numpy as np
import sounddevice as sd
from microwakeword.inference import Model

parser = argparse.ArgumentParser(description="Live microphone streaming detection with microWakeWord")
parser.add_argument("--model", default="trained_models/agent_ukrainian_wakeword.tflite", help="Path to .tflite model")
parser.add_argument("--cutoff", type=float, default=0.95, help="Detection cutoff threshold (default: 0.95)")
parser.add_argument("--device", type=int, default=None, help="Audio input device index")
parser.add_argument("--smooth", type=int, default=3, help="Sliding window smoothing average (default: 3)")
args = parser.parse_args()

model_path = Path(args.model)
if not model_path.exists():
    print(f"[ERROR] Model not found: {model_path}")
    sys.exit(1)

print(f"[INFO] Loading model {model_path} (stride=3)...")
model = Model(str(model_path), stride=3)

SAMPLE_RATE = 16000
CHUNK_DURATION_SEC = 0.5  # 500 ms chunks
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_SEC)
BUFFER_DURATION_SEC = 2.0  # rolling 2-second audio buffer
BUFFER_SAMPLES = int(SAMPLE_RATE * BUFFER_DURATION_SEC)

audio_buffer = np.zeros(BUFFER_SAMPLES, dtype=np.int16)
score_history = collections.deque(maxlen=args.smooth)
last_detection_time = 0.0

print(f"\n" + "="*60)
print(f"🎙️  LIVE WAKE WORD TEST: 'агент'")
print(f"👉 Detection threshold: {args.cutoff*100:.0f}% | Smoothing: {args.smooth} frames")
print(f"👉 Speak into the microphone: 'агент' (or other words to test false positives)")
print(f"👉 Press Ctrl+C to stop")
print("="*60 + "\n")

try:
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SAMPLES,
        device=args.device,
        channels=1,
        dtype="int16",
    ) as stream:
        while True:
            data, overflowed = stream.read(CHUNK_SAMPLES)
            if overflowed:
                pass

            chunk = np.frombuffer(data, dtype=np.int16)
            # Shift audio buffer
            audio_buffer = np.roll(audio_buffer, -len(chunk))
            audio_buffer[-len(chunk):] = chunk

            # Run inference on the latest buffer
            preds = model.predict_clip(audio_buffer, step_ms=10)
            if preds:
                # Latest predictions from the most recent chunk
                recent_preds = preds[-int(CHUNK_DURATION_SEC * 1000 / (10 * 3)):]
                for p in recent_preds:
                    score_history.append(float(p))

                smoothed_score = sum(score_history) / len(score_history) if score_history else 0.0
                max_recent = max([float(p) for p in recent_preds]) if recent_preds else 0.0

                # Visual confidence bar
                bar_len = int(smoothed_score * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)

                curr_time = time.time()
                if smoothed_score >= args.cutoff and (curr_time - last_detection_time) > 1.5:
                    last_detection_time = curr_time
                    print(f"\r🔥 [{time.strftime('%H:%M:%S')}] WAKE WORD 'АГЕНТ' DETECTED! [{bar}] {smoothed_score*100:5.1f}%\n", flush=True)
                else:
                    sys.stdout.write(f"\rMatch level: [{bar}] {smoothed_score*100:5.1f}% (peak: {max_recent*100:5.1f}%)")
                    sys.stdout.flush()

except KeyboardInterrupt:
    print("\n\n[INFO] Stopped by user.")
except Exception as e:
    print(f"\n[ERROR] Microphone or audio device error: {e}")
    print("[TIP] You can list available microphones with: python -m sounddevice")

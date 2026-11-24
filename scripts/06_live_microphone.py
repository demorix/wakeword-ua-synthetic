#!/usr/bin/env python3
"""
Етап 6.1: Тестування в реальному часі через мікрофон (Live Microphone Detection).

Підходить як для ПК/сервера, так і для Raspberry Pi.

Використання:
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
    print(f"[ERROR] Модель {model_path} не знайдена!")
    sys.exit(1)

print(f"[INFO] Завантаження моделі {model_path} (stride=3)...")
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
print(f"🎙️  МІКРОФОННИЙ ТЕСТ WAKE WORD: 'агент'")
print(f"👉 Поріг детекції: {args.cutoff*100:.0f}% | Згладжування: {args.smooth} фреймів")
print(f"👉 Говоріть у мікрофон: 'агент' (або інші слова для тесту хибних спрацьовувань)")
print(f"👉 Натисніть Ctrl+C для завершення")
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
            # Зсув буфера
            audio_buffer = np.roll(audio_buffer, -len(chunk))
            audio_buffer[-len(chunk):] = chunk

            # Інференс на останньому буфері
            preds = model.predict_clip(audio_buffer, step_ms=10)
            if preds:
                # Останні кілька передбачень від останнього чанка
                recent_preds = preds[-int(CHUNK_DURATION_SEC * 1000 / (10 * 3)):]
                for p in recent_preds:
                    score_history.append(float(p))
                
                smoothed_score = sum(score_history) / len(score_history) if score_history else 0.0
                max_recent = max([float(p) for p in recent_preds]) if recent_preds else 0.0
                
                # Візуальна шкала
                bar_len = int(smoothed_score * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                
                curr_time = time.time()
                if smoothed_score >= args.cutoff and (curr_time - last_detection_time) > 1.5:
                    last_detection_time = curr_time
                    print(f"\r🔥 [{time.strftime('%H:%M:%S')}] ВАКЕWORD 'АГЕНТ' ВИЯВЛЕНО! [{bar}] {smoothed_score*100:5.1f}%\n", flush=True)
                else:
                    sys.stdout.write(f"\rРівень відповідності: [{bar}] {smoothed_score*100:5.1f}% (пік: {max_recent*100:5.1f}%)")
                    sys.stdout.flush()

except KeyboardInterrupt:
    print("\n\n[INFO] Зупинено користувачем.")
except Exception as e:
    print(f"\n[ERROR] Помилка мікрофона або аудіопристрою: {e}")
    print("[TIP] Ви можете перевірити доступні мікрофони за допомогою: python -m sounddevice")


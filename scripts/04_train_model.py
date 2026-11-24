#!/usr/bin/env python3
"""
Етап 4: Тренування та квантизація моделі microWakeWord (streaming MixedNet).

Використовує офіційний модуль `microwakeword.model_train_eval` з конфігурацією:
- MixedNet (MixConv) архітектура, адаптована для потокового інференсу на пристроях
- 10 000 кроків тренування
- Автоматичний вибір кращих ваг (best_weights)
- Автоматична квантизація INT8 у streaming TFLite формат

Запуск:
  source env-mww/bin/activate
  python scripts/04_train_model.py
"""

import os
import subprocess
import sys
import yaml
import shutil
from pathlib import Path

train_dir = Path("trained_models/wakeword_agent")
if train_dir.exists():
    shutil.rmtree(train_dir)

config = {
    "window_step_ms": 10,
    "train_dir": str(train_dir),
    "features": [
        {
            "features_dir": "generated_augmented_features",
            "sampling_weight": 2.0,
            "penalty_weight": 1.0,
            "truth": True,
            "truncation_strategy": "truncate_start",
            "type": "mmap",
        },
        {
            "features_dir": "augmentation_data/negative_datasets/speech",
            "sampling_weight": 10.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "random",
            "type": "mmap",
        },
        {
            "features_dir": "augmentation_data/negative_datasets/dinner_party",
            "sampling_weight": 10.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "random",
            "type": "mmap",
        },
        {
            "features_dir": "augmentation_data/negative_datasets/no_speech",
            "sampling_weight": 5.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "random",
            "type": "mmap",
        },
        {
            "features_dir": "augmentation_data/negative_datasets/dinner_party_eval",
            "sampling_weight": 0.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "split",
            "type": "mmap",
        },
    ],
    "training_steps": [8000],
    "positive_class_weight": [1],
    "negative_class_weight": [20],
    "learning_rates": [0.001],
    "batch_size": 128,
    "time_mask_max_size": [15],
    "time_mask_count": [2],
    "freq_mask_max_size": [8],
    "freq_mask_count": [2],
    "eval_step_interval": 500,
    "clip_duration_ms": 1500,
    "target_minimization": 0.9,
    "minimization_metric": None,
    "maximization_metric": "average_viable_recall",
}

config_file = "training_parameters.yaml"
with open(config_file, "w") as f:
    yaml.dump(config, f)

print(f"[INFO] Створено конфігураційний файл {config_file}")
print("[INFO] Запуск офіційного циклу тренування microWakeWord...")

cmd = [
    sys.executable,
    "-m",
    "microwakeword.model_train_eval",
    f"--training_config={config_file}",
    "--train", "1",
    "--restore_checkpoint", "0",
    "--test_tf_nonstreaming", "0",
    "--test_tflite_nonstreaming", "0",
    "--test_tflite_nonstreaming_quantized", "0",
    "--test_tflite_streaming", "0",
    "--test_tflite_streaming_quantized", "1",
    "--use_weights", "best_weights",
    "mixednet",
    "--pointwise_filters", "64,64,64,64",
    "--repeat_in_block", "1, 1, 1, 1",
    "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
    "--residual_connection", "0,0,0,0",
    "--first_conv_filters", "32",
    "--first_conv_kernel_size", "5",
    "--stride", "3",
]

print(f"[CMD] {' '.join(cmd)}")
res = subprocess.run(cmd)
if res.returncode != 0:
    print(f"[ERROR] Тренування завершилось з кодом {res.returncode}")
    sys.exit(res.returncode)

print("\n[SUCCESS] Тренування та квантизація завершені!")
print(f"Результати у: {train_dir.resolve()}")

# Копіюємо на фінальне стандартизоване місце
src_tflite = train_dir / "tflite_stream_state_internal_quant" / "stream_state_internal_quant.tflite"
dst_tflite = Path("trained_models/aiva_ukrainian_wakeword.tflite")
if src_tflite.exists():
    import shutil
    shutil.copy2(src_tflite, dst_tflite)
    print(f"[EXPORT] Модель успішно скопійована в {dst_tflite} ({dst_tflite.stat().st_size / 1024:.1f} KB)")


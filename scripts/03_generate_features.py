#!/usr/bin/env python3
"""
Етап 3.5: Генерація та аугментація спектрограм для microWakeWord.

Бере згенеровані WAV-семпли StyleTTS2, накладає аугментації (реверберацію RIR,
фоновий шум, фільтри, зміну гучності) та перетворює їх на 40-канальні
mmap-спектрограми (mmap_ninja), готові для подачі в нейромережу.

Запуск:
  source env-mww/bin/activate
  python scripts/03_generate_features.py
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Перевірка наявності потрібних бібліотек
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration
from mmap_ninja.ragged import RaggedMmap

parser = argparse.ArgumentParser(description="Generate augmented spectrogram features for microWakeWord")
parser.add_argument("--samples-dir", type=str, default="generated_samples/semp_aivo",
                    help="Directory containing positive WAV samples (default: generated_samples/semp_aivo)")
parser.add_argument("--output-dir", type=str, default="generated_augmented_features",
                    help="Output directory for generated mmap features (default: generated_augmented_features)")
parser.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing features in output directory")
args = parser.parse_args()

samples_dir = Path(args.samples_dir)
if not samples_dir.exists() and Path("generated_samples/styletts2").exists():
    samples_dir = Path("generated_samples/styletts2")

aug_base = Path("augmentation_data")
rirs_dir = aug_base / "mit_rirs"
fma_dir = aug_base / "fma_16k"
output_dir = Path(args.output_dir)

if not samples_dir.exists() or len(list(samples_dir.glob("*.wav"))) == 0:
    print(f"[ERROR] Не знайдено WAV файлів у {samples_dir}")
    sys.exit(1)

sample_count = len(list(samples_dir.glob("*.wav")))
print(f"[INFO] Використовуються семпли: {samples_dir} ({sample_count} файлів)")
print(f"[INFO] RIR імпульси: {rirs_dir} ({len(list(rirs_dir.glob('*.wav')))} файлів)")
print(f"[INFO] Фоновий шум: {fma_dir} ({len(list(fma_dir.glob('*.wav')))} файлів)")
print(f"[INFO] Вихідний каталог: {output_dir}")

if args.overwrite and output_dir.exists():
    print(f"[INFO] Прапорець --overwrite активний: очищую {output_dir}...")
    shutil.rmtree(output_dir)

output_dir.mkdir(parents=True, exist_ok=True)

print("[INFO] Ініціалізація Clips та Augmenter...")
clips = Clips(
    input_directory=str(samples_dir),
    file_pattern="*.wav",
    max_clip_duration_s=None,
    remove_silence=False,
    random_split_seed=10,
    split_count=0.1,
)

augmenter = Augmentation(
    augmentation_duration_s=3.2,
    augmentation_probabilities={
        "SevenBandParametricEQ": 0.4,
        "TanhDistortion": 0.2,
        "PitchShift": 0.6,    # Активно варіюємо висоту голосу (-4..+4 півтонів)
        "TimeStretch": 0.5,   # Активно варіюємо темп мовлення (0.8..1.25x)
        "BandStopFilter": 0.2,
        "AddColorNoise": 0.3,
        "AddBackgroundNoise": 0.8,
        "Gain": 1.0,
        "GainTransition": 0.3,
        "RIR": 0.8,           # 80% семплів проходять реальну кімнатну реверберацію
    },
    impulse_paths=[str(rirs_dir)],
    background_paths=[str(fma_dir)],
    background_min_snr_db=-5,
    background_max_snr_db=15,
    min_jitter_s=0.05,
    max_jitter_s=0.35,
)

splits = ["training", "validation", "testing"]
for split in splits:
    split_out_dir = output_dir / split
    split_out_dir.mkdir(parents=True, exist_ok=True)
    mmap_target = split_out_dir / "wakeword_mmap"

    if mmap_target.exists():
        print(f"[INFO] Спектрограми для {split} вже згенеровано — пропускаю")
        continue

    print(f"\n[INFO] Генерація спектрограм для {split}...")
    if split == "training":
        split_name = "train"
        repetition = 3
        slide_frames = 10
    elif split == "validation":
        split_name = "validation"
        repetition = 1
        slide_frames = 10
    elif split == "testing":
        split_name = "test"
        repetition = 1
        slide_frames = 1

    spectrograms = SpectrogramGeneration(
        clips=clips,
        augmenter=augmenter,
        slide_frames=slide_frames,
        step_ms=10,
    )

    RaggedMmap.from_generator(
        out_dir=str(mmap_target),
        sample_generator=spectrograms.spectrogram_generator(split=split_name, repeat=repetition),
        batch_size=100,
        verbose=True,
    )
    print(f"  ✓ {split} збережено в {mmap_target}")

print("\n[DONE] Усі позитивні ознаки (спектрограми) згенеровано успішно!")


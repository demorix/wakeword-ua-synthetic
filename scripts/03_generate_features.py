#!/usr/bin/env python3
"""
Step 3.5: Generate and augment spectrograms for microWakeWord.

Takes StyleTTS2-generated WAV samples, applies augmentations (RIR reverb,
background noise, filters, volume changes) and converts them to 40-channel
mmap spectrograms (mmap_ninja), ready to feed into the neural network.

Usage:
  source env-mww/bin/activate
  python scripts/03_generate_features.py
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Check required libraries are available
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
    print(f"[ERROR] No WAV files found in {samples_dir}")
    sys.exit(1)

sample_count = len(list(samples_dir.glob("*.wav")))
print(f"[INFO] Using samples from: {samples_dir} ({sample_count} files)")
print(f"[INFO] RIR impulses: {rirs_dir} ({len(list(rirs_dir.glob('*.wav')))} files)")
print(f"[INFO] Background noise: {fma_dir} ({len(list(fma_dir.glob('*.wav')))} files)")
print(f"[INFO] Output directory: {output_dir}")

if args.overwrite and output_dir.exists():
    print(f"[INFO] --overwrite flag set: clearing {output_dir}...")
    shutil.rmtree(output_dir)

output_dir.mkdir(parents=True, exist_ok=True)

print("[INFO] Initializing Clips and Augmenter...")
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
        "PitchShift": 0.6,    # Actively vary pitch (-4..+4 semitones)
        "TimeStretch": 0.5,   # Actively vary speech tempo (0.8..1.25x)
        "BandStopFilter": 0.2,
        "AddColorNoise": 0.3,
        "AddBackgroundNoise": 0.8,
        "Gain": 1.0,
        "GainTransition": 0.3,
        "RIR": 0.8,           # 80% of samples pass through real room reverb
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
        print(f"[INFO] Spectrograms for {split} already generated — skipping")
        continue

    print(f"\n[INFO] Generating spectrograms for {split}...")
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
    print(f"  ✓ {split} saved to {mmap_target}")

print("\n[DONE] All positive features (spectrograms) generated successfully!")

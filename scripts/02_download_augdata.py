#!/usr/bin/env python3
"""
Етап 3: Завантаження аугментаційних даних для тренування microWakeWord.

Завантажує:
  - MIT Room Impulse Responses → augmentation_data/mit_rirs/ (WAV)
  - kahrendt/microwakeword negative sets → augmentation_data/negative_datasets/ (mmap_ninja)
  - FMA xsmall music → augmentation_data/fma/ (MP3/WAV)
  - AudioSet (опційно, великий ~1GB)

Формат negative datasets: mmap_ninja (HF zip архіви), НЕ WAV!
Використовуються напряму microWakeWord через config["features"].

Запуск:
  source env-mww/bin/activate
  python scripts/02_download_augdata.py [--skip-audioset] [--skip-fma]
"""

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import soundfile as sf
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="augmentation_data", help="Base output dir")
parser.add_argument("--skip-audioset", action="store_true", help="Skip AudioSet (large, ~1GB)")
parser.add_argument("--skip-fma", action="store_true", help="Skip FMA music dataset")
args = parser.parse_args()

base = Path(args.out)
base.mkdir(parents=True, exist_ok=True)

# ── 1. MIT Room Impulse Responses (WAV — для audiomentations reverb) ──────────
rir_dir = base / "mit_rirs"
if not rir_dir.exists():
    rir_dir.mkdir()
    print("[1/4] Завантаження MIT RIR (room impulse responses)...")
    try:
        import datasets as hf_datasets
        rir_dataset = hf_datasets.load_dataset(
            "davidscripka/MIT_environmental_impulse_responses",
            split="train",
            streaming=True,
        ).cast_column("audio", hf_datasets.Audio(decode=False))
        count = 0
        for row in tqdm(rir_dataset, desc="MIT RIR"):
            audio_bytes = row["audio"].get("bytes") or b""
            path_hint = row["audio"].get("path", "")
            name = (path_hint.split("/")[-1] or f"{count:05d}") if path_hint else f"{count:05d}.wav"
            if not name.endswith(".wav"):
                name += ".wav"
            if audio_bytes:
                try:
                    data, sr = sf.read(io.BytesIO(audio_bytes))
                    if sr != 16000:
                        import resampy
                        data = resampy.resample(data.astype(np.float32), sr, 16000)
                    scipy.io.wavfile.write(str(rir_dir / name), 16000, (data * 32767).astype(np.int16))
                    count += 1
                except Exception:
                    pass
        print(f"  ✓ MIT RIR: {count} файлів")
    except Exception as e:
        print(f"  ✗ MIT RIR: {e}")
else:
    n = len(list(rir_dir.glob("*.wav")))
    print(f"[1/4] MIT RIR вже є ({n} файлів) — пропускаю")

# ── 2. kahrendt/microwakeword negative datasets (mmap_ninja zip) ───────────────
neg_dir = base / "negative_datasets"
if not neg_dir.exists():
    neg_dir.mkdir()
    print("[2/4] Завантаження negative datasets (mmap_ninja format)...")
    try:
        from huggingface_hub import hf_hub_download
        filenames = [
            "dinner_party.zip",
            "dinner_party_eval.zip",
            "no_speech.zip",
            "speech.zip",
        ]
        for fname in filenames:
            print(f"  Завантажую {fname}...")
            try:
                zip_path = hf_hub_download(
                    repo_id="kahrendt/microwakeword",
                    filename=fname,
                    repo_type="dataset",
                )
                print(f"  Розпаковую {fname}...")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(str(neg_dir))
                print(f"  ✓ {fname}")
            except Exception as e:
                print(f"  ✗ {fname}: {e}")
    except Exception as e:
        print(f"  ✗ negative datasets: {e}")
else:
    subdirs = [d.name for d in neg_dir.iterdir() if d.is_dir()]
    print(f"[2/4] Negative datasets вже є ({subdirs}) — пропускаю")

# ── 3. AudioSet (великий, опційно) ─────────────────────────────────────────────
audioset_dir = base / "audioset_16k"
if not args.skip_audioset and not audioset_dir.exists():
    audioset_dir.mkdir()
    import subprocess, tarfile
    print("[3/4] Завантаження AudioSet bal_train09.tar (~1GB)...")
    raw_dir = base / "audioset_raw"
    raw_dir.mkdir(exist_ok=True)
    tar_path = raw_dir / "bal_train09.tar"
    if not tar_path.exists():
        subprocess.run([
            "wget", "-q", "--show-progress", "-O", str(tar_path),
            "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train09.tar"
        ], check=True)
    print("  Розпаковую...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(str(raw_dir))
    flac_files = list(raw_dir.glob("**/*.flac"))
    print(f"  Конвертую {len(flac_files)} .flac → 16kHz wav...")
    count = 0
    for fp in tqdm(flac_files):
        try:
            data, sr = sf.read(str(fp))
            if sr != 16000:
                import resampy
                data = resampy.resample(data.astype(np.float32), sr, 16000)
            name = fp.stem + ".wav"
            scipy.io.wavfile.write(str(audioset_dir / name), 16000, (data * 32767).astype(np.int16))
            count += 1
        except Exception:
            pass
    print(f"  ✓ AudioSet: {count} файлів")
else:
    if args.skip_audioset:
        print("[3/4] AudioSet — пропущено (--skip-audioset)")
    else:
        print(f"[3/4] AudioSet вже є ({len(list(audioset_dir.glob('*.wav')))} файлів) — пропускаю")

# ── 4. FMA xsmall music ────────────────────────────────────────────────────────
fma_dir = base / "fma"
if not args.skip_fma and not fma_dir.exists():
    fma_dir.mkdir()
    import subprocess
    print("[4/4] Завантаження FMA xsmall...")
    zip_path = fma_dir / "fma_xs.zip"
    subprocess.run([
        "wget", "-q", "--show-progress", "-O", str(zip_path),
        "https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip"
    ], check=True)
    print("  Розпаковую...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(str(fma_dir))
    print(f"  ✓ FMA xsmall")
else:
    if args.skip_fma:
        print("[4/4] FMA — пропущено (--skip-fma)")
    else:
        print(f"[4/4] FMA вже є — пропускаю")

print("\n[DONE] Аугментаційні дані готові!")
print(f"Директорія: {base.resolve()}")
for d in sorted(base.iterdir()):
    if d.is_dir():
        wav_count = len(list(d.rglob("*.wav")))
        all_count = sum(1 for _ in d.rglob("*") if _.is_file())
        print(f"  {d.name}/: {wav_count} wav, {all_count} total файлів")

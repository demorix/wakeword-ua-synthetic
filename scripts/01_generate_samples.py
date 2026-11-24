#!/usr/bin/env python3
"""
Етап 2: Генерація wake word семплів через StyleTTS2 Ukrainian (multispeaker).

API: model.tokenizer.encode(ipa_text) → tokens
     model(tokens, speed=speed, s_prev=style_tensor) → wav (24kHz)

Запуск (з кореневої директорії ua_wakeword):
  source styletts2-ukrainian/env-tts/bin/activate
  python scripts/01_generate_samples.py --word "агент" --count 1500 --out generated_samples/styletts2
"""

import argparse
import glob
import os
import re
import sys
import random
import subprocess
from pathlib import Path
from unicodedata import normalize

# ── Автоматичне перемикання на venv StyleTTS2, якщо torch відсутній ───────────
try:
    import torch
    import numpy as np
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parent.parent
    candidate_venvs = [
        repo_root / "styletts2-ukrainian" / "env-tts" / "bin" / "python",
        Path("/home/sysvb/antigravity/styletts2-ukrainian/env-tts/bin/python"),
    ]
    tts_python = next((p for p in candidate_venvs if p.exists()), None)
    if tts_python and sys.executable != str(tts_python):
        print(f"[INFO] Torch не знайдено в поточному середовищі. Автоматично перемикаюсь на {tts_python}...")
        res = subprocess.run([str(tts_python), *sys.argv])
        sys.exit(res.returncode)
    else:
        print("[ERROR] Не знайдено модуль 'torch'. Будь ласка, активуйте середовище StyleTTS2:")
        print("  source styletts2-ukrainian/env-tts/bin/activate")
        sys.exit(1)

# ── Аргументи ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="StyleTTS2 wake word batch generator")
parser.add_argument("--word", type=str, required=True,
                    help="Wake word text (e.g. 'агент' or 'окей агент')")
parser.add_argument("--count", type=int, default=1500,
                    help="Total number of samples to generate (default: 1500)")
parser.add_argument("--out", type=str, default="generated_samples/styletts2",
                    help="Output directory for WAV files")
parser.add_argument("--voices-dir", type=str,
                    default="styletts2-ukrainian/voices",
                    help="Directory with .pt voice embedding files")
parser.add_argument("--styletts2-dir", type=str,
                    default="styletts2-ukrainian",
                    help="Path to styletts2-ukrainian repo root")
parser.add_argument("--speed-range", type=float, nargs=2, default=[0.85, 1.15],
                    metavar=("MIN", "MAX"),
                    help="Speed variation range (default: 0.85 1.15)")
parser.add_argument("--no-verbalize", "--no-verbalization", "--disable-verbalization",
                    dest="verbalize", action="store_false", default=True,
                    help="Вимкнути вербалізацію тексту (за замовчуванням увімкнено)")
parser.add_argument("--verbalize", dest="verbalize", action="store_true",
                    help="Увімкнути вербалізацію тексту")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

# ── Додаємо styletts2-ukrainian до шляху ───────────────────────────────────────
sys.path.insert(0, args.styletts2_dir)

# ── Імпорт моделі та залежностей ───────────────────────────────────────────────
try:
    from styletts2_inference.models import StyleTTS2
    from ipa_uk import ipa
    import stanza

    # Stanza NLP форсуємо на CPU (уникаємо cuDNN RNN проблем на sm_121)
    _orig_stanza_pipeline = stanza.Pipeline
    def _cpu_stanza_pipeline(*a, **kw):
        kw["device"] = "cpu"
        return _orig_stanza_pipeline(*a, **kw)
    stanza.Pipeline = _cpu_stanza_pipeline

    from ukrainian_word_stress import Stressifier, StressSymbol
except ImportError as e:
    print(f"[ERROR] Не вдалося імпортувати залежності StyleTTS2: {e}")
    print("Переконайтесь, що активовано venv: source styletts2-ukrainian/env-tts/bin/activate")
    sys.exit(1)

try:
    import soundfile as sf
    import resampy
except ImportError:
    print("[ERROR] Потрібно: pip install soundfile resampy  (в env-tts)")
    sys.exit(1)

# ── Ініціалізація ──────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Пристрій: {device}")

print("[INFO] Завантаження StyleTTS2 multispeaker моделі...")
model = StyleTTS2(hf_path="patriotyk/styletts2_ukrainian_multispeaker", device=device)

stressify = Stressifier()

verbalizer = None
if args.verbalize:
    try:
        from verbalizer import Verbalizer
        print("[INFO] Ініціалізація Verbalizer (вербалізація увімкнена)...")
        verbalizer = Verbalizer()
    except Exception as e:
        print(f"[WARN] Не вдалося ініціалізувати Verbalizer ({e}), продовжую без вербалізації.")
else:
    print("[INFO] Вербалізацію вимкнено користувачем (--no-verbalize).")

# ── Завантаження голосів ───────────────────────────────────────────────────────
voice_files = sorted(glob.glob(os.path.join(args.voices_dir, "*.pt")))
if not voice_files:
    print(f"[ERROR] Не знайдено .pt голосів у {args.voices_dir}")
    sys.exit(1)

print(f"[INFO] Знайдено {len(voice_files)} голосів:")
for v in voice_files:
    print(f"  - {os.path.basename(v)}")


def safe_load_style(path):
    with open(path, "rb") as f:
        header = f.read(20)
    if header.startswith(b"version https://git"):
        print(f"[WARN] {path} — Git LFS pointer, завантажую з HuggingFace...")
        from huggingface_hub import hf_hub_download
        import shutil
        rel = os.path.relpath(path, args.styletts2_dir)
        downloaded = hf_hub_download(
            repo_id="patriotyk/styletts2-ukrainian", repo_type="space", filename=rel
        )
        shutil.copy2(downloaded, path)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


print("[INFO] Завантаження стилів голосів...")
voice_styles = {}
for vf in voice_files:
    name = Path(vf).stem
    try:
        voice_styles[name] = safe_load_style(vf)
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name} — {e}")

if not voice_styles:
    print("[ERROR] Жоден голос не завантажився.")
    sys.exit(1)

voice_names = list(voice_styles.keys())

# ── Функція підготовки тексту ──────────────────────────────────────────────────
def verbalize_text(text: str) -> str:
    """Вербалізація чисел, скорочень та символів у слова."""
    if not text.strip():
        return text
    try:
        parts = text.splitlines()
        res = []
        for part in parts:
            if part.strip():
                res.append(verbalizer.process_text(part.strip())[0])
        return " ".join(res)
    except Exception as e:
        print(f"[WARN] Помилка вербалізації: {e}")
        return text

def prepare_text(raw: str, use_verbalize: bool = True) -> str:
    """Нормалізація → вербалізація (опціонально) → наголоси → IPA транскрипція."""
    t = raw.strip()
    if use_verbalize and verbalizer is not None:
        t = verbalize_text(t)
    t = t.strip()
    t = t.replace('"', '')
    t = normalize('NFKC', t)
    t = re.sub(r'[᠆‐‑‒–—―⁻₋−⸺⸻]', '-', t)
    # Додаємо крапку якщо нема кінцевого розділового знаку
    if t and t[-1] not in '.?!:-':
        t += '.'
    t = re.sub(r' - ', ': ', t)
    t = stressify(t)
    ps = ipa(t)
    return ps

# ── Текстові варіації wake word ────────────────────────────────────────────────
raw_word = args.word
verb_status = "увімкнена" if (args.verbalize and verbalizer is not None) else "вимкнена"
print(f"[INFO] Підготовка тексту '{raw_word}' (вербалізація: {verb_status})...")
ipa_text = prepare_text(raw_word, use_verbalize=args.verbalize)
print(f"[INFO] IPA транскрипція: {ipa_text!r}")

if not ipa_text:
    print("[ERROR] Порожня IPA транскрипція — перевірте слово.")
    sys.exit(1)

tokens_base = model.tokenizer.encode(ipa_text)
print(f"[INFO] Токени ({len(tokens_base)}): {tokens_base.tolist()}")

# ── Вихідна директорія ─────────────────────────────────────────────────────────
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Вихід: {out_dir}")

# ── Генерація ─────────────────────────────────────────────────────────────────
SPEED_MIN, SPEED_MAX = args.speed_range
total = args.count
generated = 0
skipped = 0

print(f"\n[START] Генерую {total} семплів слова '{raw_word}' (speed {SPEED_MIN}–{SPEED_MAX})...\n")

for i in range(total):
    voice_name = random.choice(voice_names)
    ref_s      = voice_styles[voice_name]
    speed      = round(random.uniform(SPEED_MIN, SPEED_MAX), 3)

    try:
        with torch.no_grad():
            wav = model(tokens_base, speed=speed, s_prev=ref_s)

        # StyleTTS2 повертає 24kHz тензор — конвертуємо в 16kHz mono numpy
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        wav = wav.squeeze()
        if wav.ndim > 1:
            wav = wav.mean(axis=0)  # stereo → mono

        # Ресемплінг 24000 → 16000
        wav_16k = resampy.resample(wav.astype(np.float32), 24000, 16000)

        # Нормалізація амплітуди
        peak = np.abs(wav_16k).max()
        if peak > 0:
            wav_16k = wav_16k / peak * 0.9

        out_path = out_dir / f"{i:05d}.wav"
        sf.write(str(out_path), wav_16k, 16000, subtype="PCM_16")

        generated += 1
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{total}] '{voice_name}' | speed={speed:.3f} → {out_path.name}")

    except Exception as e:
        skipped += 1
        print(f"  [SKIP #{i}] {voice_name}: {e}")

print(f"\n[DONE] Згенеровано: {generated}, пропущено: {skipped}")
print(f"Файли у: {out_dir.resolve()}")

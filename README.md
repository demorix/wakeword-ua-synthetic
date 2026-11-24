# 🇺🇦 Ukrainian Wake Word — "Agent" / "Aiva"

A complete pipeline for training a custom **wake word** (keyword activation) in Ukrainian, based on [microWakeWord](https://github.com/kahrendt/microWakeWord) and [StyleTTS2 Ukrainian](https://github.com/Romisnyi/styletts2-ukrainian).

> Pre-trained models: **"agent"** and **"aiva"** — ready to use on PC, server, or Raspberry Pi.

---

## 📋 Pipeline Overview

```
StyleTTS2 Ukrainian          microWakeWord
(synthetic WAV sample    →   (MixedNet neural network  →   .tflite model
 generation via TTS)          training INT8)                (streaming inference)
```

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `01_generate_samples.py` | Generate WAV samples via StyleTTS2 TTS |
| 2 | `02_download_augdata.py` | Download augmentation data (MIT RIR, negative datasets) |
| 3 | `03_generate_features.py` | Augmentation + conversion to mmap spectrograms |
| 4 | `04_train_model.py` | Train MixedNet, quantize → streaming TFLite |
| 5 | `05_test_inference.py` | Test on an audio file |
| 6 | `06_live_microphone.py` | Live test via microphone (PC / Raspberry Pi) |

---

## 🗂️ Repository Structure

```
ua_wakeword/
├── scripts/
│   ├── 01_generate_samples.py      # TTS → WAV (StyleTTS2)
│   ├── 02_download_augdata.py      # Download augmentation data
│   ├── 03_generate_features.py     # WAV → mmap spectrograms
│   ├── 03_train.py                 # Alternative training script
│   ├── 04_train_model.py           # Training + TFLite quantization
│   ├── 05_test_inference.py        # Test on a .wav file
│   └── 06_live_microphone.py       # Live microphone test
├── trained_models/
│   ├── agent_ukrainian_wakeword.tflite   # "agent" model
│   ├── agent_ukrainian_wakeword.json     # Metadata
│   ├── aiva_ukrainian_wakeword.tflite    # "aiva" model
│   └── aiva_ukrainian_wakeword.json      # Metadata
├── training_parameters.yaml        # Training configuration
├── test_parameters.yaml            # Dry-run test configuration
└── microWakeWord/                  # Submodule / library copy
```

> **Not included in the repository** (large files, reproducible via scripts):
> `augmentation_data/` (~26 GB), `generated_samples/` (~106 MB),
> `generated_augmented_features/` (~1.8 GB), `env-mww/` (venv)

---

## ⚡ Quick Start

### Requirements

- Python 3.10+
- CUDA GPU (recommended for TTS) or CPU
- ~30 GB free disk space

### 1. Clone

```bash
git clone https://github.com/demorix/wakeword-ua-synthetic.git
cd wakeword-ua-synthetic

# Clone required dependencies
git clone https://github.com/kahrendt/microWakeWord
git clone https://github.com/Romisnyi/styletts2-ukrainian
```

### 2. Environments

```bash
# Environment for microWakeWord (training, testing, live detection)
python -m venv env-mww
source env-mww/bin/activate
pip install -r microWakeWord/requirements.txt
pip install sounddevice soundfile resampy mmap_ninja tqdm

# Environment for StyleTTS2 (sample generation)
cd styletts2-ukrainian
python -m venv env-tts
source env-tts/bin/activate
pip install -r requirements.txt
cd ..
```

### 3. Generate Samples

```bash
source styletts2-ukrainian/env-tts/bin/activate
python scripts/01_generate_samples.py --word "агент" --count 1500 \
    --out generated_samples/styletts2
```

### 4. Download Augmentation Data

```bash
source env-mww/bin/activate
python scripts/02_download_augdata.py --skip-audioset
```

### 5. Generate Spectrogram Features

```bash
python scripts/03_generate_features.py \
    --samples-dir generated_samples/styletts2
```

### 6. Train the Model

```bash
python scripts/04_train_model.py
# or via config:
python scripts/03_train.py --word "агент" \
    --samples-dir generated_samples/styletts2
```

### 7. Test

```bash
# On an audio file:
python scripts/05_test_inference.py \
    --model trained_models/agent_ukrainian_wakeword.tflite \
    --audio generated_samples/styletts2/00000.wav

# Live via microphone:
python scripts/06_live_microphone.py \
    --model trained_models/agent_ukrainian_wakeword.tflite \
    --cutoff 0.95
```

---

## 🎯 Pre-trained Models

| Wake word | File | Threshold |
|-----------|------|-----------|
| **агент** (agent) | `trained_models/agent_ukrainian_wakeword.tflite` | 0.95 |
| **аіва** (aiva)   | `trained_models/aiva_ukrainian_wakeword.tflite`  | 0.95 |

Models are in **streaming TFLite INT8** format — compatible with:
- Home Assistant (via Wyoming / openWakeWord integration)
- Raspberry Pi Zero 2W+
- ESP32-S3 (with TFLite Micro)
- Any Linux/macOS/Windows host

---

## 🏗️ Tech Stack

| Component | Details |
|-----------|---------|
| TTS synthesis | [StyleTTS2 Ukrainian](https://github.com/Romisnyi/styletts2-ukrainian) (multispeaker) |
| Wake word engine | [microWakeWord](https://github.com/kahrendt/microWakeWord) (MixedNet) |
| Model format | Streaming TFLite INT8 |
| Sample rate | 16 kHz, mono |
| Augmentation | MIT RIR, FMA music, speech/no-speech negative datasets |
| Window step | 10 ms |

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

- [kahrendt/microWakeWord](https://github.com/kahrendt/microWakeWord) — wake word framework
- [Romisnyi/styletts2-ukrainian](https://github.com/Romisnyi/styletts2-ukrainian) — Ukrainian TTS system

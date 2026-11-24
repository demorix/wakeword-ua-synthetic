# 🇺🇦 Ukrainian Wake Word — «Агент» / «Аіва»

Повний пайплайн для тренування власного **wake word** (ключового слова активації) українською мовою на основі [microWakeWord](https://github.com/kahrendt/microWakeWord) і [StyleTTS2 Ukrainian](https://github.com/Romisnyi/styletts2-ukrainian).

> Натреновані моделі: **«агент»** та **«аіва»** — готові до використання на ПК, сервері або Raspberry Pi.

---

## 📋 Опис пайплайну

```
StyleTTS2 Ukrainian          microWakeWord
(генерація синтетичних   →   (тренування нейромережі   →   .tflite модель
 WAV-семплів)                 MixedNet INT8)                (стрімінговий інференс)
```

| Крок | Скрипт | Призначення |
|------|--------|-------------|
| 1 | `01_generate_samples.py` | Генерація WAV-семплів через StyleTTS2 TTS |
| 2 | `02_download_augdata.py` | Завантаження аугментаційних даних (MIT RIR, негативні датасети) |
| 3 | `03_generate_features.py` | Аугментація + перетворення у mmap-спектрограми |
| 4 | `04_train_model.py` | Тренування MixedNet, квантизація → streaming TFLite |
| 5 | `05_test_inference.py` | Тестування на аудіофайлі |
| 6 | `06_live_microphone.py` | Live-тест через мікрофон (ПК / Raspberry Pi) |

---

## 🗂️ Структура репозиторію

```
ua_wakeword/
├── scripts/
│   ├── 01_generate_samples.py      # TTS → WAV (StyleTTS2)
│   ├── 02_download_augdata.py      # Завантаження аугментаційних даних
│   ├── 03_generate_features.py     # WAV → mmap spectrograms
│   ├── 03_train.py                 # (альтернативний) тренувальний скрипт
│   ├── 04_train_model.py           # Тренування + TFLite квантизація
│   ├── 05_test_inference.py        # Тест на .wav файлі
│   └── 06_live_microphone.py       # Live тест через мікрофон
├── trained_models/
│   ├── agent_ukrainian_wakeword.tflite   # Модель «агент»
│   ├── agent_ukrainian_wakeword.json     # Метадані
│   ├── aiva_ukrainian_wakeword.tflite    # Модель «аіва»
│   └── aiva_ukrainian_wakeword.json      # Метадані
├── training_parameters.yaml        # Конфіг тренування
├── test_parameters.yaml            # Конфіг тест-рану
└── microWakeWord/                  # Субмодуль / копія бібліотеки
```

> **Не включено у репозиторій** (великі файли, відтворюються скриптами):
> `augmentation_data/` (~26 GB), `generated_samples/` (~106 MB),
> `generated_augmented_features/` (~1.8 GB), `env-mww/` (venv)

---

## ⚡ Швидкий старт

### Вимоги

- Python 3.10+
- CUDA GPU (рекомендовано для TTS) або CPU
- ~30 GB вільного місця на диску

### 1. Клонування

```bash
git clone https://github.com/YOUR_USERNAME/ua_wakeword.git
cd ua_wakeword

# Клонуємо залежні репозиторії
git clone https://github.com/kahrendt/microWakeWord
git clone https://github.com/Romisnyi/styletts2-ukrainian
```

### 2. Середовища

```bash
# Середовище для microWakeWord (тренування, тест, live)
python -m venv env-mww
source env-mww/bin/activate
pip install -r microWakeWord/requirements.txt
pip install sounddevice soundfile resampy mmap_ninja tqdm

# Середовище для StyleTTS2 (генерація семплів)
cd styletts2-ukrainian
python -m venv env-tts
source env-tts/bin/activate
pip install -r requirements.txt
cd ..
```

### 3. Генерація семплів

```bash
source styletts2-ukrainian/env-tts/bin/activate
python scripts/01_generate_samples.py --word "агент" --count 1500 \
    --out generated_samples/styletts2
```

### 4. Завантаження аугментаційних даних

```bash
source env-mww/bin/activate
python scripts/02_download_augdata.py --skip-audioset
```

### 5. Генерація ознак (спектрограм)

```bash
python scripts/03_generate_features.py \
    --samples-dir generated_samples/styletts2
```

### 6. Тренування моделі

```bash
python scripts/04_train_model.py
# або через конфіг:
python scripts/03_train.py --word "агент" \
    --samples-dir generated_samples/styletts2
```

### 7. Тестування

```bash
# На аудіофайлі:
python scripts/05_test_inference.py \
    --model trained_models/agent_ukrainian_wakeword.tflite \
    --audio generated_samples/styletts2/00000.wav

# Live через мікрофон:
python scripts/06_live_microphone.py \
    --model trained_models/agent_ukrainian_wakeword.tflite \
    --cutoff 0.95
```

---

## 🎯 Готові моделі

| Wake word | Файл | Поріг |
|-----------|------|-------|
| **агент** | `trained_models/agent_ukrainian_wakeword.tflite` | 0.95 |
| **аіва**  | `trained_models/aiva_ukrainian_wakeword.tflite`  | 0.95 |

Моделі у форматі **streaming TFLite INT8** — підходять для:
- Home Assistant (via Wyoming / openWakeWord інтеграція)
- Raspberry Pi Zero 2W+
- ESP32-S3 (з TFLite Micro)
- Будь-якого Linux/macOS/Windows хосту

---

## 🏗️ Технічний стек

| Компонент | Деталі |
|-----------|--------|
| TTS синтез | [StyleTTS2 Ukrainian](https://github.com/Romisnyi/styletts2-ukrainian) (multispeaker) |
| Wake word engine | [microWakeWord](https://github.com/kahrendt/microWakeWord) (MixedNet) |
| Формат моделі | Streaming TFLite INT8 |
| Частота дискретизації | 16 kHz, mono |
| Аугментація | MIT RIR, FMA music, speech/no-speech негативні датасети |
| Крок вікна | 10 ms |

---

## 📄 Ліцензія

MIT — дивись [LICENSE](LICENSE).

---

## 🙏 Подяки

- [kahrendt/microWakeWord](https://github.com/kahrendt/microWakeWord) — фреймворк wake word
- [Romisnyi/styletts2-ukrainian](https://github.com/Romisnyi/styletts2-ukrainian) — українська TTS система

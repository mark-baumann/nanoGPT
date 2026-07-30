# 🧠 nanoGPT

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![W&B](https://img.shields.io/badge/W%26B-Tracking-orange.svg)](https://wandb.ai/)

**GPT-2 Modellarchitektur in reinem PyTorch** — Training, Sampling und Architektur-Exploration nach Andrej Karpathys nanoGPT.

## 📋 Beschreibung

Dieses Repository enthält eine vollständige, eigenständige Implementierung der GPT-2-Architektur in PyTorch. Basierend auf Karpathys nanoGPT/minGPT bietet es Training, Textgenerierung, Attention-Visualisierung und W&B-Integration — alles in klarem, gut dokumentiertem Code.

- **GPT-2 Architektur** — CausalSelfAttention, MLP, LayerNorm, Transformer-Blöcke
- **Training** — DDP, Gradient Accumulation, Cosine Decay, Mixed Precision
- **Sampling** — Autoregressive Textgenerierung mit Temperatur, Top-K, Top-P
- **Konfiguration** — Flexible Configs für GPT-2 small/medium/large/xl und benutzerdefinierte Modelle

## ✨ Features

- 🏗️ **Reine PyTorch-Implementierung** — Keine Abhängigkeit von HuggingFace Transformers
- ⚡ **Flash Attention** — Automatische Nutzung von `scaled_dot_product_attention` wenn verfügbar
- 🚀 **DDP Training** — Multi-GPU Distributed Data Parallel
- 📊 **W&B Tracking** — Loss, Learning Rate, GPU-Auslastung, Gradient-Norm
- 🎨 **Attention-Visualisierung** — Heatmaps der Attention-Weights
- 📝 **Textgenerierung** — Mit Temperatur, Top-K und Top-P Sampling
- 🔧 **Flexible Konfiguration** — Config-Dateien für verschiedene Modellgrößen
- 🧪 **Test-Suite** — pytest-Tests für Modell, Training und Sampling

## 🚀 Installation

```bash
# Repository klonen
git clone https://github.com/mark-baumann/nanoGPT.git
cd nanoGPT

# Virtuelle Umgebung erstellen
python3 -m venv .venv
source .venv/bin/activate

# PyTorch installieren (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Abhängigkeiten
pip install -r requirements.txt
```

## 🎮 Nutzung

### Training

```bash
# Standard-Training (GPT-2 small Konfiguration)
python train.py

# Baby-Modell für schnelle Experimente
python train.py --config=gpt2-baby

# Auf CPU trainieren
python train.py --device=cpu

# Mit benutzerdefinierter Konfiguration
python train.py --config=config/train_shakespeare_char.py
```

### Textgenerierung

```bash
# Text aus trainiertem Checkpoint generieren
python sample.py --checkpoint=out/ckpt.pt --prompt="Es war einmal"

# Mehrere Samples mit höherer Temperatur
python sample.py --checkpoint=out/ckpt.pt --num_samples=5 --max_new_tokens=200 --temperature=0.8

# Mit Top-K Sampling
python sample.py --checkpoint=out/ckpt.pt --top_k=40
```

### Attention-Visualisierung

```bash
python visualize_attention.py --checkpoint=out/ckpt.pt --prompt="Die Zukunft der KI"
```

### Streamlit-App

```bash
streamlit run app.py
```

### Tests

```bash
pytest tests/ -v
```

## 🏗️ Tech-Stack

| Komponente | Technologie |
|---|---|
| **Sprache** | Python 3.10+ |
| **Framework** | PyTorch 2.0+ |
| **Training** | DDP, Mixed Precision (AMP) |
| **UI** | Streamlit |
| **Tracking** | Weights & Biases |
| **Testing** | pytest |

## 📁 Projektstruktur

```
nanoGPT/
├── model.py                # GPT-2 Architektur (GPT, GPTConfig, CausalSelfAttention)
├── train.py                # Training-Loop mit DDP, Gradient Accumulation
├── sample.py               # Textgenerierung aus Checkpoint
├── config.py               # Konfigurations-Loader
├── configurator.py         # Config-Overlay-System
├── bench.py                # Benchmarking-Tool
├── visualize_attention.py  # Attention-Heatmap-Visualisierung
├── app.py                  # Streamlit-App
├── wandb_utils.py          # W&B Integration
├── config/                 # Vordefinierte Konfigurationen
│   ├── train_gpt2.py
│   ├── train_shakespeare_char.py
│   ├── finetune_shakespeare.py
│   ├── eval_gpt2.py
│   ├── eval_gpt2_medium.py
│   ├── eval_gpt2_large.py
│   └── eval_gpt2_xl.py
└── tests/
    ├── test_model.py
    ├── test_train.py
    ├── test_sample.py
    ├── test_config.py
    └── test_wandb_utils.py
```

## 👤 Autor

**Mark Baumann** — [GitHub](https://github.com/mark-baumann)

---

*Basierend auf Andrej Karpathys [nanoGPT](https://github.com/karpathy/nanoGPT). Für Fragen oder Beiträge: Issue erstellen oder Pull Request öffnen.*

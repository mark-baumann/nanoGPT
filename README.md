# nanoGPT

Eine vollständige Implementierung von Karpathys nanoGPT in reinem PyTorch — **ohne** die `transformers`-Bibliothek.

Dieses Repository enthält eine GPT-2-Architektur mit Training-Loop, Konfigurationen für verschiedene Modellgrößen und Text-Generierung. Der Code ist minimalistisch, gut dokumentiert und direkt lauffähig.

## 📁 Dateien

| Datei | Beschreibung |
|-------|-------------|
| `model.py` | GPT-2 Architektur: `CausalSelfAttention`, `MLP`, `LayerNorm`, `Block`, `GPT` |
| `config.py` | Vordefinierte Konfigurationen für GPT-2 small/medium/large/xl, Baby, Micro, Nano |
| `train.py` | Vollständiger Training-Loop mit DDP, Mixed Precision, Cosine LR Schedule, Checkpointing |
| `sample.py` | Text-Generierung aus trainiertem Checkpoint mit Temperatur- und Top-k-Steuerung |

## 🏗️ Architektur

### `model.py`

Implementiert die GPT-2-Architektur nach dem Paper *"Language Models are Unsupervised Multitask Learners"*:

- **LayerNorm** — Pre-Norm (vor Attention und MLP), mit optionalem Bias
- **CausalSelfAttention** — Multi-Head Self-Attention mit kausaler Maske, Flash-Attention-Support (PyTorch 2.0+)
- **MLP** — Feed-Forward mit GELU-Aktivierung (tanh-Approximation wie GPT-2)
- **Block** — Transformer-Block: `x = x + Attn(LN(x))`, `x = x + MLP(LN(x))`
- **GPT** — Das vollständige Modell mit Token/Position Embeddings, Weight Tying, `generate()`-Methode

Besonderheiten:
- **Weight Tying**: `wte` und `lm_head` teilen sich die Gewichte
- **Scaled Init**: Residual-Projektionen werden mit `1/√(2·n_layer)` skaliert
- **Flash Attention**: Automatisch via `F.scaled_dot_product_attention` wenn verfügbar
- **AdamW**: Separate Weight-Decay-Behandlung (nur auf Matrizen, nicht auf Biases/LayerNorms)

### `config.py`

Vordefinierte Modell-Konfigurationen:

| Name | Parameter | Layer | Heads | Embedding |
|------|-----------|-------|-------|-----------|
| `gpt2-nano` | ~0.8M | 2 | 2 | 64 |
| `gpt2-micro` | ~4M | 4 | 4 | 128 |
| `gpt2-baby` | ~10M | 6 | 6 | 384 |
| `gpt2-small` | 124M | 12 | 12 | 768 |
| `gpt2-medium` | 355M | 24 | 16 | 1024 |
| `gpt2-large` | 774M | 36 | 20 | 1280 |
| `gpt2-xl` | 1.5B | 48 | 25 | 1600 |

### `train.py`

Der Training-Loop unterstützt:

- **DDP** (Distributed Data Parallel) für Multi-GPU-Training
- **Gradient Accumulation** für effektiv größere Batch-Größen
- **Mixed Precision** (float16, bfloat16, float32)
- **Cosine Learning Rate Schedule** mit linearem Warmup
- **Gradient Clipping**
- **Evaluation** während des Trainings
- **Checkpointing** (speichert bestes Modell nach Val-Loss)
- **torch.compile()** für PyTorch 2.0+

### `sample.py`

Generiert Text aus einem trainierten Checkpoint:

- **Temperatur-Steuerung**: `<1.0` = deterministischer, `>1.0` = kreativer
- **Top-k Sampling**: Beschränkt auf die k wahrscheinlichsten Tokens
- **Mehrere Samples**: Beliebig viele Generierungen aus demselben Prompt

## 🚀 Schnellstart

### Voraussetzungen

- Python 3.10+
- PyTorch 2.0+ (`pip install torch`)
- Optional: numpy (für `.bin`-Daten), CUDA-fähige GPU

### Installation

```bash
git clone <repo-url>
cd nanoGPT
pip install torch numpy
```

### Training

```bash
# Baby-Modell für schnelle Experimente (CPU, ~1 Minute)
python train.py --config=gpt2-baby --device=cpu --max_iters=100

# GPT-2 Small (benötigt GPU und Daten)
python train.py --config=gpt2-small --device=cuda
```

### Text-Generierung

```bash
# Aus trainiertem Checkpoint generieren
python sample.py --checkpoint=out/ckpt.pt --prompt="Es war einmal" --max_new_tokens=200

# Mit Temperatur und Top-k
python sample.py --checkpoint=out/ckpt.pt --temperature=0.8 --top_k=40

# Mehrere Samples
python sample.py --checkpoint=out/ckpt.pt --num_samples=3
```

### Import als Bibliothek

```python
from model import GPT, GPTConfig
from config import get_config

# Modell erstellen
config = get_config("gpt2-small")
model = GPT(config)

# Text generieren
import torch
start_ids = torch.randint(0, config.vocab_size, (1, 10))
generated = model.generate(start_ids, max_new_tokens=50, temperature=0.8)
```

## 📊 Daten

Das Training erwartet Daten im `data/<dataset>/`-Verzeichnis als `train.bin` und `val.bin` (uint16 numpy memmap). Falls keine Daten vorhanden sind, wird automatisch ein einfacher Fallback-Datensatz generiert.

Für echtes Training wird ein vorverarbeiteter Datensatz benötigt, z.B.:
- OpenWebText
- Shakespeare (Zeichen-basiert)
- Eigene Textdaten

## 🔧 Konfiguration

Alle Trainings-Parameter sind in `config.py` als Dictionary definiert und können per CLI überschrieben werden:

```bash
python train.py --config=gpt2-small --device=cuda --max_iters=10000 --eval_interval=500
```

## 📝 Lizenz

MIT — frei verwendbar für Forschung und kommerzielle Projekte.

## 🙏 Credits

Basierend auf [Andrej Karpathys nanoGPT](https://github.com/karpathy/nanoGPT) und [minGPT](https://github.com/karpathy/minGPT).

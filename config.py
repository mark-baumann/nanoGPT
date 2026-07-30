"""
Konfigurationen für verschiedene GPT-Modellgrößen und Trainings-Setups.
Alle Größen basieren auf der GPT-2 / GPT-3 Architektur.
"""

from model import GPTConfig

# ─────────────────────────────────────────────────────────────
# Modell-Konfigurationen
# ─────────────────────────────────────────────────────────────

# GPT-2 Small (124M Parameter) — das Original
gpt2_small = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    bias=True,
)

# GPT-2 Medium (355M Parameter)
gpt2_medium = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=24,
    n_head=16,
    n_embd=1024,
    dropout=0.0,
    bias=True,
)

# GPT-2 Large (774M Parameter)
gpt2_large = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=36,
    n_head=20,
    n_embd=1280,
    dropout=0.0,
    bias=True,
)

# GPT-2 XL (1.5B Parameter)
gpt2_xl = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=48,
    n_head=25,
    n_embd=1600,
    dropout=0.0,
    bias=True,
)

# Baby GPT — für schnelle Experimente und Tests (ca. 10M Parameter)
gpt2_baby = GPTConfig(
    block_size=256,
    vocab_size=50304,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True,
)

# Micro GPT — minimal, für Unit-Tests und Debugging
gpt2_micro = GPTConfig(
    block_size=64,
    vocab_size=50304,
    n_layer=4,
    n_head=4,
    n_embd=128,
    dropout=0.0,
    bias=True,
)

# Nano GPT — winzig, für schnelle Smoke-Tests
gpt2_nano = GPTConfig(
    block_size=32,
    vocab_size=50304,
    n_layer=2,
    n_head=2,
    n_embd=64,
    dropout=0.0,
    bias=True,
)

# ─────────────────────────────────────────────────────────────
# Trainings-Konfigurationen
# ─────────────────────────────────────────────────────────────

# Standard-Training auf OpenWebText / eigenem Datensatz
train_gpt2_small = dict(  # noqa: C408
    # Daten
    dataset="openwebtext",          # Name des Datensatzes
    gradient_accumulation_steps=5,  # Akkumulierte Gradienten-Schritte
    batch_size=12,                  # Batch-Größe pro GPU
    block_size=1024,                # Kontextlänge

    # Modell
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    bias=True,

    # Optimizer
    learning_rate=6e-4,             # Maximale Learning Rate
    max_iters=600000,               # Maximale Trainings-Iterationen
    weight_decay=1e-1,              # Weight Decay
    beta1=0.9,                      # Adam beta1
    beta2=0.95,                     # Adam beta2
    grad_clip=1.0,                  # Gradient Clipping

    # Learning Rate Schedule
    decay_lr=True,                  # LR Decay aktivieren
    warmup_iters=2000,              # Warmup-Iterationen
    lr_decay_iters=600000,          # Iterationen für LR-Decay
    min_lr=6e-5,                    # Minimale Learning Rate

    # Evaluation
    eval_interval=2000,             # Alle N Iterationen evaluieren
    eval_iters=200,                 # Anzahl Eval-Batches
    log_interval=10,                # Logging-Intervall

    # System
    device="cuda",                  # "cuda", "cpu", "mps"
    compile=True,                   # torch.compile() verwenden
    dtype="float16",                # "float32", "bfloat16", "float16"
)

# Schnelles Training für Baby-Modell (Experimente)
train_gpt2_baby = dict(  # noqa: C408
    dataset="shakespeare_char",
    gradient_accumulation_steps=1,
    batch_size=64,
    block_size=256,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True,
    learning_rate=1e-3,
    max_iters=5000,
    weight_decay=1e-1,
    beta1=0.9,
    beta2=0.99,
    grad_clip=1.0,
    decay_lr=True,
    warmup_iters=100,
    lr_decay_iters=5000,
    min_lr=1e-4,
    eval_interval=500,
    eval_iters=200,
    log_interval=10,
    device="cpu",
    compile=False,
    dtype="float32",
)

# ─────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────

def get_config(name: str) -> GPTConfig:
    """Gibt eine Modell-Konfiguration anhand des Namens zurück."""
    configs = {
        "gpt2": gpt2_small,
        "gpt2-small": gpt2_small,
        "gpt2-medium": gpt2_medium,
        "gpt2-large": gpt2_large,
        "gpt2-xl": gpt2_xl,
        "gpt2-baby": gpt2_baby,
        "gpt2-micro": gpt2_micro,
        "gpt2-nano": gpt2_nano,
    }
    if name not in configs:
        raise ValueError(f"Unbekannte Konfiguration: {name}. Verfügbar: {list(configs.keys())}")
    return configs[name]


def get_train_config(name: str) -> dict:
    """Gibt eine Trainings-Konfiguration anhand des Namens zurück."""
    configs = {
        "gpt2-small": train_gpt2_small,
        "gpt2-baby": train_gpt2_baby,
    }
    if name not in configs:
        raise ValueError(f"Unbekannte Trainings-Konfiguration: {name}. Verfügbar: {list(configs.keys())}")
    return configs[name]

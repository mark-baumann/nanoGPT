"""nanoGPT — GPT-2 Implementierung in reinem PyTorch.

Enthält: GPT-2 Architektur, Training-Loop, Text-Generierung,
Konfigurationen für verschiedene Modellgrößen.
"""

from model import GPT, GPTConfig, LayerNorm, CausalSelfAttention, MLP, Block
from config import get_config, get_train_config

__version__ = "0.2.0"
__all__ = [
    "GPT",
    "GPTConfig",
    "LayerNorm",
    "CausalSelfAttention",
    "MLP",
    "Block",
    "get_config",
    "get_train_config",
]

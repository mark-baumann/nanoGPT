# noqa: N999
"""nanoGPT — GPT-2 Implementierung in reinem PyTorch.

Enthält: GPT-2 Architektur, Training-Loop, Text-Generierung,
Konfigurationen für verschiedene Modellgrößen.
"""

from config import get_config, get_train_config
from model import GPT, MLP, Block, CausalSelfAttention, GPTConfig, LayerNorm

__version__ = "0.2.0"
__all__ = [
    "GPT",
    "MLP",
    "Block",
    "CausalSelfAttention",
    "GPTConfig",
    "LayerNorm",
    "get_config",
    "get_train_config",
]

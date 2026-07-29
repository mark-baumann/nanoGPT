"""Unit-Tests für train.py — Training-Loop-Hilfsfunktionen."""

import pytest
import torch
from train import get_batch, get_lr, _generate_fallback_data


class TestGetBatch:
    """Tests für get_batch()."""

    def test_shapes(self):
        data = torch.arange(1000)
        x, y = get_batch(data, block_size=16, batch_size=4, device="cpu")
        assert x.shape == (4, 16)
        assert y.shape == (4, 16)

    def test_target_is_shifted(self):
        """y sollte um 1 Position gegenüber x verschoben sein."""
        data = torch.arange(100)
        x, y = get_batch(data, block_size=8, batch_size=1, device="cpu")
        # Für jedes Element: y[i] == x[i] + 1 (weil data = arange)
        assert torch.all(y[0] == x[0] + 1)

    def test_device(self):
        data = torch.arange(1000)
        x, y = get_batch(data, block_size=8, batch_size=2, device="cpu")
        assert x.device.type == "cpu"
        assert y.device.type == "cpu"


class TestGetLR:
    """Tests für get_lr() — Learning Rate Schedule."""

    def test_warmup_start(self):
        """Bei Iteration 0 sollte LR > 0 sein (nicht 0)."""
        lr = get_lr(it=0, learning_rate=1e-3, min_lr=1e-4, warmup_iters=100, lr_decay_iters=1000)
        assert lr > 0

    def test_warmup_increases(self):
        """LR sollte während Warmup monoton steigen."""
        lr0 = get_lr(it=0, learning_rate=1e-3, min_lr=1e-4, warmup_iters=100, lr_decay_iters=1000)
        lr50 = get_lr(it=50, learning_rate=1e-3, min_lr=1e-4, warmup_iters=100, lr_decay_iters=1000)
        assert lr50 > lr0

    def test_warmup_end(self):
        """Am Ende des Warmups sollte LR ≈ learning_rate sein."""
        lr = get_lr(it=99, learning_rate=1e-3, min_lr=1e-4, warmup_iters=100, lr_decay_iters=1000)
        assert abs(lr - 1e-3) < 1e-5

    def test_cosine_decay(self):
        """Nach Warmup sollte LR per Cosine Decay sinken."""
        lr_start = get_lr(it=100, learning_rate=1e-3, min_lr=1e-4, warmup_iters=100, lr_decay_iters=1000)
        lr_mid = get_lr(it=550, learning_rate=1e-3, min_lr=1e-4, warmup_iters=100, lr_decay_iters=1000)
        assert lr_mid < lr_start

    def test_min_lr(self):
        """Nach lr_decay_iters sollte LR == min_lr sein."""
        lr = get_lr(it=2000, learning_rate=1e-3, min_lr=1e-4, warmup_iters=100, lr_decay_iters=1000)
        assert lr == 1e-4

    def test_no_warmup(self):
        """Ohne Warmup (warmup_iters=0) sollte LR sofort mit Decay starten."""
        lr = get_lr(it=0, learning_rate=1e-3, min_lr=1e-4, warmup_iters=0, lr_decay_iters=1000)
        assert lr == 1e-3  # Start bei learning_rate


class TestFallbackData:
    """Tests für _generate_fallback_data()."""

    def test_returns_tuple(self):
        train_data, val_data = _generate_fallback_data()
        assert isinstance(train_data, torch.Tensor)
        assert isinstance(val_data, torch.Tensor)

    def test_train_larger_than_val(self):
        train_data, val_data = _generate_fallback_data()
        assert len(train_data) > len(val_data)

    def test_split_ratio(self):
        train_data, val_data = _generate_fallback_data()
        total = len(train_data) + len(val_data)
        ratio = len(train_data) / total
        assert 0.85 <= ratio <= 0.95  # ~90/10 Split

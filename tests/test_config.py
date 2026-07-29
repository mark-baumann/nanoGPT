"""Unit-Tests für config.py — Konfigurationen."""

import pytest
from model import GPTConfig
from config import get_config, get_train_config


class TestGetConfig:
    """Tests für get_config()."""

    def test_known_configs(self):
        for name in ["gpt2", "gpt2-small", "gpt2-medium", "gpt2-large",
                      "gpt2-xl", "gpt2-baby", "gpt2-micro", "gpt2-nano"]:
            cfg = get_config(name)
            assert isinstance(cfg, GPTConfig)

    def test_unknown_config(self):
        with pytest.raises(ValueError, match="Unbekannte Konfiguration"):
            get_config("nonexistent")

    def test_gpt2_small_values(self):
        cfg = get_config("gpt2-small")
        assert cfg.n_layer == 12
        assert cfg.n_head == 12
        assert cfg.n_embd == 768

    def test_gpt2_nano_values(self):
        cfg = get_config("gpt2-nano")
        assert cfg.n_layer == 2
        assert cfg.n_head == 2
        assert cfg.n_embd == 64
        assert cfg.block_size == 32


class TestGetTrainConfig:
    """Tests für get_train_config()."""

    def test_known_configs(self):
        for name in ["gpt2-small", "gpt2-baby"]:
            cfg = get_train_config(name)
            assert isinstance(cfg, dict)
            assert "learning_rate" in cfg
            assert "max_iters" in cfg
            assert "batch_size" in cfg

    def test_unknown_config(self):
        with pytest.raises(ValueError, match="Unbekannte Trainings-Konfiguration"):
            get_train_config("nonexistent")

    def test_gpt2_small_train(self):
        cfg = get_train_config("gpt2-small")
        assert cfg["learning_rate"] == 6e-4
        assert cfg["max_iters"] == 600000
        assert cfg["n_layer"] == 12

    def test_gpt2_baby_train(self):
        cfg = get_train_config("gpt2-baby")
        assert cfg["learning_rate"] == 1e-3
        assert cfg["max_iters"] == 5000
        assert cfg["n_layer"] == 6

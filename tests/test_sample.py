"""Unit-Tests für sample.py — Text-Generierung."""

import pytest
import torch
import tempfile
import os
from model import GPT, GPTConfig
from sample import encode_prompt, decode_tokens, load_model


class TestEncodeDecode:
    """Tests für encode_prompt und decode_tokens."""

    def test_encode_shape(self):
        tokens = encode_prompt("Hello", device="cpu")
        assert tokens.dim() == 2
        assert tokens.size(0) == 1
        assert tokens.size(1) == 5

    def test_encode_empty(self):
        tokens = encode_prompt("", device="cpu")
        assert tokens.size(1) == 0

    def test_roundtrip(self):
        text = "Hello World 123"
        tokens = encode_prompt(text, device="cpu")
        decoded = decode_tokens(tokens)
        assert decoded == text

    def test_roundtrip_special_chars(self):
        text = "ABC, def! 123?"
        tokens = encode_prompt(text, device="cpu")
        decoded = decode_tokens(tokens)
        assert decoded == text

    def test_decode_1d_tensor(self):
        tokens = torch.tensor([0, 1, 2, 3])
        result = decode_tokens(tokens)
        assert len(result) == 4

    def test_decode_2d_tensor(self):
        tokens = torch.tensor([[0, 1, 2, 3]])
        result = decode_tokens(tokens)
        assert len(result) == 4


class TestLoadModel:
    """Tests für load_model()."""

    def test_load_from_checkpoint(self):
        """Erstellt einen temporären Checkpoint und lädt ihn."""
        cfg = GPTConfig(block_size=32, vocab_size=100, n_layer=2, n_head=2, n_embd=64)
        model = GPT(cfg)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint = {
                "model": model.state_dict(),
                "model_config": cfg,
                "iter_num": 42,
                "best_val_loss": 1.5,
            }
            torch.save(checkpoint, f.name)
            tmp_path = f.name

        try:
            loaded = load_model(tmp_path, device="cpu")
            assert isinstance(loaded, GPT)
            # Parameter sollten gleich sein
            for p1, p2 in zip(model.parameters(), loaded.parameters()):
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(tmp_path)

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path.pt", device="cpu")

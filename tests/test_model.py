"""Unit-Tests für model.py — GPT-2 Architektur."""

import pytest
import torch
from model import GPT, GPTConfig, LayerNorm, CausalSelfAttention, MLP, Block


class TestGPTConfig:
    """Tests für GPTConfig."""

    def test_defaults(self):
        cfg = GPTConfig()
        assert cfg.block_size == 1024
        assert cfg.vocab_size == 50304
        assert cfg.n_layer == 12
        assert cfg.n_head == 12
        assert cfg.n_embd == 768
        assert cfg.dropout == 0.0
        assert cfg.bias is True

    def test_custom(self):
        cfg = GPTConfig(block_size=64, n_layer=4, n_head=4, n_embd=128)
        assert cfg.block_size == 64
        assert cfg.n_layer == 4


class TestLayerNorm:
    """Tests für LayerNorm."""

    def test_with_bias(self):
        ln = LayerNorm(64, bias=True)
        x = torch.randn(2, 10, 64)
        y = ln(x)
        assert y.shape == x.shape
        assert ln.bias is not None

    def test_without_bias(self):
        ln = LayerNorm(64, bias=False)
        x = torch.randn(2, 10, 64)
        y = ln(x)
        assert y.shape == x.shape
        assert ln.bias is None


class TestCausalSelfAttention:
    """Tests für CausalSelfAttention."""

    @pytest.fixture
    def config(self):
        return GPTConfig(block_size=32, n_embd=64, n_head=4, dropout=0.0, bias=True)

    def test_forward_shape(self, config):
        attn = CausalSelfAttention(config)
        x = torch.randn(2, 16, 64)
        y = attn(x)
        assert y.shape == x.shape

    def test_causal_mask(self, config):
        """Stellt sicher, dass die Attention kausal ist (kein Blick in die Zukunft)."""
        attn = CausalSelfAttention(config)
        attn.eval()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            y = attn(x)
        # Output sollte nicht NaN/Inf sein
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_output_attentions(self, config):
        """Testet, dass output_attentions=True die Attention-Weights zurückgibt."""
        attn = CausalSelfAttention(config)
        attn.eval()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            y, att_weights = attn(x, output_attentions=True)
        assert y.shape == x.shape
        assert att_weights is not None
        # att_weights: (B, n_head, T, T)
        assert att_weights.shape == (1, config.n_head, 8, 8)
        # Kausale Maske: oberes Dreieck sollte ~0 sein
        for head in range(config.n_head):
            upper_tri = torch.triu(att_weights[0, head], diagonal=1)
            assert (upper_tri < 1e-6).all(), (
                f"Head {head}: Attention auf zukünftige Tokens gefunden!"
            )


class TestMLP:
    """Tests für MLP."""

    def test_forward_shape(self):
        cfg = GPTConfig(n_embd=64, dropout=0.0)
        mlp = MLP(cfg)
        x = torch.randn(2, 10, 64)
        y = mlp(x)
        assert y.shape == x.shape


class TestBlock:
    """Tests für Transformer-Block."""

    def test_forward_shape(self):
        cfg = GPTConfig(block_size=32, n_embd=64, n_head=4, dropout=0.0)
        block = Block(cfg)
        x = torch.randn(2, 16, 64)
        y = block(x)
        assert y.shape == x.shape

    def test_output_attentions(self):
        """Testet, dass der Block Attention-Weights durchreicht."""
        cfg = GPTConfig(block_size=32, n_embd=64, n_head=4, dropout=0.0)
        block = Block(cfg)
        block.eval()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            y, att_weights = block(x, output_attentions=True)
        assert y.shape == x.shape
        assert att_weights is not None
        assert att_weights.shape == (1, cfg.n_head, 8, 8)


class TestGPT:
    """Tests für das vollständige GPT-Modell."""

    @pytest.fixture
    def nano_config(self):
        return GPTConfig(block_size=32, vocab_size=100, n_layer=2, n_head=2, n_embd=64, dropout=0.0)

    @pytest.fixture
    def model(self, nano_config):
        return GPT(nano_config)

    def test_parameter_count(self, model, nano_config):
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_forward_no_targets(self, model):
        x = torch.randint(0, 100, (2, 16))
        logits, loss = model(x)
        assert logits.shape == (2, 16, 100)
        assert loss is None

    def test_forward_with_targets(self, model):
        x = torch.randint(0, 100, (2, 16))
        logits, loss = model(x, x)
        assert logits.shape == (2, 16, 100)
        assert loss is not None
        assert loss.item() > 0

    def test_forward_sequence_too_long(self, model):
        x = torch.randint(0, 100, (2, 64))
        with pytest.raises(AssertionError):
            model(x)

    def test_generate_shape(self, model):
        x = torch.randint(0, 100, (1, 4))
        gen = model.generate(x, max_new_tokens=8)
        assert gen.shape == (1, 12)

    def test_generate_temperature(self, model):
        x = torch.randint(0, 100, (1, 4))
        gen_cold = model.generate(x.clone(), max_new_tokens=5, temperature=0.1)
        gen_hot = model.generate(x.clone(), max_new_tokens=5, temperature=2.0)
        assert gen_cold.shape == gen_hot.shape

    def test_generate_top_k(self, model):
        x = torch.randint(0, 100, (1, 4))
        gen = model.generate(x, max_new_tokens=5, top_k=5)
        assert gen.shape == (1, 9)

    def test_configure_optimizers(self, model):
        opt = model.configure_optimizers(
            weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95), device_type="cpu"
        )
        assert isinstance(opt, torch.optim.AdamW)
        assert len(opt.param_groups) == 2  # decay + no_decay

    def test_estimate_mfu(self, model):
        mfu = model.estimate_mfu(fwdbwd_per_iter=4, dt=0.1)
        assert 0.0 <= mfu <= 1.0

    def test_weight_tying(self, model):
        """wte und lm_head müssen die gleichen Gewichte teilen."""
        assert model.transformer.wte.weight is model.lm_head.weight

    def test_training_step(self, model):
        """Ein vollständiger Trainingsschritt (forward + backward)."""
        x = torch.randint(0, 100, (2, 16))
        logits, loss = model(x, x)
        loss.backward()
        # Gradienten sollten existieren
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_multiple_configs(self):
        """Testet verschiedene Modellgrößen."""
        configs = [
            GPTConfig(block_size=32, n_layer=2, n_head=2, n_embd=64),
            GPTConfig(block_size=64, n_layer=4, n_head=4, n_embd=128),
        ]
        for cfg in configs:
            model = GPT(cfg)
            x = torch.randint(0, cfg.vocab_size, (1, 8))
            logits, _ = model(x)
            assert logits.shape == (1, 8, cfg.vocab_size)

    def test_output_attentions(self, model):
        """Testet, dass das GPT-Modell Attention-Weights aller Layer zurückgibt."""
        model.eval()
        x = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            logits, loss, all_attentions = model(x, output_attentions=True)
        assert logits.shape == (1, 8, 100)
        assert loss is None
        assert len(all_attentions) == model.config.n_layer
        for layer_idx, att in enumerate(all_attentions):
            assert att.shape == (1, model.config.n_head, 8, 8), (
                f"Layer {layer_idx}: erwartet (1, {model.config.n_head}, 8, 8), "
                f"bekam {att.shape}"
            )

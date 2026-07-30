"""
GPT-2 Modellarchitektur in reinem PyTorch.
Enthält: LayerNorm, CausalSelfAttention, MLP, Block, GPT.
Basierend auf Karpathys nanoGPT / minGPT.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm mit optionalem Bias (wie in GPT-2)."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention mit Flash-Attention Support."""

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd muss durch n_head teilbar sein"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Q, K, V Projektionen in einer Matrix
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output-Projektion
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal Mask (wird einmalig als Buffer registriert)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(
        self, x: torch.Tensor, output_attentions: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.size()  # Batch, Sequenzlänge, Embedding-Dimension

        # Q, K, V berechnen
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape für Multi-Head: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention-Weights für Visualisierung
        att_weights = None

        # Flash Attention wenn verfügbar, sonst manuelle Implementierung
        if hasattr(F, "scaled_dot_product_attention") and not output_attentions:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manuelle Attention (immer wenn output_attentions=True)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att_weights = att.detach()  # (B, n_head, T, T)
            att = self.attn_dropout(att)
            y = att @ v

        # Zurück zu (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output-Projektion
        y = self.resid_dropout(self.c_proj(y))

        if output_attentions:
            return y, att_weights
        return y


class MLP(nn.Module):
    """Feed-Forward Netzwerk mit GELU-Aktivierung (GPT-2 Style)."""

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")  # GPT-2 verwendet tanh-Approximation
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Ein Transformer-Block: Attention + MLP mit Pre-LayerNorm und Residual Connections."""

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self, x: torch.Tensor, output_attentions: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        att_out = self.attn(self.ln_1(x), output_attentions=output_attentions)
        if output_attentions:
            attn_y, att_weights = att_out
            x = x + attn_y
        else:
            x = x + att_out
            att_weights = None
        x = x + self.mlp(self.ln_2(x))
        if output_attentions:
            return x, att_weights
        return x


@dataclass
class GPTConfig:
    """Konfiguration für das GPT-Modell."""
    block_size: int = 1024       # Maximale Kontextlänge
    vocab_size: int = 50304      # Vokabulargröße (GPT-2: 50257, auf Vielfaches von 64 gepadded)
    n_layer: int = 12            # Anzahl Transformer-Blöcke
    n_head: int = 12             # Anzahl Attention-Heads
    n_embd: int = 768            # Embedding-Dimension
    dropout: float = 0.0         # Dropout-Rate
    bias: bool = True            # Bias in Linearen Layern (True = GPT-2 Style)


class GPT(nn.Module):
    """Das vollständige GPT-Sprachmodell."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(  # noqa: C408
            wte=nn.Embedding(config.vocab_size, config.n_embd),       # Token Embedding
            wpe=nn.Embedding(config.block_size, config.n_embd),       # Position Embedding
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying: wte und lm_head teilen sich die Gewichte
        self.transformer.wte.weight = self.lm_head.weight

        # Gewichte initialisieren
        self.apply(self._init_weights)

        # Scaled-Init für Residual-Projektionen (GPT-2 Paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # Anzahl Parameter zählen
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Modell-Parameter: {n_params / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        """
        Args:
            idx: Token-Indizes, Shape (B, T)
            targets: Ziel-Token für Loss-Berechnung, Shape (B, T) oder None
            output_attentions: Wenn True, werden Attention-Weights aller Layer zurückgegeben
        Returns:
            (logits, loss) oder (logits, loss, all_attentions)
        """
        device = idx.device
        _B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Sequenzlänge {T} überschreitet block_size {self.config.block_size}"
        )

        # Position-Indizes
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # Token + Position Embeddings
        tok_emb = self.transformer.wte(idx)      # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)       # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer-Blöcke
        all_attentions = [] if output_attentions else None
        for block in self.transformer.h:
            if output_attentions:
                x, att_weights = block(x, output_attentions=True)
                all_attentions.append(att_weights)
            else:
                x = block(x)

        # Finale LayerNorm
        x = self.transformer.ln_f(x)

        # Logits über lm_head
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Loss berechnen wenn targets gegeben
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        if output_attentions:
            return logits, loss, all_attentions
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Generiert neue Tokens autoregressiv.

        Args:
            idx: Start-Tokens, Shape (B, T)
            max_new_tokens: Anzahl zu generierender Tokens
            temperature: Sampling-Temperatur (1.0 = kein Effekt, <1.0 = deterministischer)
            top_k: Top-k Sampling (None = kein Top-k)
        Returns:
            Generierte Sequenz inkl. Start-Tokens, Shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Kontext auf block_size kürzen
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            # Forward-Pass
            logits, _ = self(idx_cond)

            # Nur das letzte Token betrachten
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Temperatur anwenden
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k Sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Softmax und Sampling
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Anhängen
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """
        Erstellt einen AdamW-Optimizer mit separater Weight-Decay-Behandlung.
        Weight Decay wird nur auf 2D-Parameter (Matrizen) angewendet,
        nicht auf 1D-Parameter (Biases, LayerNorms).
        """
        # Parameter in zwei Gruppen aufteilen
        decay_params = []
        no_decay_params = []
        for pn, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # Weight Decay nur auf Matrizen (ndim >= 2)
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Fused AdamW wenn verfügbar (CUDA)
        fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        Schätzt die Model Flops Utilization (MFU) in Prozent.
        Basierend auf PaLM-Paper: MFU = tatsächliche FLOPs / theoretische Peak-FLOPs.
        """
        N = sum(p.numel() for p in self.parameters())
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)  # pro Sekunde
        # A100 FP16 Peak: 312 TFLOPS
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

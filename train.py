"""
Training-Loop für nanoGPT.
Unterstützt: DDP (Distributed Data Parallel), Gradient Accumulation,
Learning Rate Schedule (Cosine Decay mit Warmup), Mixed Precision,
Logging, Checkpointing und Evaluation.

Verwendung:
    python train.py                          # Standard-Training (GPT-2 small)
    python train.py --config=gpt2-baby       # Baby-Modell für Experimente
    python train.py --device=cpu             # Auf CPU trainieren
"""

import math
import os
import time
from contextlib import nullcontext

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from config import get_train_config
from model import GPT, GPTConfig

# ── W&B (optional) ──────────────────────────────────────────
try:
    from wandb_utils import WandBTracker
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Daten-Loading
# ─────────────────────────────────────────────────────────────

def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zieht einen zufälligen Batch aus den Trainingsdaten."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def load_data(data_dir: str, dataset: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Lädt Trainings- und Validierungsdaten.
    Erwartet train.bin und val.bin im data_dir/<dataset>/ Verzeichnis.
    Falls nicht vorhanden, wird ein einfacher Shakespeare-Datensatz generiert.
    """
    import numpy as np

    data_path = os.path.join(data_dir, dataset)
    train_path = os.path.join(data_path, "train.bin")
    val_path = os.path.join(data_path, "val.bin")

    if os.path.exists(train_path) and os.path.exists(val_path):
        train_data = torch.from_numpy(
            np.memmap(train_path, dtype=np.uint16, mode="r")
        ).long()
        val_data = torch.from_numpy(
            np.memmap(val_path, dtype=np.uint16, mode="r")
        ).long()
        return train_data, val_data

    # Fallback: Einfachen Shakespeare-Datensatz generieren
    print(f"Keine Daten unter {data_path} gefunden. Generiere Fallback-Datensatz...")
    return _generate_fallback_data()


def _generate_fallback_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Generiert einen einfachen Zeichen-basierten Datensatz aus einem eingebauten Text."""

    text = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:!?-'\n"
        * 1000
    )
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}

    print(f"Fallback-Datensatz: {len(chars)} eindeutige Zeichen, {len(text)} Zeichen insgesamt")

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


# ─────────────────────────────────────────────────────────────
# Learning Rate Schedule
# ─────────────────────────────────────────────────────────────

def get_lr(
    it: int,
    learning_rate: float,
    min_lr: float,
    warmup_iters: int,
    lr_decay_iters: int,
) -> float:
    """Cosine Learning Rate Schedule mit linearem Warmup."""
    # Lineares Warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)

    # Nach Decay-Iterationen: min_lr
    if it > lr_decay_iters:
        return min_lr

    # Cosine Decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    eval_iters: int,
    block_size: int,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    """Schätzt den Loss auf Trainings- und Validierungsdaten."""
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ─────────────────────────────────────────────────────────────
# Haupt-Training
# ─────────────────────────────────────────────────────────────

def train(train_cfg: dict | None = None):
    """
    Führt das Training durch.

    Args:
        train_cfg: Vollständige Trainings-Konfiguration (dict).
                   Wenn None, wird gpt2-small als Standard verwendet.
    """

    # ── Konfiguration ────────────────────────────────────────
    if train_cfg is None:
        train_cfg = get_train_config("gpt2-small")

    # Entpacken
    dataset = train_cfg["dataset"]
    gradient_accumulation_steps = train_cfg["gradient_accumulation_steps"]
    batch_size = train_cfg["batch_size"]
    block_size = train_cfg["block_size"]

    n_layer = train_cfg["n_layer"]
    n_head = train_cfg["n_head"]
    n_embd = train_cfg["n_embd"]
    dropout = train_cfg["dropout"]
    bias = train_cfg["bias"]

    learning_rate = train_cfg["learning_rate"]
    max_iters = train_cfg["max_iters"]
    weight_decay = train_cfg["weight_decay"]
    beta1 = train_cfg["beta1"]
    beta2 = train_cfg["beta2"]
    grad_clip = train_cfg["grad_clip"]

    decay_lr = train_cfg["decay_lr"]
    warmup_iters = train_cfg["warmup_iters"]
    lr_decay_iters = train_cfg["lr_decay_iters"]
    min_lr = train_cfg["min_lr"]

    eval_interval = train_cfg["eval_interval"]
    eval_iters = train_cfg["eval_iters"]
    log_interval = train_cfg["log_interval"]

    device = train_cfg["device"]
    compile_model = train_cfg["compile"]
    dtype_str = train_cfg["dtype"]

    # ── Distributed Setup ────────────────────────────────────
    ddp = int(os.environ.get("RANK", "-1")) != -1
    ddp_local_rank = 0
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    # Device-Typ für Mixed Precision
    device_type = "cuda" if "cuda" in device else "cpu"

    # ── Mixed Precision ──────────────────────────────────────
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    ptdtype = dtype_map.get(dtype_str, torch.float32)
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    # ── Reproduzierbarkeit ───────────────────────────────────
    torch.manual_seed(1337 + seed_offset)
    if device_type == "cuda":
        torch.cuda.manual_seed(1337 + seed_offset)

    # ── Daten ────────────────────────────────────────────────
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    train_data, val_data = load_data(data_dir, dataset)

    # ── Modell ───────────────────────────────────────────────
    model_config = GPTConfig(
        block_size=block_size,
        vocab_size=50304,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
    )
    model = GPT(model_config)

    if device_type == "cuda":
        model = model.to(device)

    # torch.compile (PyTorch 2.0+)
    if compile_model and hasattr(torch, "compile"):
        if master_process:
            print("Kompiliere Modell mit torch.compile()...")
        model = torch.compile(model)

    # DDP Wrapper
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Roh-Modell für Checkpointing (ohne DDP-Wrapper)
    raw_model = model.module if ddp else model

    # ── Optimizer ─────────────────────────────────────────────
    optimizer = raw_model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        betas=(beta1, beta2),
        device_type=device_type,
    )

    # ── Checkpointing ────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out_dir, exist_ok=True)

    iter_num = 0
    best_val_loss = float("inf")

    # ── W&B Tracking ─────────────────────────────────────────
    tracker = None
    if master_process and WANDB_AVAILABLE:
        tracker = WandBTracker(
            project="nanoGPT",
            config={
                "model": f"GPT-{n_layer}L-{n_head}H-{n_embd}E",
                "n_layer": n_layer,
                "n_head": n_head,
                "n_embd": n_embd,
                "block_size": block_size,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "max_iters": max_iters,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "device": device,
                "dtype": dtype_str,
            },
            tags=["gpt", "transformer", "language-model", dataset],
            group=dataset,
            job_type="train",
            notes=f"nanoGPT Training: {n_layer}L-{n_head}H-{n_embd}E, {max_iters} iters",
        )

    # ── Training Loop ────────────────────────────────────────
    X, Y = get_batch(train_data, block_size, batch_size, device)
    t0 = time.time()
    local_iter_num = 0
    raw_model.train()

    if master_process:
        print(f"\n{'='*60}")
        print(f"Training startet: {max_iters:,} Iterationen")
        print(f"Modell: {sum(p.numel() for p in model.parameters())/1e6:.2f}M Parameter")
        print(f"Device: {device}, Dtype: {dtype_str}, DDP: {ddp}")
        print(f"{'='*60}\n")

    while True:
        # Learning Rate setzen
        lr = get_lr(iter_num, learning_rate, min_lr, warmup_iters, lr_decay_iters) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluation und Checkpointing
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss(
                model, train_data, val_data, eval_iters, block_size, batch_size, device
            )
            print(
                f"Step {iter_num:>6d}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

            # ── W&B Eval Logging ─────────────────────────────
            if tracker and tracker.is_active:
                tracker.log_eval(iter_num, losses['train'], losses['val'])

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_config": model_config,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "train_config": train_cfg,
                    }
                    checkpoint_path = os.path.join(out_dir, "ckpt.pt")
                    torch.save(checkpoint, checkpoint_path)
                    print(f"  → Checkpoint gespeichert unter {checkpoint_path}")

                    # ── W&B Checkpoint Logging ────────────────
                    if tracker and tracker.is_active:
                        tracker.log_checkpoint(iter_num, best_val_loss, is_best=True)

        # Abbruchbedingung
        if iter_num > max_iters:
            break

        # Forward + Backward mit Gradient Accumulation
        loss = torch.tensor(0.0, device=device)
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )

            with ctx:
                _logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps

            # Nächstes Batch sofort laden (Prefetch)
            X, Y = get_batch(train_data, block_size, batch_size, device)

            loss.backward()

        # Gradient Clipping
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optimizer Step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Timing und Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            print(
                f"Iter {iter_num:>6d}: loss {lossf:.4f}, "
                f"time {dt*1000:.0f}ms, "
                f"mfu {mfu*100:.1f}%, "
                f"lr {lr:.2e}"
            )

            # ── W&B Step Logging ──────────────────────────────
            if tracker and tracker.is_active:
                tracker.log_step(
                    iter_num=iter_num,
                    train_loss=lossf,
                    lr=lr,
                    mfu=mfu,
                    dt_ms=dt * 1000,
                )

        iter_num += 1
        local_iter_num += 1

    # ── Finales Speichern ────────────────────────────────────
    if master_process:
        final_checkpoint = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": model_config,
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            "train_config": train_cfg,
        }
        final_path = os.path.join(out_dir, "ckpt.pt")
        torch.save(final_checkpoint, final_path)
        print(f"\nTraining abgeschlossen! Finaler Checkpoint: {final_path}")

        # ── W&B Cleanup ──────────────────────────────────────
        if tracker and tracker.is_active:
            tracker.log_checkpoint(iter_num, best_val_loss, is_best=False)
            tracker.finish()

    if ddp:
        destroy_process_group()


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────


def main():
    """CLI-Entry-Point für das Training (auch via pyproject.toml scripts)."""
    import argparse

    parser = argparse.ArgumentParser(description="nanoGPT Training")
    parser.add_argument(
        "--config",
        type=str,
        default="gpt2-small",
        help="Trainings-Konfiguration (gpt2-small, gpt2-baby)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, cpu, mps)")
    parser.add_argument("--max_iters", type=int, default=None, help="Maximale Iterationen")
    parser.add_argument("--eval_interval", type=int, default=None, help="Eval-Intervall")
    parser.add_argument("--compile", type=str, default=None, help="torch.compile (true/false)")

    args = parser.parse_args()

    overrides = {}
    if args.device:
        overrides["device"] = args.device
    if args.max_iters:
        overrides["max_iters"] = args.max_iters
    if args.eval_interval:
        overrides["eval_interval"] = args.eval_interval
    if args.compile is not None:
        overrides["compile"] = args.compile.lower() in ("true", "1", "yes")

    # Trainings-Konfiguration laden und Overrides anwenden
    base_cfg = get_train_config(args.config)
    base_cfg.update(overrides)

    train(base_cfg)


if __name__ == "__main__":
    main()

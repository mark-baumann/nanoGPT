"""
Attention-Visualisierung für nanoGPT.
Zeigt, wie die Attention-Heads auf verschiedene Token-Positionen achten.
Erzeugt Heatmaps für jeden Head und eine Zusammenfassung über alle Layer.
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless-Backend
import matplotlib.pyplot as plt
import numpy as np
import torch

# nanoGPT-Modul importieren
sys.path.insert(0, str(Path(__file__).parent))
from model import GPT, GPTConfig


def create_model(config_name: str = "gpt2-nano") -> GPT:
    """Erstellt ein nanoGPT-Modell mit der angegebenen Konfiguration."""
    configs = {
        "gpt2-nano": GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=256, vocab_size=50304),
        "gpt2-micro": GPTConfig(n_layer=4, n_head=4, n_embd=128, block_size=256, vocab_size=50304),
        "gpt2-baby": GPTConfig(n_layer=6, n_head=6, n_embd=384, block_size=256, vocab_size=50304),
    }
    config = configs.get(config_name, configs["gpt2-nano"])
    model = GPT(config)
    model.eval()
    return model


def visualize_attention(
    model: GPT,
    input_ids: torch.Tensor,
    tokens: list[str] | None = None,
    output_dir: str = "attention_plots",
    max_tokens: int = 30,
):
    """Erzeugt Attention-Heatmaps für alle Layer und Heads."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Auf max_tokens kürzen
    input_ids = input_ids[:, :max_tokens]
    seq_len = input_ids.size(1)

    # Token-Labels
    if tokens is None:
        tokens = [f"T{i}" for i in range(seq_len)]
    else:
        tokens = tokens[:seq_len]

    # Forward-Pass mit Attention-Weights
    with torch.no_grad():
        _, _, all_attentions = model(input_ids, output_attentions=True)

    n_layers = len(all_attentions)
    n_heads = all_attentions[0].size(1)  # (B, n_head, T, T)

    # --- 1. Heatmap pro Layer (gemittelt über alle Heads) ---
    fig, axes = plt.subplots(
        1, n_layers, figsize=(5 * n_layers, 5), squeeze=False
    )
    fig.suptitle("Attention pro Layer (gemittelt über alle Heads)", fontsize=14, y=1.02)

    for layer_idx, att in enumerate(all_attentions):
        # att: (1, n_head, T, T) -> (T, T) gemittelt
        avg_att = att[0].mean(dim=0).cpu().numpy()

        ax = axes[0, layer_idx]
        im = ax.imshow(avg_att, cmap="Blues", aspect="auto", vmin=0, vmax=avg_att.max())
        ax.set_title(f"Layer {layer_idx + 1}")
        ax.set_xlabel("Key-Position")
        ax.set_ylabel("Query-Position")

        # Token-Labels (nur wenn nicht zu viele)
        if seq_len <= 20:
            ax.set_xticks(range(seq_len))
            ax.set_yticks(range(seq_len))
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(tokens, fontsize=7)

        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(output_path / "attention_per_layer.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Gespeichert: {output_path / 'attention_per_layer.png'}")

    # --- 2. Heatmap pro Head (nur erster Layer, wenn viele Heads) ---
    first_layer_att = all_attentions[0][0]  # (n_head, T, T)
    n_heads_plot = min(n_heads, 8)

    cols = min(n_heads_plot, 4)
    rows = math.ceil(n_heads_plot / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    fig.suptitle(f"Attention pro Head — Layer 1", fontsize=14, y=1.02)

    for head_idx in range(n_heads_plot):
        ax = axes[head_idx // cols, head_idx % cols]
        head_att = first_layer_att[head_idx].cpu().numpy()
        im = ax.imshow(head_att, cmap="Reds", aspect="auto", vmin=0, vmax=head_att.max())
        ax.set_title(f"Head {head_idx + 1}")
        ax.set_xlabel("Key-Position")
        ax.set_ylabel("Query-Position")

        if seq_len <= 20:
            ax.set_xticks(range(seq_len))
            ax.set_yticks(range(seq_len))
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=6)
            ax.set_yticklabels(tokens, fontsize=6)

        plt.colorbar(im, ax=ax, fraction=0.046)

    # Leere Subplots ausblenden
    for idx in range(n_heads_plot, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path / "attention_per_head_layer1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Gespeichert: {output_path / 'attention_per_head_layer1.png'}")

    # --- 3. Kausale Maske visualisieren ---
    causal_mask = np.tril(np.ones((seq_len, seq_len)))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(causal_mask, cmap="gray_r", aspect="auto")
    ax.set_title("Kausale Attention-Maske (Causal Mask)", fontsize=12)
    ax.set_xlabel("Key-Position")
    ax.set_ylabel("Query-Position")
    if seq_len <= 20:
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
    plt.colorbar(im, ax=ax, label="Erlaubt (1) / Maskiert (0)")
    plt.tight_layout()
    plt.savefig(output_path / "causal_mask.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Gespeichert: {output_path / 'causal_mask.png'}")

    # --- 4. Attention-Flow über Layer (wie ändert sich die Aufmerksamkeit?) ---
    fig, axes = plt.subplots(
        1, n_layers, figsize=(5 * n_layers, 5), squeeze=False
    )
    fig.suptitle("Attention-Flow: Wie ändert sich die Aufmerksamkeit über Layer?", fontsize=14, y=1.02)

    for layer_idx, att in enumerate(all_attentions):
        avg_att = att[0].mean(dim=0).cpu().numpy()

        # Zeilenweise normalisieren für bessere Vergleichbarkeit
        row_max = avg_att.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        avg_att_norm = avg_att / row_max

        ax = axes[0, layer_idx]
        im = ax.imshow(avg_att_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"Layer {layer_idx + 1} (zeilennormiert)")
        ax.set_xlabel("Key-Position")
        ax.set_ylabel("Query-Position")

        if seq_len <= 20:
            ax.set_xticks(range(seq_len))
            ax.set_yticks(range(seq_len))
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(tokens, fontsize=7)

        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(output_path / "attention_flow.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Gespeichert: {output_path / 'attention_flow.png'}")

    # --- 5. Zusammenfassung als Text ---
    summary_path = output_path / "attention_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== Attention-Visualisierung Zusammenfassung ===\n\n")
        f.write(f"Modell: {model.config.n_layer} Layer, {n_heads} Heads, "
                f"Embedding={model.config.n_embd}\n")
        f.write(f"Sequenzlänge: {seq_len} Tokens\n\n")

        for layer_idx, att in enumerate(all_attentions):
            avg_att = att[0].mean(dim=0).cpu().numpy()
            # Durchschnittliche Attention auf sich selbst (Diagonale)
            diag_mean = np.diag(avg_att).mean()
            # Durchschnittliche Attention auf vorherige Tokens
            off_diag_mask = np.tril(np.ones((seq_len, seq_len)), k=-1)
            off_diag_sum = (avg_att * off_diag_mask).sum()
            off_diag_count = off_diag_mask.sum()
            off_diag_mean = off_diag_sum / off_diag_count if off_diag_count > 0 else 0

            f.write(f"Layer {layer_idx + 1}:\n")
            f.write(f"  Selbst-Attention (Diagonale): {diag_mean:.4f}\n")
            f.write(f"  Kontext-Attention (vorherige Tokens): {off_diag_mean:.4f}\n")
            f.write(f"  Verhältnis Selbst/Kontext: {diag_mean / max(off_diag_mean, 1e-8):.2f}x\n\n")

    print(f"✅ Gespeichert: {summary_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Attention-Visualisierung für nanoGPT")
    parser.add_argument(
        "--config", default="gpt2-nano",
        choices=["gpt2-nano", "gpt2-micro", "gpt2-baby"],
        help="Modell-Konfiguration"
    )
    parser.add_argument(
        "--text", default="Die Transformer-Architektur revolutionierte das maschinelle Lernen",
        help="Eingabetext für die Visualisierung"
    )
    parser.add_argument(
        "--output", default="attention_plots",
        help="Ausgabeverzeichnis für die Plots"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=20,
        help="Maximale Anzahl Tokens"
    )
    args = parser.parse_args()

    print(f"🚀 Erstelle Modell: {args.config}")
    model = create_model(args.config)

    # Tokenisierung (einfach: Buchstaben als Tokens für Demo)
    text = args.text
    # Verwende ASCII-Werte als Pseudo-Token-IDs
    token_ids = [ord(c) % 1000 + 100 for c in text]  # Einfache Pseudo-Tokenisierung
    input_ids = torch.tensor([token_ids], dtype=torch.long)

    tokens = list(text)
    print(f"📝 Text: '{text}' ({len(tokens)} Tokens)")

    print(f"🎨 Erstelle Visualisierungen...")
    output_dir = visualize_attention(
        model, input_ids, tokens=tokens,
        output_dir=args.output, max_tokens=args.max_tokens
    )

    print(f"\n✨ Fertig! Alle Plots in: {output_dir}/")
    print(f"   - attention_per_layer.png")
    print(f"   - attention_per_head_layer1.png")
    print(f"   - causal_mask.png")
    print(f"   - attention_flow.png")
    print(f"   - attention_summary.txt")


if __name__ == "__main__":
    main()

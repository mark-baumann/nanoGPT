"""
Text-Generierung aus einem trainierten nanoGPT-Modell.
Lädt einen Checkpoint und generiert Text autoregressiv.

Verwendung:
    python sample.py --checkpoint=out/ckpt.pt --prompt="Es war einmal"
    python sample.py --checkpoint=out/ckpt.pt --num_samples=5 --max_new_tokens=200
    python sample.py --checkpoint=out/ckpt.pt --temperature=0.8 --top_k=40
"""

import argparse
import os

import torch

from model import GPT, GPTConfig


def load_model(checkpoint_path: str, device: str = "cpu") -> GPT:
    """
    Lädt ein trainiertes Modell aus einem Checkpoint.

    Args:
        checkpoint_path: Pfad zur Checkpoint-Datei (.pt)
        device: Device für das Modell ("cpu", "cuda", "mps")
    Returns:
        Geladenes GPT-Modell im eval-Modus
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Modell-Konfiguration aus Checkpoint extrahieren
    model_config = checkpoint.get("model_config")
    if model_config is None:
        # Fallback: GPT-2 small
        print("Warnung: Keine model_config im Checkpoint, verwende GPT-2 small Default.")
        model_config = GPTConfig()

    # Modell erstellen und Gewichte laden
    model = GPT(model_config)
    state_dict = checkpoint.get("model", checkpoint)  # "model" key oder direkt
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    # Trainings-Info anzeigen
    iter_num = checkpoint.get("iter_num", "?")
    best_val_loss = checkpoint.get("best_val_loss", "?")
    print(f"Modell geladen: {checkpoint_path}")
    print(f"  Iteration: {iter_num}, Best Val Loss: {best_val_loss}")
    print(f"  Parameter: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model


def encode_prompt(
    prompt: str, device: str = "cpu"
) -> torch.Tensor:
    """
    Kodiert einen Text-Prompt in Token-Indizes.
    Verwendet eine einfache Zeichen-basierte Kodierung (wie im Fallback-Datensatz).

    Args:
        prompt: Text-Prompt
        device: Ziel-Device
    Returns:
        Token-Tensor, Shape (1, T)
    """
    # Einfache Zeichen-Kodierung (wie im Fallback-Datensatz von train.py)
    chars = sorted(set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:!?-'\\n"
    ))
    stoi = {ch: i for i, ch in enumerate(chars)}

    tokens = [stoi.get(c, 0) for c in prompt]  # Unbekannte Zeichen → 0
    return torch.tensor([tokens], dtype=torch.long, device=device)


def decode_tokens(tokens: torch.Tensor) -> str:
    """
    Dekodiert Token-Indizes zurück in Text.
    Verwendet die gleiche Zeichen-basierte Kodierung.

    Args:
        tokens: Token-Tensor, Shape (T,) oder (1, T)
    Returns:
        Dekodierter Text
    """
    chars = sorted(set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:!?-'\\n"
    ))
    itos = {i: ch for i, ch in enumerate(chars)}

    if tokens.dim() == 2:
        tokens = tokens[0]
    return "".join(itos.get(t.item(), "?") for t in tokens)


def generate_samples(
    model: GPT,
    prompt: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: str,
) -> list[str]:
    """
    Generiert mehrere Text-Samples aus einem Prompt.

    Args:
        model: Das geladene GPT-Modell
        prompt: Start-Text
        num_samples: Anzahl zu generierender Samples
        max_new_tokens: Maximale Anzahl neuer Tokens pro Sample
        temperature: Sampling-Temperatur
        top_k: Top-k Sampling (None = deaktiviert)
        device: Device
    Returns:
        Liste generierter Texte
    """
    # Prompt kodieren
    start_ids = encode_prompt(prompt, device)

    samples = []
    for i in range(num_samples):
        with torch.no_grad():
            generated = model.generate(
                start_ids.clone(),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        text = decode_tokens(generated)
        samples.append(text)

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Text-Generierung mit nanoGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python sample.py --checkpoint=out/ckpt.pt
  python sample.py --checkpoint=out/ckpt.pt --prompt="Es war einmal" --temperature=0.8
  python sample.py --checkpoint=out/ckpt.pt --num_samples=3 --max_new_tokens=500
  python sample.py --checkpoint=out/ckpt.pt --top_k=40 --temperature=0.7
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="out/ckpt.pt",
        help="Pfad zum Modell-Checkpoint (default: out/ckpt.pt)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="\\n",
        help="Start-Prompt für die Generierung (default: Zeilenumbruch)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Anzahl zu generierender Samples (default: 1)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximale Anzahl neuer Tokens (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling-Temperatur: <1.0 = deterministischer, >1.0 = kreativer (default: 1.0)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k Sampling: Nur die k wahrscheinlichsten Tokens behalten (default: None = deaktiviert)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: cpu, cuda, mps (default: cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random Seed für Reproduzierbarkeit",
    )

    args = parser.parse_args()

    # Seed setzen
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Random Seed: {args.seed}")

    # Device prüfen
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA nicht verfügbar, verwende CPU.")
        args.device = "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS nicht verfügbar, verwende CPU.")
        args.device = "cpu"

    # Modell laden
    model = load_model(args.checkpoint, args.device)

    # Samples generieren
    print(f"\n{'='*60}")
    print(f"Generiere {args.num_samples} Sample(s)...")
    print(f"Prompt: {args.prompt!r}")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}")
    print(f"Max neue Tokens: {args.max_new_tokens}")
    print(f"{'='*60}\n")

    samples = generate_samples(
        model=model,
        prompt=args.prompt,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )

    for i, sample in enumerate(samples):
        print(f"─── Sample {i+1} ───")
        print(sample)
        print()


if __name__ == "__main__":
    main()

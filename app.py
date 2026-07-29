"""
Streamlit-App: nanoGPT — Modell-Architektur & Textgenerierung
=============================================================
Modell-Architektur visualisieren, Konfigurationen vergleichen, Text generieren.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import GPTConfig, GPT
from config import (
    gpt2_small, gpt2_medium, gpt2_large, gpt2_xl,
    gpt2_baby, gpt2_micro, gpt2_nano,
    get_config, get_train_config,
)

st.set_page_config(
    page_title="nanoGPT — Architektur & Generierung",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 nanoGPT — Modell-Architektur & Textgenerierung")
st.markdown("### GPT-2 Architektur visualisieren · Konfigurationen vergleichen · Text generieren")

# ── Sidebar ──
seite = st.sidebar.radio(
    "📂 Bereich wählen",
    ["🏗️ Architektur", "⚙️ Konfigurationen", "✍️ Text generieren"],
)

# ═══════════════════════════════════════════════════════════════
# ARCHITEKTUR
# ═══════════════════════════════════════════════════════════════
if seite == "🏗️ Architektur":
    st.header("🏗️ GPT-2 Modell-Architektur")

    col1, col2 = st.columns([1, 2])

    with col1:
        config_name = st.selectbox(
            "Modell-Konfiguration",
            ["gpt2-nano", "gpt2-micro", "gpt2-baby", "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"],
            index=3,
        )
        config = get_config(config_name)

        st.markdown("### 📋 Parameter")
        st.markdown(f"""
        | Parameter | Wert |
        |-----------|------|
        | **Block-Größe** | {config.block_size} |
        | **Vokabular** | {config.vocab_size:,} |
        | **Layer** | {config.n_layer} |
        | **Heads** | {config.n_head} |
        | **Embedding-Dim** | {config.n_embd} |
        | **Head-Dim** | {config.n_embd // config.n_head} |
        | **Dropout** | {config.dropout} |
        | **Bias** | {config.bias} |
        """)

        # Parameter schätzen
        try:
            model = GPT(config)
            n_params = sum(p.numel() for p in model.parameters())
            st.metric("Gesamt-Parameter", f"{n_params / 1e6:.1f}M")
        except Exception:
            st.metric("Gesamt-Parameter", "—")

    with col2:
        st.markdown("### 🎨 Architektur-Diagramm")

        # Architektur als Matplotlib-Diagramm
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 14)
        ax.axis('off')

        y = 13
        box_h = 0.8

        def draw_box(x, y, w, h, text, color='#4ECDC4', fontsize=9):
            rect = plt.Rectangle((x - w / 2, y - h / 2), w, h,
                                 facecolor=color, edgecolor='white',
                                 linewidth=2, alpha=0.9, zorder=2)
            ax.add_patch(rect)
            ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                    fontweight='bold', color='white', zorder=3)

        def draw_arrow(x1, y1, x2, y2, color='#888888'):
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

        # Input Embeddings
        draw_box(5, y, 8, box_h, "Input Embeddings\nToken + Position", '#45B7D1')
        y -= 1.2

        # Transformer Blocks
        n_blocks_to_show = min(config.n_layer, 6)
        for i in range(n_blocks_to_show):
            draw_box(5, y, 8, box_h,
                     f"Transformer Block {i + 1}\nMulti-Head Attention + MLP + LayerNorm",
                     '#4ECDC4')
            if i < n_blocks_to_show - 1:
                draw_arrow(5, y - box_h / 2, 5, y - 1.2 + box_h / 2)
            y -= 1.2

        if config.n_layer > 6:
            ax.text(5, y + 0.3, f"... ({config.n_layer - 6} weitere Blöcke)",
                    ha='center', fontsize=9, color='#888888', style='italic')
            y -= 0.6

        # Final LayerNorm
        draw_box(5, y, 8, box_h, "Final LayerNorm", '#FFE66D', fontsize=9)
        y -= 1.2

        # LM Head
        draw_box(5, y, 8, box_h, f"LM Head\nLinear({config.n_embd} → {config.vocab_size})", '#FF6B6B', fontsize=9)

        # Detail: Attention Block
        ax.text(0.5, 13, "Detail: Attention-Block", fontsize=10, fontweight='bold', color='#333')
        detail_y = 12
        draw_box(0.5, detail_y, 1.5, 0.6, "Q", '#95E1D3', 8)
        draw_box(2.2, detail_y, 1.5, 0.6, "K", '#95E1D3', 8)
        draw_box(3.9, detail_y, 1.5, 0.6, "V", '#95E1D3', 8)
        detail_y -= 0.8
        draw_box(2.2, detail_y, 3.5, 0.6, "Q·Kᵀ / √d · Softmax", '#F38181', 8)
        detail_y -= 0.8
        draw_box(2.2, detail_y, 3.5, 0.6, "Attention · V", '#AA96DA', 8)
        detail_y -= 0.8
        draw_box(2.2, detail_y, 3.5, 0.6, "Output-Projektion", '#4ECDC4', 8)

        ax.set_title(f"GPT-2 Architektur — {config_name} ({config.n_layer} Layer, {config.n_head} Heads)",
                     fontsize=14, fontweight='bold', pad=20)
        st.pyplot(fig)
        plt.close(fig)

    # Parameter-Vergleich
    st.markdown("---")
    st.subheader("📊 Parameter-Vergleich aller Konfigurationen")

    configs_data = [
        ("Nano", gpt2_nano),
        ("Micro", gpt2_micro),
        ("Baby", gpt2_baby),
        ("Small", gpt2_small),
        ("Medium", gpt2_medium),
        ("Large", gpt2_large),
        ("XL", gpt2_xl),
    ]

    names = []
    params = []
    layers = []
    embds = []

    for name, cfg in configs_data:
        names.append(name)
        layers.append(cfg.n_layer)
        embds.append(cfg.n_embd)
        try:
            m = GPT(cfg)
            params.append(sum(p.numel() for p in m.parameters()) / 1e6)
        except Exception:
            # Schätzung: ~12 * n_layer * n_embd²
            params.append(12 * cfg.n_layer * cfg.n_embd ** 2 / 1e6)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(names, params, color='#4ECDC4', edgecolor='white')
    ax1.set_ylabel("Parameter (Millionen)")
    ax1.set_title("Modellgröße")
    ax1.set_yscale('log')
    for i, v in enumerate(params):
        ax1.text(i, v * 1.1, f"{v:.0f}M", ha='center', fontsize=9)

    x = np.arange(len(names))
    width = 0.35
    ax2.bar(x - width / 2, layers, width, label='Layer', color='#4ECDC4', edgecolor='white')
    ax2_twin = ax2.twinx()
    ax2_twin.bar(x + width / 2, embds, width, label='Embedding-Dim', color='#FF6B6B', edgecolor='white')
    ax2.set_ylabel("Anzahl Layer")
    ax2_twin.set_ylabel("Embedding-Dimension")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_title("Layer & Embedding-Dimension")
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    st.pyplot(fig)
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════
# KONFIGURATIONEN
# ═══════════════════════════════════════════════════════════════
elif seite == "⚙️ Konfigurationen":
    st.header("⚙️ Trainings-Konfigurationen vergleichen")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("GPT-2 Small Training")
        train_small = get_train_config("gpt2-small")
        st.json({k: v for k, v in train_small.items() if not isinstance(v, (list, dict))})

    with col2:
        st.subheader("GPT-2 Baby Training")
        train_baby = get_train_config("gpt2-baby")
        st.json({k: v for k, v in train_baby.items() if not isinstance(v, (list, dict))})

    # Learning Rate Schedule visualisieren
    st.markdown("---")
    st.subheader("📈 Learning Rate Schedule")

    def cosine_lr_schedule(warmup_iters, lr_decay_iters, learning_rate, min_lr):
        iters = np.arange(lr_decay_iters)
        lr = np.zeros(lr_decay_iters)

        # Warmup
        warmup = np.minimum(iters / max(1, warmup_iters), 1.0)

        # Cosine decay
        decay_ratio = (iters - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
        decay_ratio = np.clip(decay_ratio, 0, 1)
        cosine = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))

        lr = min_lr + (learning_rate - min_lr) * warmup * cosine
        return iters, lr

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # GPT-2 Small
    iters_s, lr_s = cosine_lr_schedule(
        train_small["warmup_iters"],
        train_small["lr_decay_iters"],
        train_small["learning_rate"],
        train_small["min_lr"],
    )
    ax1.plot(iters_s[:10000], lr_s[:10000] * 1000, color='#4ECDC4')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Learning Rate (×10⁻³)")
    ax1.set_title("GPT-2 Small — LR Schedule (erste 10k)")
    ax1.grid(True, alpha=0.3)

    # GPT-2 Baby
    iters_b, lr_b = cosine_lr_schedule(
        train_baby["warmup_iters"],
        train_baby["lr_decay_iters"],
        train_baby["learning_rate"],
        train_baby["min_lr"],
    )
    ax2.plot(iters_b, lr_b * 1000, color='#FF6B6B')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Learning Rate (×10⁻³)")
    ax2.set_title("GPT-2 Baby — LR Schedule")
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    # Konfigurations-Diff
    st.markdown("---")
    st.subheader("🔍 Konfigurations-Unterschiede")

    diff_data = []
    all_keys = set(train_small.keys()) | set(train_baby.keys())
    for key in sorted(all_keys):
        v1 = train_small.get(key, "—")
        v2 = train_baby.get(key, "—")
        if v1 != v2:
            diff_data.append({"Parameter": key, "GPT-2 Small": str(v1), "GPT-2 Baby": str(v2)})

    st.table(diff_data)

# ═══════════════════════════════════════════════════════════════
# TEXT GENERIEREN
# ═══════════════════════════════════════════════════════════════
elif seite == "✍️ Text generieren":
    st.header("✍️ Text generieren mit nanoGPT")

    st.info(
        "⚠️ **Hinweis:** Diese Demo generiert Text mit einem **un trainierten Modell** "
        "(zufällige Gewichte). Die Ausgabe ist daher zufälliges Rauschen. "
        "Für sinnvolle Generierung muss das Modell zuerst trainiert werden."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        gen_config = st.selectbox(
            "Modell für Generierung",
            ["gpt2-nano", "gpt2-micro", "gpt2-baby"],
            index=0,
            help="Kleinere Modelle für schnellere Demo.",
        )

        prompt = st.text_area(
            "Prompt",
            value="Es war einmal",
            height=80,
        )

        max_tokens = st.slider("Max. neue Tokens", 10, 200, 50)
        temperature = st.slider("Temperatur", 0.1, 2.0, 1.0, 0.1)
        top_k = st.slider("Top-k", 0, 100, 0, help="0 = deaktiviert")

        if st.button("✍️ Generieren", type="primary", use_container_width=True):
            st.session_state.generate_clicked = True
        else:
            if "generate_clicked" not in st.session_state:
                st.session_state.generate_clicked = False

    with col2:
        if st.session_state.generate_clicked:
            with st.spinner("Initialisiere Modell und generiere..."):
                try:
                    import torch

                    config = get_config(gen_config)
                    model = GPT(config)
                    model.eval()

                    # Einfaches Token-Mapping (Demo)
                    chars = sorted(set("abcdefghijklmnopqrstuvwxyzäöüßABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ .,!?\"'-"))
                    stoi = {ch: i + 1 for i, ch in enumerate(chars)}  # 0 = padding
                    itos = {i: ch for ch, i in stoi.items()}

                    # Prompt tokenisieren
                    input_ids = [stoi.get(ch, 0) for ch in prompt if ch in stoi]
                    if not input_ids:
                        input_ids = [stoi.get(' ', 0)]

                    idx = torch.tensor([input_ids], dtype=torch.long)

                    # Generieren
                    top_k_val = top_k if top_k > 0 else None
                    output = model.generate(
                        idx,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k_val,
                    )

                    # Detokenisieren
                    output_ids = output[0].tolist()
                    generated = ''.join(itos.get(i, '?') for i in output_ids)

                    st.subheader("📝 Generierter Text")
                    st.code(generated, language="text")

                    st.markdown(f"**Prompt-Länge:** {len(prompt)} Zeichen | "
                                f"**Generierte Tokens:** {max_tokens} | "
                                f"**Temperatur:** {temperature} | "
                                f"**Top-k:** {top_k if top_k > 0 else 'aus'}")

                except Exception as e:
                    st.error(f"Fehler bei der Generierung: {e}")
                    st.info(
                        "Mögliche Ursachen:\n"
                        "- PyTorch nicht installiert\n"
                        "- Speicher zu klein für das gewählte Modell"
                    )
        else:
            st.info("👈 Konfiguriere links die Parameter und klicke auf **Generieren**.")

    # Sampling-Parameter erklären
    st.markdown("---")
    st.subheader("🌡️ Sampling-Parameter erklärt")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Temperatur**")
        st.markdown(
            "- `0.1–0.5`: Deterministisch, wiederholend\n"
            "- `0.7–1.0`: Ausgewogen\n"
            "- `1.2–2.0`: Kreativ, chaotisch"
        )
    with col_b:
        st.markdown("**Top-k**")
        st.markdown(
            "- `0`: Deaktiviert\n"
            "- `10–50`: Fokussiert\n"
            "- `50–100`: Vielfältig"
        )
    with col_c:
        st.markdown("**Max Tokens**")
        st.markdown(
            "- Begrenzt durch `block_size`\n"
            "- Längere Texte = mehr Kontext\n"
            "- Kurze Texte = schneller"
        )

st.sidebar.markdown("---")
st.sidebar.markdown("📁 **Repo:** [nanoGPT](https://github.com/mark-baumann/nanoGPT)")
st.sidebar.markdown("🐍 **Python 3.13** · **Streamlit** · **PyTorch**")

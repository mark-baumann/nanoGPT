"""
nanoGPT Streamlit App - Textgenerierung mit nanoGPT.
"""
import streamlit as st
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig

st.set_page_config(page_title="nanoGPT", page_icon="🤖")
st.title("nanoGPT - Textgenerierung")

checkpoint = st.text_input("Checkpoint-Pfad", "out/ckpt.pt")
prompt = st.text_area("Prompt", "Es war einmal")
temperature = st.slider("Temperature", 0.0, 2.0, 0.8, 0.1)
top_k = st.slider("Top-K", 1, 200, 40)
max_tokens = st.slider("Max Tokens", 10, 500, 100)

if st.button("Generieren"):
    if not os.path.exists(checkpoint):
        st.error(f"Checkpoint nicht gefunden: {checkpoint}")
    else:
        with st.spinner("Generiere..."):
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=True)
                config = GPTConfig(**checkpoint_data["config"])
                model = GPT(config)
                model.load_state_dict(checkpoint_data["model"])
                model.to(device)
                model.eval()

                import tiktoken
                enc = tiktoken.get_encoding("gpt2")
                tokens = enc.encode(prompt)
                x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

                with torch.no_grad():
                    y = model.generate(x, max_tokens, temperature=temperature, top_k=top_k)
                    generated = enc.decode(y[0].tolist())
                    st.text_area("Generierter Text", generated, height=300)
            except Exception as e:
                st.error(f"Fehler: {e}")

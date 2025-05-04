import json
import os
import sys

import pandas as pd
import streamlit as st
import torch
from gensim.models import Word2Vec

# Add utils folder to sys.path
sys.path.append(os.path.abspath("utils"))

from kmer_2_vec import (
    SimpleCNN,
    build_embedding_matrix,
    build_vocab,
    predict_W2V_sequence,
)
from optuna_cnn_kmer_utils import DynamicCNN as KmerCNN
from optuna_cnn_kmer_utils import (
    build_kmer_vocab,
    load_optuna_cnn_kmer_model,
    predict_optuna_cnn_kmer,
)

# Import your model utilities
from optuna_cnn_one_hot_utils import DynamicCNN as OneHotCNN
from optuna_cnn_one_hot_utils import predict_onehot_sequence

# ------------------------------
# CONFIG
st.set_page_config(page_title="TFBS Predictor", layout="centered")


# ------------------------------
# MODEL LOADING
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # One-Hot CNN
    with open("Models/CNN_OH.json") as f:
        oh_config = json.load(f)
    model_oh = OneHotCNN(oh_config, input_len=101)
    model_oh.load_state_dict(
        torch.load("Models/50_CNN_OH.pt", map_location=device)
    )
    model_oh.to(device)

    # K-mer CNN
    kmer_vocab = build_kmer_vocab(k=5)
    model_kmer, _ = load_optuna_cnn_kmer_model(
        "Models/Cnn_kmer_50.pt",
        "Models/final_model_hparams.json",
        vocab_size=len(kmer_vocab) + 1,
        device=device,
    )

    # Word2Vec CNN
    w2v_model = Word2Vec.load("Models/kmer2vec_k_6_s_1.model")
    w2v_vocab = build_vocab(k=6)
    w2v_kv_dict = {k: w2v_model.wv[k] for k in w2v_model.wv.key_to_index}
    embedding_matrix = build_embedding_matrix(
        w2v_vocab, w2v_kv_dict, embedding_dim=w2v_model.vector_size
    )
    model_w2v = SimpleCNN(embedding_matrix)
    model_w2v.load_state_dict(
        torch.load("Models/50_W2V.pt", map_location=device)
    )
    model_w2v.to(device)

    return (
        model_oh,
        model_kmer,
        model_w2v,
        kmer_vocab,
        w2v_model,
        w2v_vocab,
        device,
    )


model_oh, model_kmer, model_w2v, kmer_vocab, w2v_model, w2v_vocab, device = (
    load_models()
)

# ------------------------------
# SIDEBAR NAV
page = st.sidebar.radio("📍 Navigation", ["🏠 Home", "🧬 Predict TFBS"])

# ------------------------------
# HOME PAGE
if page == "🏠 Home":
    st.title("🔬 Transcription Factor Binding Site (TFBS) Prediction Tool")
    st.markdown(
        """
Welcome to the **TFBS Prediction App**, a research-driven platform for identifying DNA sequences that are likely to be **Transcription Factor Binding Sites** using state-of-the-art deep learning models.

---

### 🔎 Why this project?

TFBSs are short regions of DNA that regulate gene expression by serving as docking points for transcription factors. Identifying them is critical to understanding gene regulation, epigenetics, and disease mechanisms. This app uses multiple machine learning models to enhance prediction accuracy over traditional motif-based approaches.

---

### 🧠 Models Integrated

- **CNN with One-Hot Encoding** (Fixed length: 101 bp)
- **CNN with K-mer Encoding** (k=5, stride=2)
- **CNN with Word2Vec Embeddings** (k=6, stride=1)

---

### 👨‍🔬 Project Authors

- **👨 Harsh Gujarathi & Dhairya Harshadbhai Chauhan** (Rutgers University)
- **👨‍🏫 Dr. Iman Dehzangi** (Mentor)
- **👨‍🏫 Dr. Sunil Shende** (Graduate Director)

---

📫 **Contact Information:**
- [harshgujarathi07@gmail.com](mailto:harshgujarathi07@gmail.com)
- [dhairyachauhan619@gmail.com](mailto:dhairyachauhan619@gmail.com)
"""
    )

# ------------------------------
# PREDICT PAGE
elif page == "🧬 Predict TFBS":
    st.title("🧪 DNA Sequence Classification")

    # Track user input with session_state
    if "sequence" not in st.session_state:
        st.session_state.sequence = ""

    model_choice = st.selectbox(
        "Select a model", ["CNN-OneHot", "CNN-Kmer", "CNN-Word2Vec"]
    )
    sequence = st.text_area(
        "Enter DNA Sequence (Only A, C, G, T):",
        value=st.session_state.sequence,
        height=150,
    )

    col1, col2 = st.columns([1, 1])
    predict_btn = col1.button("🚀 Predict")
    clear_btn = col2.button("🔄 Clear Input")

    if clear_btn:
        st.session_state.sequence = ""
        st.experimental_rerun()

    if predict_btn:
        st.session_state.sequence = sequence
        if not sequence:
            st.error("Please enter a DNA sequence.")
        else:
            sequence = (
                sequence.upper()
                .replace(" ", "")
                .replace("\n", "")
                .replace("\t", "")
            )
            if not set(sequence).issubset({"A", "C", "G", "T"}):
                st.error(
                    "❌ Sequence contains invalid characters. Only A, C, G, T are allowed."
                )
            else:
                try:
                    if model_choice == "CNN-OneHot":
                        if len(sequence) != 101:
                            st.warning(
                                "⚠️ Sequence length must be exactly 101 bp for One-Hot model."
                            )
                        else:
                            label, conf = predict_onehot_sequence(
                                model_oh, sequence, device
                            )

                    elif model_choice == "CNN-Kmer":
                        label, conf = predict_optuna_cnn_kmer(
                            sequence,
                            model_kmer,
                            kmer_vocab,
                            k=5,
                            stride=2,
                            max_len=96,
                            device=device,
                        )

                    elif model_choice == "CNN-Word2Vec":
                        label, conf = predict_W2V_sequence(
                            model_w2v,
                            w2v_model,
                            w2v_vocab,
                            sequence,
                            k=6,
                            stride=1,
                            max_len=96,
                            device=device,
                        )

                    if label == "TFBS":
                        st.success(
                            f"Prediction: **{label}** (Confidence: {conf}%)"
                        )
                    else:
                        st.error(
                            f"Prediction: **{label}** (Confidence: {conf}%)"
                        )

                except Exception as e:
                    st.error(f"⚠️ Error during prediction: {e}")

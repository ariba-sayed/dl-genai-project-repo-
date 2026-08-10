import json
import re

import torch
import streamlit as st

from models import Model3TextCNN


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "model3_textcnn_best.pt"
VOCAB_PATH = "vocab_scratch.json"

QUESTION_MAX_LEN = 64
OPTION_MAX_LEN = 96


# ============================================================
# EXACT SAME TOKENIZER USED DURING TRAINING
# ============================================================

TOKEN_RE = re.compile(
    r"[A-Za-z]+|\d+|[^\sA-Za-z\d]"
)


def tokenize_scratch(text):
    return TOKEN_RE.findall(text.lower())


# ============================================================
# LOAD VOCABULARY
# ============================================================

@st.cache_resource
def load_vocab():

    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        stoi = json.load(f)

    return stoi


stoi = load_vocab()


# ============================================================
# EXACT SAME ENCODING USED DURING TRAINING
# ============================================================

def encode(text, max_len):

    ids = [
        stoi.get(token, 1)
        for token in tokenize_scratch(text)
    ][:max_len]

    ids += [0] * (max_len - len(ids))

    return ids


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = Model3TextCNN(
        vocab_size=len(stoi)
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


model = load_model()


# ============================================================
# PREDICTION
# ============================================================

def predict(question, options):

    # Encode question
    q_ids = torch.tensor(
        [
            encode(
                question,
                QUESTION_MAX_LEN
            )
        ],
        dtype=torch.long
    )

    # Encode options
    opt_ids = [
        torch.tensor(
            [
                encode(
                    option,
                    OPTION_MAX_LEN
                )
            ],
            dtype=torch.long
        )
        for option in options
    ]

    # Inference
    with torch.no_grad():

        logits = model(
            q_ids,
            opt_ids
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    # Get best option
    predicted_index = torch.argmax(
        probabilities
    ).item()

    predicted_answer = chr(
        ord("A") + predicted_index
    )

    scores = {
        "A": float(probabilities[0]),
        "B": float(probabilities[1]),
        "C": float(probabilities[2]),
        "D": float(probabilities[3]),
        "E": float(probabilities[4])
    }

    return predicted_answer, scores


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Smart MCQ Solver",
    page_icon="🧠",
    layout="centered"
)


st.title("🧠 Smart MCQ Solver")

st.subheader(
    "Model 3 — Multi-Kernel TextCNN"
)

st.write(
    "Enter a multiple-choice question "
    "and five answer options."
)


# ============================================================
# QUESTION
# ============================================================

question = st.text_area(
    "Question",
    height=120,
    placeholder="Enter your question here..."
)


# ============================================================
# OPTIONS
# ============================================================

option_a = st.text_input("Option A")
option_b = st.text_input("Option B")
option_c = st.text_input("Option C")
option_d = st.text_input("Option D")
option_e = st.text_input("Option E")


# ============================================================
# PREDICT
# ============================================================

if st.button(
    "🔮 Predict Answer",
    use_container_width=True
):

    options = [
        option_a,
        option_b,
        option_c,
        option_d,
        option_e
    ]

    # Validate input
    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif not all(
        option.strip()
        for option in options
    ):

        st.warning(
            "Please enter all five options."
        )

    else:

        answer, scores = predict(
            question,
            options
        )

        # Prediction
        st.success(
            f"Predicted Answer: **{answer}**"
        )

        # Scores
        st.subheader("Option Scores")

        st.bar_chart(scores)

        for letter, score in scores.items():

            st.write(
                f"**Option {letter}:** "
                f"{score:.4f} "
                f"({score * 100:.2f}%)"
            )
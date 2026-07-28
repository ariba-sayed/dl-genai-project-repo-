"""
utils.py
Shared data loading, cleaning, splitting, vocabulary building, and
evaluation metrics used identically across all 3 models.
"""
import re
import random
import collections

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

OPTION_COLS = ["A", "B", "C", "D", "E"]
LABEL2IDX = {c: i for i, c in enumerate(OPTION_COLS)}
IDX2LABEL = {i: c for c, i in LABEL2IDX.items()}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_text(t):
    """Light cleanup only: fix whitespace. No lowercasing/stopword removal
    here -- that would hurt the pretrained model's subword tokenizer.
    Lowercasing happens separately, only inside tokenize() for the
    from-scratch models."""
    return re.sub(r"\s+", " ", str(t)).strip()


def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    required = {"id", "prompt", "A", "B", "C", "D", "E", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df = df.dropna(subset=["prompt", "A", "B", "C", "D", "E", "answer"]).drop_duplicates()
    df["prompt"] = df["prompt"].apply(clean_text)
    for c in OPTION_COLS:
        df[c] = df[c].apply(clean_text)
    df["label"] = df["answer"].map(LABEL2IDX)
    if df["label"].isna().any():
        raise ValueError("Found answers outside A-E")
    df["label"] = df["label"].astype(int)
    return df


def load_test(csv_path):
    df = pd.read_csv(csv_path)
    df["prompt"] = df["prompt"].apply(clean_text)
    for c in OPTION_COLS:
        df[c] = df[c].apply(clean_text)
    return df


def split_train_val(df, val_size=0.15, seed=42):
    """Stratified split on label so class balance (A-E) is preserved."""
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=seed, stratify=df["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Hand-built vocab + tokenizer for the from-scratch models (Model 1, Model 3)
# --------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")


def tokenize(t):
    return TOKEN_RE.findall(t.lower())


class Vocab:
    def __init__(self, min_freq=2, max_size=30000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.stoi = {"<pad>": 0, "<unk>": 1}

    def build(self, texts):
        counter = collections.Counter()
        for t in texts:
            counter.update(tokenize(t))
        for tok, freq in counter.most_common():
            if freq < self.min_freq or len(self.stoi) >= self.max_size:
                continue
            if tok not in self.stoi:
                self.stoi[tok] = len(self.stoi)
        return self

    def encode(self, text, max_len):
        ids = [self.stoi.get(t, 1) for t in tokenize(text)][:max_len]
        return ids + [0] * (max_len - len(ids))

    def __len__(self):
        return len(self.stoi)


def build_vocab(train_df, min_freq=2, max_size=30000):
    texts = list(train_df["prompt"])
    for c in OPTION_COLS:
        texts += list(train_df[c])
    return Vocab(min_freq, max_size).build(texts)


# --------------------------------------------------------------------------
# Shared evaluation metrics (identical across all 3 models)
# --------------------------------------------------------------------------
def map_at_3(logits, labels):
    """Rank options by score; 1/(rank+1) if true label in top 3, else 0."""
    order = np.argsort(-logits, axis=1)[:, :3]
    scores = np.zeros(len(labels))
    for i, lab in enumerate(labels):
        hits = np.where(order[i] == lab)[0]
        if len(hits):
            scores[i] = 1.0 / (hits[0] + 1)
    return scores.mean()


def evaluate_all(logits, labels):
    preds = logits.argmax(1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "map@3": map_at_3(logits, labels),
    }


def top3_labels(logits):
    order = np.argsort(-logits, axis=1)[:, :3]
    return [" ".join(IDX2LABEL[i] for i in row) for row in order]


def shuffled_option_val(df, seed=123):
    """Shuffles A-E option order per row and remaps the label. Used as a
    stress test: if a model's score drops sharply on this vs. the original
    order, it was relying on option POSITION rather than content."""
    df2 = df.copy()
    for i in df2.index:
        cols = OPTION_COLS.copy()
        random.Random(seed + i).shuffle(cols)
        orig_vals = {c: df2.loc[i, c] for c in OPTION_COLS}
        new_label = None
        for new_pos, old_col in enumerate(cols):
            df2.loc[i, OPTION_COLS[new_pos]] = orig_vals[old_col]
            if old_col == OPTION_COLS[df.loc[i, "label"]]:
                new_label = new_pos
        df2.loc[i, "label"] = new_label
    return df2
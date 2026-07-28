"""
inference.py
Generates Kaggle submissions for all 3 trained models. Run directly:
python inference.py
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from models import Model1ConvAttention, Model3TextCNN
from utils import load_test, top3_labels, OPTION_COLS, Vocab

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_PATH = "data/test.csv"
CHECKPOINT_DIR = "models"
RESULTS_DIR = "results"
MAX_LEN = 128
os.makedirs(RESULTS_DIR, exist_ok=True)

test_df = load_test(TEST_PATH)

vocab = Vocab()
vocab.stoi = json.load(open(os.path.join(CHECKPOINT_DIR, "vocab.json")))


def predict_scratch(model, qlen=64, olen=96, batch_size=32):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for start in range(0, len(test_df), batch_size):
            batch = test_df.iloc[start:start + batch_size]
            q_ids = torch.tensor([vocab.encode(t, qlen) for t in batch["prompt"]]).to(device)
            opt_ids = [torch.tensor([vocab.encode(t, olen) for t in batch[c]]).to(device) for c in OPTION_COLS]
            logits = model(q_ids, opt_ids)
            all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits)


def predict_pretrained(model, tokenizer, max_len=MAX_LEN, batch_size=16):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for start in range(0, len(test_df), batch_size):
            batch = test_df.iloc[start:start + batch_size]
            first = sum([[q] * 5 for q in batch["prompt"]], [])
            second = sum([[batch.iloc[j][c] for c in OPTION_COLS] for j in range(len(batch))], [])
            enc = tokenizer(first, second, truncation="only_second", max_length=max_len,
                             padding="max_length", return_tensors="pt")
            enc = {k: v.view(len(batch), 5, -1).to(device) for k, v in enc.items()}
            out = model(**enc)
            all_logits.append(out.logits.cpu().numpy())
    return np.concatenate(all_logits)


def save_submission(logits, name):
    submission = pd.DataFrame()
    submission["id"] = test_df["id"]
    submission["Prediction"] = top3_labels(logits)
    out_path = os.path.join(RESULTS_DIR, f"submission_{name}.csv")
    submission.to_csv(out_path, index=False)
    print(f"Wrote {len(submission)} predictions to {out_path}")
    print(submission.head())
    return submission


# ==========================================================================
# MODEL 1 -- Conv+Attention (scratch)
# ==========================================================================

model1 = Model1ConvAttention(len(vocab)).to(device)
model1.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "model1_best.pt"), map_location=device))
test_logits_model1 = predict_scratch(model1)
submission_model1 = save_submission(test_logits_model1, "model1")


# ==========================================================================
# MODEL 2 -- RoBERTa (pretrained, k-fold ensemble)
# ==========================================================================
fold_dirs = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "model2_roberta_fold*")))
print("found RoBERTa fold checkpoints:", fold_dirs)

all_fold_logits = []
for fold_dir in fold_dirs:
    tokenizer = AutoTokenizer.from_pretrained(fold_dir)
    model2 = AutoModelForMultipleChoice.from_pretrained(fold_dir).to(device)
    fold_logits = predict_pretrained(model2, tokenizer)
    all_fold_logits.append(fold_logits)
    del model2
    torch.cuda.empty_cache()

test_logits_model2 = np.mean(all_fold_logits, axis=0)
submission_model2 = save_submission(test_logits_model2, "model2_roberta")


# ==========================================================================
# MODEL 3 -- TextCNN (scratch)
# ==========================================================================

model3 = Model3TextCNN(len(vocab)).to(device)
model3.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "model3_best.pt"), map_location=device))
test_logits_model3 = predict_scratch(model3)
submission_model3 = save_submission(test_logits_model3, "model3")


# ==========================================================================
# Fingerprint check -- confirm all 3 submissions are genuinely different
# ==========================================================================
print("\nModel1 vs Model2 identical:", (submission_model1["Prediction"].values == submission_model2["Prediction"].values).all())
print("Model1 vs Model3 identical:", (submission_model1["Prediction"].values == submission_model3["Prediction"].values).all())
print("Model2 vs Model3 identical:", (submission_model2["Prediction"].values == submission_model3["Prediction"].values).all())
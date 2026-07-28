"""
train.py
Trains all 3 models back to back:
  Model 1 -- from scratch: Conv1d + attention pooling
  Model 2 -- pretrained RoBERTa, fine-tuned with k-fold cross-validation
  Model 3 -- from scratch: multi-kernel TextCNN
Run directly: python train.py
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForMultipleChoice, get_linear_schedule_with_warmup
import wandb

from utils import set_seed, load_and_preprocess, split_train_val, build_vocab, evaluate_all, OPTION_COLS
from models import (
    Model1ConvAttention, Model3TextCNN,
    MCQDatasetScratch, collate_scratch,
    MCQDatasetBert, collate_bert,
)

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


DATA_PATH = "data/train.csv"
CHECKPOINT_DIR = "models"
WANDB_PROJECT = "smart-mcq-solver"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

wandb.login() 

df = load_and_preprocess(DATA_PATH)
train_split, val_split = split_train_val(df, seed=42)


# ==========================================================================
# MODEL 1 -- from scratch: Conv1d + attention pooling
# ==========================================================================
vocab = build_vocab(train_split)
print("vocab size:", len(vocab))
json.dump(vocab.stoi, open(os.path.join(CHECKPOINT_DIR, "vocab.json"), "w"))


train_loader_scratch = DataLoader(MCQDatasetScratch(train_split, vocab), batch_size=32, shuffle=True, collate_fn=collate_scratch)
val_loader_scratch = DataLoader(MCQDatasetScratch(val_split, vocab), batch_size=32, collate_fn=collate_scratch)

model1 = Model1ConvAttention(len(vocab)).to(device)
opt_model1 = torch.optim.Adam(model1.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn_scratch = nn.CrossEntropyLoss()

wandb.init(project=WANDB_PROJECT, name="model1_conv_attention", reinit=True)
best_map3_model1 = -1
for epoch in range(20):
    model1.train()
    total_loss = 0
    for q, opts, labels in train_loader_scratch:
        q, opts, labels = q.to(device), [o.to(device) for o in opts], labels.to(device)
        logits = model1(q, opts)
        loss = loss_fn_scratch(logits, labels)
        opt_model1.zero_grad(); loss.backward(); opt_model1.step()
        total_loss += loss.item() * len(labels)

    model1.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for q, opts, labels in val_loader_scratch:
            q, opts = q.to(device), [o.to(device) for o in opts]
            logits = model1(q, opts)
            all_logits.append(logits.cpu().numpy()); all_labels.append(labels.numpy())
    logits = np.concatenate(all_logits); labels_np = np.concatenate(all_labels)
    m = evaluate_all(logits, labels_np)
    print("Model1", epoch, "train_loss:", total_loss / len(train_split), m)
    wandb.log({"epoch": epoch, "train_loss": total_loss / len(train_split), **{f"val_{k}": v for k, v in m.items()}})
    if m["map@3"] > best_map3_model1:
        best_map3_model1 = m["map@3"]
        torch.save(model1.state_dict(), os.path.join(CHECKPOINT_DIR, "model1_best.pt"))

wandb.summary["best_val_map@3"] = best_map3_model1
wandb.finish()
print("Model 1 best val MAP@3:", best_map3_model1)


# ==========================================================================
# MODEL 2 -- pretrained RoBERTa, fine-tuned with k-fold cross-validation
# ==========================================================================
ROBERTA_MODEL_NAME = "roberta-base"
N_FOLDS = 3
ROBERTA_EPOCHS = 2
ROBERTA_LR = 1e-5
ROBERTA_BATCH_SIZE = 8
MAX_LEN = 128


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_scores_roberta = []

for fold_id, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
    fold_train = df.iloc[train_idx].reset_index(drop=True)
    fold_val = df.iloc[val_idx].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    model2 = AutoModelForMultipleChoice.from_pretrained(ROBERTA_MODEL_NAME).to(device)

    train_loader_bert = DataLoader(MCQDatasetBert(fold_train, tokenizer, MAX_LEN), batch_size=ROBERTA_BATCH_SIZE, shuffle=True, collate_fn=collate_bert)
    val_loader_bert = DataLoader(MCQDatasetBert(fold_val, tokenizer, MAX_LEN), batch_size=ROBERTA_BATCH_SIZE, collate_fn=collate_bert)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_params = [
        {"params": [p for n, p in model2.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model2.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt_model2 = torch.optim.AdamW(optimizer_params, lr=ROBERTA_LR)
    total_steps = len(train_loader_bert) * ROBERTA_EPOCHS
    scheduler = get_linear_schedule_with_warmup(opt_model2, int(total_steps * 0.1), total_steps)

    wandb.init(project=WANDB_PROJECT, name=f"model2_roberta_fold{fold_id}", reinit=True)
    best_map3_fold, best_state = -1, None

    for epoch in range(ROBERTA_EPOCHS):
        model2.train()
        total_loss, n_seen = 0, 0
        for batch in train_loader_bert:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model2(**batch, labels=labels)
            loss = out.loss
            if torch.isnan(loss):
                opt_model2.zero_grad(); continue
            opt_model2.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
            opt_model2.step(); scheduler.step()
            total_loss += loss.item() * len(labels); n_seen += len(labels)

        model2.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader_bert:
                labels = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model2(**batch)
                all_logits.append(out.logits.cpu().numpy()); all_labels.append(labels.numpy())
        logits = np.concatenate(all_logits); labels_np = np.concatenate(all_labels)
        m = evaluate_all(logits, labels_np)
        avg_loss = total_loss / max(n_seen, 1)
        print(f"Model2 fold {fold_id} epoch {epoch}: train_loss={avg_loss:.4f}", m)
        wandb.log({"epoch": epoch, "train_loss": avg_loss, **{f"val_{k}": v for k, v in m.items()}})

        if m["map@3"] > best_map3_fold:
            best_map3_fold = m["map@3"]
            best_state = {k: v.cpu().clone() for k, v in model2.state_dict().items()}

    model2.load_state_dict(best_state)
    wandb.summary["best_val_map@3"] = best_map3_fold
    wandb.finish()

    save_dir = os.path.join(CHECKPOINT_DIR, f"model2_roberta_fold{fold_id}")
    model2.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    fold_scores_roberta.append(best_map3_fold)
    del model2
    torch.cuda.empty_cache()

print("Model 2 (RoBERTa) fold val MAP@3 scores:", fold_scores_roberta)
print("mean:", np.mean(fold_scores_roberta), "std:", np.std(fold_scores_roberta))


# ==========================================================================
# MODEL 3 -- from scratch: multi-kernel TextCNN (no attention -- kept
# architecturally distinct from Model 1)
# ==========================================================================
model3 = Model3TextCNN(len(vocab)).to(device)
opt_model3 = torch.optim.Adam(model3.parameters(), lr=1e-3, weight_decay=1e-5)

wandb.init(project=WANDB_PROJECT, name="model3_textcnn", reinit=True)
best_map3_model3 = -1
for epoch in range(20):
    model3.train()
    total_loss = 0
    for q, opts, labels in train_loader_scratch:
        q, opts, labels = q.to(device), [o.to(device) for o in opts], labels.to(device)
        logits = model3(q, opts)
        loss = loss_fn_scratch(logits, labels)
        opt_model3.zero_grad(); loss.backward(); opt_model3.step()
        total_loss += loss.item() * len(labels)

    model3.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for q, opts, labels in val_loader_scratch:
            q, opts = q.to(device), [o.to(device) for o in opts]
            logits = model3(q, opts)
            all_logits.append(logits.cpu().numpy()); all_labels.append(labels.numpy())
    logits = np.concatenate(all_logits); labels_np = np.concatenate(all_labels)
    m = evaluate_all(logits, labels_np)
    print("Model3", epoch, "train_loss:", total_loss / len(train_split), m)
    wandb.log({"epoch": epoch, "train_loss": total_loss / len(train_split), **{f"val_{k}": v for k, v in m.items()}})
    if m["map@3"] > best_map3_model3:
        best_map3_model3 = m["map@3"]
        torch.save(model3.state_dict(), os.path.join(CHECKPOINT_DIR, "model3_best.pt"))

wandb.summary["best_val_map@3"] = best_map3_model3
wandb.finish()
print("Model 3 best val MAP@3:", best_map3_model3)


# ==========================================================================
# Final summary
# ==========================================================================
print("\n=== ALL MODELS TRAINED ===")
print("Model 1 (Conv+Attention, scratch):", best_map3_model1)
print("Model 2 (RoBERTa, pretrained, 5-fold mean):", np.mean(fold_scores_roberta))
print("Model 3 (TextCNN, scratch):", best_map3_model3)
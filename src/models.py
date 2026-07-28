"""
models.py
Model architecture definitions only, shared by train.py and inference.py.
  Model 1 -- from scratch: Conv1d + attention pooling
  Model 3 -- from scratch: multi-kernel TextCNN
(Model 2 uses HuggingFace's AutoModelForMultipleChoice directly, so it
doesn't need a custom class here.)
"""
import torch
import torch.nn as nn


class Model1ConvAttention(nn.Module):
    def __init__(self, vocab_size, emb_dim=150, conv_channels=150, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv = nn.Conv1d(emb_dim, conv_channels, kernel_size=3, padding=1)
        self.conv_act = nn.ReLU()
        self.attn_proj = nn.Linear(conv_channels, 1)
        pooled_dim = conv_channels * 3
        combo_dim = pooled_dim * 4
        self.scorer = nn.Sequential(
            nn.Linear(combo_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, ids):
        mask = (ids != 0).unsqueeze(-1).float()
        emb = self.embedding(ids)
        conv_out = self.conv_act(self.conv(emb.transpose(1, 2))).transpose(1, 2)
        feats = conv_out * mask
        mean_pool = feats.sum(1) / mask.sum(1).clamp(min=1e-6)
        max_pool = feats.masked_fill(mask == 0, -1e9).max(1).values
        attn_scores = self.attn_proj(feats).squeeze(-1).masked_fill(mask.squeeze(-1) == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        attn_pool = (feats * attn_weights).sum(1)
        return torch.cat([mean_pool, max_pool, attn_pool], dim=-1)

    def score(self, q_vec, o_vec):
        combo = torch.cat([q_vec, o_vec, q_vec * o_vec, torch.abs(q_vec - o_vec)], dim=-1)
        return self.scorer(combo).squeeze(-1)

    def forward(self, q_ids, opt_ids_list):
        q_vec = self.encode(q_ids)
        return torch.stack([self.score(q_vec, self.encode(o)) for o in opt_ids_list], dim=1)


class Model3TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, num_filters=100, kernel_sizes=(2, 3, 4, 5), dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, num_filters, k, padding=k // 2) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        out_dim = num_filters * len(kernel_sizes)
        combo_dim = out_dim * 4
        self.scorer = nn.Sequential(nn.Linear(combo_dim, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1))

    def encode(self, ids):
        emb = self.dropout(self.embedding(ids)).transpose(1, 2)
        pooled = [torch.max(torch.relu(conv(emb)), dim=2).values for conv in self.convs]
        return torch.cat(pooled, dim=-1)

    def score(self, q_vec, o_vec):
        combo = torch.cat([q_vec, o_vec, q_vec * o_vec, torch.abs(q_vec - o_vec)], dim=-1)
        return self.scorer(combo).squeeze(-1)

    def forward(self, q_ids, opt_ids_list):
        q_vec = self.encode(q_ids)
        return torch.stack([self.score(q_vec, self.encode(o)) for o in opt_ids_list], dim=1)


class MCQDatasetScratch(torch.utils.data.Dataset):
    def __init__(self, df, vocab, qlen=64, olen=96):
        self.df, self.vocab, self.qlen, self.olen = df, vocab, qlen, olen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        from utils import OPTION_COLS
        row = self.df.iloc[i]
        q = torch.tensor(self.vocab.encode(row["prompt"], self.qlen))
        opts = [torch.tensor(self.vocab.encode(row[c], self.olen)) for c in OPTION_COLS]
        return q, opts, torch.tensor(row["label"])


def collate_scratch(batch):
    q = torch.stack([b[0] for b in batch])
    opts = [torch.stack([b[1][i] for b in batch]) for i in range(5)]
    labels = torch.stack([b[2] for b in batch])
    return q, opts, labels


class MCQDatasetBert(torch.utils.data.Dataset):
    def __init__(self, df, tok, max_len=192):
        self.df, self.tok, self.max_len = df, tok, max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        from utils import OPTION_COLS
        row = self.df.iloc[i]
        q = [row["prompt"]] * 5
        opts = [row[c] for c in OPTION_COLS]
        enc = self.tok(q, opts, truncation="only_second", max_length=self.max_len,
                        padding="max_length", return_tensors="pt")
        enc = dict(enc)
        enc["labels"] = torch.tensor(row["label"])
        return enc


def collate_bert(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
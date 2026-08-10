import torch
import torch.nn as nn


class Model3TextCNN(nn.Module):

    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        num_filters=100,
        kernel_sizes=(2, 3, 4, 5),
        dropout=0.3
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=0
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                emb_dim,
                num_filters,
                k,
                padding=k // 2
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)

        out_dim = num_filters * len(kernel_sizes)
        combo_dim = out_dim * 4

        self.scorer = nn.Sequential(
            nn.Linear(combo_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def encode(self, ids):

        emb = self.dropout(
            self.embedding(ids)
        ).transpose(1, 2)

        pooled = [
            torch.max(
                torch.relu(conv(emb)),
                dim=2
            ).values
            for conv in self.convs
        ]

        return torch.cat(pooled, dim=-1)

    def score(self, q_vec, o_vec):

        combo = torch.cat([
            q_vec,
            o_vec,
            q_vec * o_vec,
            torch.abs(q_vec - o_vec)
        ], dim=-1)

        return self.scorer(combo).squeeze(-1)

    def forward(self, q_ids, opt_ids_list):

        q_vec = self.encode(q_ids)

        return torch.stack([
            self.score(
                q_vec,
                self.encode(option)
            )
            for option in opt_ids_list
        ], dim=1)
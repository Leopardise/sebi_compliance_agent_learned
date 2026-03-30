from __future__ import annotations

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden_dim: int = 96, pad_idx: int = 0, dropout: float = 0.2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        x = self.dropout(x)
        out, _ = self.gru(x)
        pooled, _ = torch.max(out, dim=1)
        return pooled


class PairScorer(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden_dim: int = 96, pad_idx: int = 0, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = TextEncoder(vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim, pad_idx=pad_idx, dropout=dropout)
        rep_dim = hidden_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(rep_dim * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, clause_ids: torch.Tensor, title_ids: torch.Tensor) -> torch.Tensor:
        c = self.encoder(clause_ids)
        t = self.encoder(title_ids)
        feat = torch.cat([c, t, torch.abs(c - t), c * t], dim=-1)
        logits = self.mlp(feat).squeeze(-1)
        return logits
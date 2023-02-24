import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_len=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, n_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embedding_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout(self.attention(x2, x2, x2, key_padding_mask=mask)[0])
        x2 = self.norm2(x)
        x = x + self.dropout(self.feed_forward(x2))
        return x


class TextClassificationModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        n_heads,
        ff_dim,
        n_layers,
        num_classes,
        drop_out=0.1,
        max_seq_len=256,
    ):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEmbedding(
            embedding_dim, drop_out, max_seq_len
        )
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, n_heads, ff_dim, drop_out)
                for _ in range(n_layers)
            ]
        )
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_embedding(x)

        if mask is not None:
            mask = (1 - mask).bool()

        for transformer in self.transformer:
            x = transformer(x, mask)
        x = x.mean(dim=1)
        x = self.linear(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_casual_attention_mask


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, n_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.multi_head_attn = nn.MultiheadAttention(
            embedding_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim),
        )

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        casual_mask = get_casual_attention_mask(
            batch_size, seq_len, seq_len, seq_len, x.dtype
        )

        # make a mask for padding tokens
        key_padding_mask = (1 - mask).bool() if mask is not None else None
        context_vector, _ = self.multi_head_attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            # attn_mask=casual_mask
        )
        context_vector = self.dropout1(context_vector)
        out1 = self.layer_norm1(x + context_vector)

        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)

        return self.layer_norm2(out1 + ffn_out)


# %%
class TokenAndPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embedding_dim):
        super(TokenAndPositionalEmbedding, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)

    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(x.device)
        return self.token_embedding(x) + self.pos_embedding(positions)


# %%
class GeneratorModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        embedding_dim,
        n_heads,
        ff_dim,
        n_layers,
        dropout=0.1,
    ):
        super(GeneratorModel, self).__init__()

        self.embedding = TokenAndPositionalEmbedding(
            vocab_size, max_seq_len, embedding_dim
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, n_heads, ff_dim, dropout)
                for _ in range(n_layers)
            ]
        )
        self.fc = nn.Linear(embedding_dim, vocab_size)

        self.apply(init_weights)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc(x)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Embedding):
        torch.nn.init.normal_(m.weight, std=0.02)
    elif isinstance(m, nn.LayerNorm):
        torch.nn.init.normal_(m.weight, std=0.02)
        torch.nn.init.constant_(m.bias, 0)

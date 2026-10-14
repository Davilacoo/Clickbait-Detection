from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


def _init_linear(layer: nn.Linear) -> None:
    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class _MLPBase(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_sizes: Sequence[int],
                 dropout: float,
                 use_batchnorm: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            linear = nn.Linear(prev, h)
            _init_linear(linear)
            layers.append(linear)
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        out = nn.Linear(prev, 1)
        _init_linear(out)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ShallowMLP(_MLPBase):

    def __init__(self, input_dim: int):
        super().__init__(
            input_dim=input_dim,
            hidden_sizes=config.SHALLOW_HIDDEN,
            dropout=config.SHALLOW_DROPOUT,
            use_batchnorm=False,
        )


class DeepMLP(_MLPBase):

    def __init__(self, input_dim: int):
        super().__init__(
            input_dim=input_dim,
            hidden_sizes=config.DEEP_HIDDEN,
            dropout=config.DEEP_DROPOUT,
            use_batchnorm=True,
        )


class TextCNN(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBED_DIM,
            padding_idx=config.PAD_IDX,
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=config.EMBED_DIM,
                      out_channels=config.CNN_NUM_FILTERS,
                      kernel_size=k)
            for k in config.CNN_FILTER_SIZES
        ])
        self.dropout = nn.Dropout(config.CNN_DROPOUT)
        total_filters = config.CNN_NUM_FILTERS * len(config.CNN_FILTER_SIZES)
        self.fc = nn.Linear(total_filters, 1)
        _init_linear(self.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        emb = emb.transpose(1, 2)
        feats = []
        for conv in self.convs:
            c = F.relu(conv(emb))
            p = F.max_pool1d(c, c.size(-1))
            feats.append(p.squeeze(-1))
        h = torch.cat(feats, dim=1)
        h = self.dropout(h)
        return self.fc(h).squeeze(-1)

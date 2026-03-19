#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage 1: Hit Probability Prediction Model
预测光子击中PMT的概率

模型架构：
- Fourier特征编码
- 双头结构：gate head (是否击中) + regression head (击中概率)
- 最终输出：p_hat = gate * reg
"""

import math
import torch
import torch.nn as nn

class FourierFeatures(nn.Module):
    """
    Fourier feature mapping for NeRF-like positional encoding.
    x: (B, D) -> (B, D*(2*num_freqs)+D) if include_input=True else (B, D*(2*num_freqs))
    """
    def __init__(self, input_dim: int, num_freqs: int = 6, include_input: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.include_input = include_input

        # frequencies: 2^0, 2^1, ..., 2^(num_freqs-1)
        self.freq_bands = 2.0 ** torch.arange(num_freqs).float() * math.pi  # (num_freqs,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D), assumed already normalized (z-score or [-1,1])
        """
        if x.dim() != 2:
            x = x.view(x.size(0), -1)

        B, D = x.shape
        freqs = self.freq_bands.to(x.device)  # (F,)

        # (B, D, F)
        xb = x.unsqueeze(-1) * freqs.view(1, 1, -1)

        sin = torch.sin(xb)
        cos = torch.cos(xb)

        # (B, D*F)
        sin = sin.reshape(B, D * self.num_freqs)
        cos = cos.reshape(B, D * self.num_freqs)

        if self.include_input:
            return torch.cat([x, sin, cos], dim=-1)
        else:
            return torch.cat([sin, cos], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, n_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        layers = []
        dim = in_dim
        for i in range(n_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TwoStageProbNet(nn.Module):
    """
    Two-head model:
      - gate head: predict g(x) = P(p>0 | x) in [0,1]
      - reg  head: predict r(x) in [0,1] for nonzero region
    final p_hat = g * r

    NOTE: Works best when x is normalized.
    """
    def __init__(
        self,
        input_dim: int = 6,
        num_freqs: int = 6,
        include_input: bool = True,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.ff = FourierFeatures(input_dim=input_dim, num_freqs=num_freqs, include_input=include_input)
        ff_out_dim = input_dim * (2 * num_freqs) + (input_dim if include_input else 0)

        self.backbone = MLP(in_dim=ff_out_dim, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)

        # gate head -> sigmoid
        self.gate_head = nn.Linear(hidden_dim, 1)

        # reg head -> sigmoid (continuous probability in [0,1])
        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.ff(x)
        h = self.backbone(h)

        gate = torch.sigmoid(self.gate_head(h)).squeeze(-1)  # (B,)
        reg = torch.sigmoid(self.reg_head(h)).squeeze(-1)    # (B,)

        p_hat = gate * reg
        return p_hat, gate, reg

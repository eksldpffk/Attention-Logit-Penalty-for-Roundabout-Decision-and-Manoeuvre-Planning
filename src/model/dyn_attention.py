import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.pairwise_feats import D_PAIR


class InteractionAwareAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        lambda_bias: float = 1.0,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.lambda_bias = lambda_bias

        self.q_proj = nn.Linear(
            d_model,
            d_model,
        )
        self.k_proj = nn.Linear(
            d_model,
            d_model,
        )
        self.v_proj = nn.Linear(
            d_model,
            d_model,
        )
        self.out_proj = nn.Linear(
            d_model,
            d_model,
        )

        self.bias_mlp = nn.Sequential(
            nn.Linear(D_PAIR, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_heads),
        )

        nn.init.zeros_(
            self.bias_mlp[-1].weight
        )
        nn.init.zeros_(
            self.bias_mlp[-1].bias
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        pair_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, length, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        logits = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        logits = logits / math.sqrt(
            self.head_dim
        )

        if pair_feats is not None:
            bias = self.bias_mlp(
                pair_feats
            )

            bias = torch.tanh(
                bias
            )

            bias = bias.permute(
                0,
                3,
                1,
                2,
            )

            diagonal = torch.eye(
                length,
                device=x.device,
                dtype=bias.dtype,
            ).view(
                1,
                1,
                length,
                length,
            )

            bias = bias * (
                1.0 - diagonal
            )

            logits = (
                logits
                + self.lambda_bias * bias
            )

        if key_padding_mask is not None:
            mask = key_padding_mask[
                :,
                None,
                None,
                :,
            ]

            logits = logits.masked_fill(
                mask,
                torch.finfo(
                    logits.dtype
                ).min,
            )

        attention = F.softmax(
            logits,
            dim=-1,
        )

        attention = self.dropout(
            attention
        )

        output = torch.matmul(
            attention,
            v,
        )

        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                length,
                self.d_model,
            )
        )

        return self.out_proj(
            output
        )

    def _split_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, length, _ = x.shape

        return (
            x.view(
                batch_size,
                length,
                self.n_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
import torch
import torch.nn as nn

from src.model.encoder import SceneEncoder
from src.model.heads import MotionHead


class RoundaboutMotionModel(nn.Module):
    def __init__(
        self,
        d_in: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
        attn_type: str = "vanilla",
        lambda_bias: float = 1.0,
    ) -> None:
        super().__init__()

        self.encoder = SceneEncoder(
            d_in=d_in,
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            dropout=dropout,
            attn_type=attn_type,
            lambda_bias=lambda_bias,
        )

        self.motion_head = MotionHead(
            d_model=d_model,
            out_dim=5,
        )

    def forward(
        self,
        x_tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        pair_feats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(
            x_tokens,
            key_padding_mask=key_padding_mask,
            pair_feats=pair_feats,
        )

        ego_embedding = encoded[:, 0]

        return self.motion_head(
            ego_embedding
        )
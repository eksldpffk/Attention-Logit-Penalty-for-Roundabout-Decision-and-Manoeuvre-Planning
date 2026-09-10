import torch
import torch.nn as nn

from src.model.dyn_attention import InteractionAwareAttention


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        attn_type: str = "vanilla",
        lambda_bias: float = 1.0,
    ) -> None:
        super().__init__()

        self.attn_type = attn_type

        if attn_type == "vanilla":
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
        elif attn_type == "interaction":
            self.attn = InteractionAwareAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                lambda_bias=lambda_bias,
            )
        else:
            raise ValueError(
                f"Unknown attention type: {attn_type}"
            )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        pair_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)

        if self.attn_type == "vanilla":
            attn_output, _ = self.attn(
                x,
                x,
                x,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        else:
            attn_output = self.attn(
                x,
                key_padding_mask=key_padding_mask,
                pair_feats=pair_feats,
            )

        x = residual + self.dropout(attn_output)

        residual = x
        x = self.norm2(x)

        x = residual + self.dropout(
            self.ffn(x)
        )

        return x


class SceneEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int = 128,
        n_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
        attn_type: str = "vanilla",
        lambda_bias: float = 1.0,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(
            d_in,
            d_model,
        )

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    attn_type=attn_type,
                    lambda_bias=lambda_bias,
                )
                for _ in range(depth)
            ]
        )

        self.output_norm = nn.LayerNorm(
            d_model
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        pair_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_projection(x)

        for layer in self.layers:
            x = layer(
                x,
                key_padding_mask=key_padding_mask,
                pair_feats=pair_feats,
            )

        return self.output_norm(x)
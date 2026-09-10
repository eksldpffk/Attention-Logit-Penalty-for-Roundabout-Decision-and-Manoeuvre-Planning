import math

import torch


D_PAIR = 10

POSITION_SCALE = 30.0
VELOCITY_SCALE = 15.0
TTC_MAX = 10.0


def build_pairwise_feats(
    x_tokens: torch.Tensor,
) -> torch.Tensor:
    x = x_tokens[..., 0]
    y = x_tokens[..., 1]
    vx = x_tokens[..., 2]
    vy = x_tokens[..., 3]
    sin_heading = x_tokens[..., 6]
    cos_heading = x_tokens[..., 7]

    dx = x.unsqueeze(1) - x.unsqueeze(2)
    dy = y.unsqueeze(1) - y.unsqueeze(2)

    dvx = vx.unsqueeze(1) - vx.unsqueeze(2)
    dvy = vy.unsqueeze(1) - vy.unsqueeze(2)

    distance = torch.sqrt(
        dx.square() + dy.square() + 1e-6
    )

    closing_speed = -(
        dx * dvx + dy * dvy
    ) / distance.clamp_min(1e-3)

    ttc = torch.where(
        closing_speed > 1e-3,
        distance / closing_speed.clamp_min(1e-3),
        torch.full_like(distance, TTC_MAX),
    )

    ttc = ttc.clamp(
        min=0.0,
        max=TTC_MAX,
    )

    sin_i = sin_heading.unsqueeze(2)
    cos_i = cos_heading.unsqueeze(2)

    sin_j = sin_heading.unsqueeze(1)
    cos_j = cos_heading.unsqueeze(1)

    sin_relative_heading = (
        sin_j * cos_i
        - cos_j * sin_i
    )

    cos_relative_heading = (
        cos_j * cos_i
        + sin_j * sin_i
    )

    inverse_ttc = 1.0 / (1.0 + ttc)

    features = torch.stack(
        [
            dx / POSITION_SCALE,
            dy / POSITION_SCALE,
            dvx / VELOCITY_SCALE,
            dvy / VELOCITY_SCALE,
            distance / POSITION_SCALE,
            closing_speed / VELOCITY_SCALE,
            ttc / TTC_MAX,
            inverse_ttc,
            sin_relative_heading,
            cos_relative_heading,
        ],
        dim=-1,
    )

    length = x_tokens.size(1)

    diagonal = torch.eye(
        length,
        device=x_tokens.device,
        dtype=torch.bool,
    ).unsqueeze(0).unsqueeze(-1)

    features = features.masked_fill(
        diagonal,
        0.0,
    )

    return features
import argparse
import math
import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset

from src.datasets.collate import collate_batch
from src.datasets.torchsets import RoundDTokenDataset
from src.features.pairwise_feats import TTC_MAX, build_pairwise_feats
from src.model.model import RoundaboutMotionModel
from src.utils import make_targets


TARGET_SCALE = torch.tensor(
    [20.0, 20.0, 15.0, 1.0, 1.0],
    dtype=torch.float32,
)


def get_test_split(
    dataset: RoundDTokenDataset,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Subset:
    indices_by_recording = defaultdict(list)

    for idx, scene in enumerate(dataset.frames):
        recording_id = str(scene["recording_id"])
        indices_by_recording[recording_id].append(idx)

    recording_ids = list(indices_by_recording.keys())

    if len(recording_ids) < 3:
        raise ValueError(
            "At least three recordings are required "
            "for train/val/test splitting"
        )

    rng = random.Random(seed)
    rng.shuffle(recording_ids)

    n_recordings = len(recording_ids)

    n_train = max(1, int(train_frac * n_recordings))
    n_val = max(1, int(val_frac * n_recordings))

    if n_train + n_val >= n_recordings:
        n_val = 1
        n_train = n_recordings - 2

    test_recordings = set(
        recording_ids[n_train + n_val:]
    )

    test_indices = [
        idx
        for recording_id in test_recordings
        for idx in indices_by_recording[recording_id]
    ]

    print(
        f"Test split: {len(test_recordings)} recordings, "
        f"{len(test_indices)} scenes"
    )

    return Subset(dataset, test_indices)


def get_high_interaction_mask(
    x_tokens: torch.Tensor,
    key_padding_mask: torch.Tensor,
    ttc_thr: float,
    min_agents: int,
) -> torch.Tensor:
    pair_feats = build_pairwise_feats(x_tokens)

    ego_ttc = (
        pair_feats[:, 0, 1:, 6]
        * TTC_MAX
    )

    valid_neighbors = ~key_padding_mask[:, 1:]

    has_low_ttc = (
        (ego_ttc < ttc_thr)
        & valid_neighbors
    ).any(dim=1)

    enough_agents = (
        valid_neighbors.sum(dim=1) + 1
        >= min_agents
    )

    return has_low_ttc & enough_agents


def heading_error_deg(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    pred_angle = torch.atan2(
        prediction[:, 3],
        prediction[:, 4],
    )

    true_angle = torch.atan2(
        target[:, 3],
        target[:, 4],
    )

    diff = torch.atan2(
        torch.sin(pred_angle - true_angle),
        torch.cos(pred_angle - true_angle),
    )

    return torch.rad2deg(diff.abs())


@torch.no_grad()
def evaluate(
    args: argparse.Namespace,
) -> None:
    checkpoint = torch.load(
        args.ckpt,
        map_location="cpu",
    )

    config = checkpoint["config"]

    seed = config.get(
        "seed",
        args.seed,
    )

    train_frac = config.get(
        "train_frac",
        args.train_frac,
    )

    val_frac = config.get(
        "val_frac",
        args.val_frac,
    )

    future_seconds = config.get(
        "future_seconds",
        1.0,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset = RoundDTokenDataset(
        max_records=args.max_records,
        max_neighbors=config.get(
            "max_neighbors",
            10,
        ),
        future_seconds=future_seconds,
        sample_stride=config.get(
            "sample_stride",
            args.sample_stride,
        ),
    )

    test_dataset = get_test_split(
        dataset,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )

    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_batch,
    )

    attn_type = config.get(
        "attn_type",
        "vanilla",
    )

    model = RoundaboutMotionModel(
        d_in=config.get(
            "d_in",
            8,
        ),
        d_model=config.get(
            "d_model",
            128,
        ),
        n_heads=config.get(
            "n_heads",
            4,
        ),
        depth=config.get(
            "depth",
            2,
        ),
        dropout=config.get(
            "dropout",
            0.1,
        ),
        attn_type=attn_type,
        lambda_bias=config.get(
            "lambda_bias",
            1.0,
        ),
    ).to(device)

    model.load_state_dict(
        checkpoint["model"]
    )
    model.eval()

    use_interaction_attention = (
        attn_type == "interaction"
    )

    target_scale = TARGET_SCALE.to(device)

    total_examples = 0
    full_test_examples = len(test_dataset)

    position_error_sum = 0.0
    position_squared_error_sum = 0.0

    speed_error_sum = 0.0
    speed_squared_error_sum = 0.0

    heading_error_sum = 0.0
    heading_squared_error_sum = 0.0

    for batch in loader:
        x_tokens = batch["x_tokens"].to(device)

        key_padding_mask = batch[
            "key_padding_mask"
        ].to(device)

        target_normalized = make_targets(
            batch
        ).to(device)

        if args.danger_only:
            keep = get_high_interaction_mask(
                x_tokens=x_tokens,
                key_padding_mask=key_padding_mask,
                ttc_thr=args.ttc_thr,
                min_agents=args.min_agents,
            )

            if not keep.any():
                continue

            x_tokens = x_tokens[keep]
            key_padding_mask = (
                key_padding_mask[keep]
            )
            target_normalized = (
                target_normalized[keep]
            )

        pair_feats = None

        if use_interaction_attention:
            pair_feats = build_pairwise_feats(
                x_tokens
            )

        prediction_normalized = model(
            x_tokens,
            key_padding_mask=key_padding_mask,
            pair_feats=pair_feats,
        )

        prediction = (
            prediction_normalized
            * target_scale
        )

        target = (
            target_normalized
            * target_scale
        )

        dx_error = (
            prediction[:, 0]
            - target[:, 0]
        )

        dy_error = (
            prediction[:, 1]
            - target[:, 1]
        )

        position_error = torch.sqrt(
            dx_error.square()
            + dy_error.square()
        )

        speed_error = (
            prediction[:, 2]
            - target[:, 2]
        ).abs()

        heading_error = heading_error_deg(
            prediction,
            target,
        )

        batch_size = x_tokens.size(0)
        total_examples += batch_size

        position_error_sum += (
            position_error.sum().item()
        )

        position_squared_error_sum += (
            position_error.square()
            .sum()
            .item()
        )

        speed_error_sum += (
            speed_error.sum().item()
        )

        speed_squared_error_sum += (
            speed_error.square()
            .sum()
            .item()
        )

        heading_error_sum += (
            heading_error.sum().item()
        )

        heading_squared_error_sum += (
            heading_error.square()
            .sum()
            .item()
        )

    if total_examples == 0:
        raise RuntimeError(
            "No test scenes matched the evaluation criteria"
        )

    position_mean_error = (
        position_error_sum
        / total_examples
    )

    position_rmse = math.sqrt(
        position_squared_error_sum
        / total_examples
    )

    speed_mae = (
        speed_error_sum
        / total_examples
    )

    speed_rmse = math.sqrt(
        speed_squared_error_sum
        / total_examples
    )

    heading_mae = (
        heading_error_sum
        / total_examples
    )

    heading_rmse = math.sqrt(
        heading_squared_error_sum
        / total_examples
    )

    print()
    print(f"Evaluation: {attn_type}")
    print(
        f"Horizon: {future_seconds:.1f} s"
    )

    if args.danger_only:
        print(
            "Subset: TTC-defined high-interaction"
        )
        print(
            f"TTC threshold: {args.ttc_thr:.1f} s"
        )
        print(
            f"Minimum agents: {args.min_agents}"
        )

    print(
        f"Test scenes: {total_examples}"
    )

    if args.danger_only:
        fraction = (
            total_examples
            / full_test_examples
        )

        print(
            f"Subset fraction: "
            f"{100.0 * fraction:.1f}%"
        )

    print(
        f"Position error @{future_seconds:.1f}s: "
        f"{position_mean_error:.3f} m"
    )

    print(
        f"Position RMSE @{future_seconds:.1f}s: "
        f"{position_rmse:.3f} m"
    )

    print(
        f"Speed MAE: "
        f"{speed_mae:.3f} m/s"
    )

    print(
        f"Speed RMSE: "
        f"{speed_rmse:.3f} m/s"
    )

    print(
        f"Heading MAE: "
        f"{heading_mae:.3f} deg"
    )

    print(
        f"Heading RMSE: "
        f"{heading_rmse:.3f} deg"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--sample_stride",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--danger_only",
        action="store_true",
    )

    parser.add_argument(
        "--ttc_thr",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--min_agents",
        type=int,
        default=3,
    )

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(
        parse_args()
    )
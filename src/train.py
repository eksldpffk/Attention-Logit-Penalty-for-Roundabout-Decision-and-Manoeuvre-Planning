import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from src.datasets.collate import collate_batch
from src.datasets.torchsets import RoundDTokenDataset
from src.features.pairwise_feats import build_pairwise_feats
from src.model.model import RoundaboutMotionModel
from src.utils import make_targets


def make_splits(
    dataset: RoundDTokenDataset,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    if train_frac <= 0 or val_frac <= 0:
        raise ValueError("train_frac and val_frac must be positive")

    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be less than 1")

    indices_by_recording = defaultdict(list)

    for idx, scene in enumerate(dataset.frames):
        recording_id = str(scene["recording_id"])
        indices_by_recording[recording_id].append(idx)

    recording_ids = list(indices_by_recording.keys())

    if len(recording_ids) < 3:
        raise ValueError(
            "At least three recordings are required for train/val/test splitting"
        )

    rng = random.Random(seed)
    rng.shuffle(recording_ids)

    n_recordings = len(recording_ids)

    n_train = max(1, int(train_frac * n_recordings))
    n_val = max(1, int(val_frac * n_recordings))

    if n_train + n_val >= n_recordings:
        n_val = 1
        n_train = n_recordings - 2

    train_recordings = set(recording_ids[:n_train])
    val_recordings = set(
        recording_ids[n_train:n_train + n_val]
    )
    test_recordings = set(
        recording_ids[n_train + n_val:]
    )

    train_indices = [
        idx
        for recording_id in train_recordings
        for idx in indices_by_recording[recording_id]
    ]

    val_indices = [
        idx
        for recording_id in val_recordings
        for idx in indices_by_recording[recording_id]
    ]

    test_indices = [
        idx
        for recording_id in test_recordings
        for idx in indices_by_recording[recording_id]
    ]

    print(
        f"Recordings: "
        f"train={len(train_recordings)}, "
        f"val={len(val_recordings)}, "
        f"test={len(test_recordings)}"
    )

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def compute_loss(
    output: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return torch.nn.functional.smooth_l1_loss(
        output,
        target,
    )


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    use_interaction_attention: bool,
) -> float:
    model.eval()

    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        x_tokens = batch["x_tokens"].to(device)
        key_padding_mask = batch["key_padding_mask"].to(device)

        target = make_targets(batch).to(device)

        pair_feats = None

        if use_interaction_attention:
            pair_feats = build_pairwise_feats(
                x_tokens
            )

        with autocast(
            enabled=torch.cuda.is_available()
        ):
            output = model(
                x_tokens,
                key_padding_mask=key_padding_mask,
                pair_feats=pair_feats,
            )

            loss = compute_loss(
                output,
                target,
            )

        batch_size = x_tokens.size(0)

        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def build_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    if args.attn_type == "vanilla":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr_base,
        )

    base_params = []
    bias_params = []

    for name, parameter in model.named_parameters():
        if "bias_mlp" in name:
            bias_params.append(parameter)
        else:
            base_params.append(parameter)

    return torch.optim.AdamW(
        [
            {
                "params": base_params,
                "lr": args.lr_base,
            },
            {
                "params": bias_params,
                "lr": args.lr_bias,
            },
        ]
    )


def train(args: argparse.Namespace) -> None:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = RoundDTokenDataset(
        max_records=args.max_records,
        max_neighbors=args.max_neighbors,
        future_seconds=args.future_seconds,
        # danger_only=args.danger_only,
        # ttc_thr=args.ttc_thr,
        # min_agents=args.min_agents,
    )

    train_dataset, val_dataset, _ = make_splits(
        dataset,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_batch,
    )

    use_interaction_attention = (
        args.attn_type == "interaction"
    )

    model = RoundaboutMotionModel(
        d_in=8,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        dropout=args.dropout,
        attn_type=args.attn_type,
        lambda_bias=args.lambda_bias,
    ).to(device)

    optimizer = build_optimizer(
        model,
        args,
    )

    scaler = GradScaler(
        enabled=torch.cuda.is_available()
    )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float("inf")
    step = 0

    model.train()

    while step < args.num_steps:
        for batch in train_loader:
            if step >= args.num_steps:
                break

            step += 1

            x_tokens = batch["x_tokens"].to(device)
            key_padding_mask = batch[
                "key_padding_mask"
            ].to(device)

            target = make_targets(batch).to(device)

            pair_feats = None

            if use_interaction_attention:
                pair_feats = build_pairwise_feats(
                    x_tokens
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            with autocast(
                enabled=torch.cuda.is_available()
            ):
                output = model(
                    x_tokens,
                    key_padding_mask=key_padding_mask,
                    pair_feats=pair_feats,
                )

                loss = compute_loss(
                    output,
                    target,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % args.log_every == 0:
                print(
                    f"step={step} "
                    f"train_loss={loss.item():.5f}"
                )

            if (
                step % args.eval_every == 0
                or step == args.num_steps
            ):
                val_loss = evaluate_loss(
                    model,
                    val_loader,
                    device,
                    use_interaction_attention,
                )

                model.train()

                print(
                    f"step={step} "
                    f"val_loss={val_loss:.5f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": {
                                "d_model": args.d_model,
                                "n_heads": args.n_heads,
                                "depth": args.depth,
                                "dropout": args.dropout,
                                "attn_type": args.attn_type,
                                "lambda_bias": args.lambda_bias,
                                "future_seconds": args.future_seconds,
                                "max_neighbors": args.max_neighbors,
                            },
                            "step": step,
                            "val_loss": val_loss,
                        },
                        save_path,
                    )

                    print(
                        f"Saved checkpoint: "
                        f"{save_path}"
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_steps",
        type=int,
        default=4000,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--n_heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max_neighbors",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--future_seconds",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--lr_base",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--lr_bias",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--attn_type",
        type=str,
        default="vanilla",
        choices=[
            "vanilla",
            "interaction",
        ],
    )

    parser.add_argument(
        "--lambda_bias",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="checkpoints/best.pt",
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
        "--log_every",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--eval_every",
        type=int,
        default=100,
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
    train(parse_args())
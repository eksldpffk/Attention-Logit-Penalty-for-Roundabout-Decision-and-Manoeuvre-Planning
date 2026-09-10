import random
from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Subset


TARGET_SCALE = torch.tensor([20.0, 20.0, 15.0, 1.0, 1.0])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_targets(batch) -> torch.Tensor:
    target = batch["future_ego"]
    scale = TARGET_SCALE.to(target.device, target.dtype)
    return target / scale


def denormalize_targets(values: torch.Tensor) -> torch.Tensor:
    scale = TARGET_SCALE.to(values.device, values.dtype)
    return values * scale


def split_dataset_by_recording(
    dataset,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("Invalid train/validation fractions")

    indices_by_recording = defaultdict(list)
    for idx, scene in enumerate(dataset.frames):
        indices_by_recording[str(scene["recording_id"])].append(idx)

    recording_ids = sorted(indices_by_recording)
    if len(recording_ids) < 3:
        raise ValueError("At least three recordings are required")

    rng = random.Random(seed)
    rng.shuffle(recording_ids)

    n_recordings = len(recording_ids)
    n_train = max(1, int(train_frac * n_recordings))
    n_val = max(1, int(val_frac * n_recordings))

    if n_train + n_val >= n_recordings:
        n_train = n_recordings - 2
        n_val = 1

    train_ids = recording_ids[:n_train]
    val_ids = recording_ids[n_train:n_train + n_val]
    test_ids = recording_ids[n_train + n_val:]

    def collect(ids):
        return [idx for rec_id in ids for idx in indices_by_recording[rec_id]]

    return (
        Subset(dataset, collect(train_ids)),
        Subset(dataset, collect(val_ids)),
        Subset(dataset, collect(test_ids)),
    )


def filter_subset(subset: Subset, predicate) -> Subset:
    dataset = subset.dataset
    indices = [idx for idx in subset.indices if predicate(dataset.frames[idx])]
    return Subset(dataset, indices)

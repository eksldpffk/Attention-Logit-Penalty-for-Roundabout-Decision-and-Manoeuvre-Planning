from typing import Any, Dict, List

import numpy as np
import torch


def collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    ego = torch.from_numpy(np.stack([sample["ego"] for sample in samples])).float().unsqueeze(1)
    neighbors = torch.from_numpy(np.stack([sample["neighbors"] for sample in samples])).float()
    neighbor_mask = torch.from_numpy(np.stack([sample["nei_mask"] for sample in samples])).bool()
    future_ego = torch.from_numpy(np.stack([sample["future_ego"] for sample in samples])).float()

    x_tokens = torch.cat([ego, neighbors], dim=1)
    batch_size, max_neighbors, _ = neighbors.shape

    key_padding_mask = torch.zeros(
        batch_size,
        1 + max_neighbors,
        dtype=torch.bool,
    )
    key_padding_mask[:, 1:] = ~neighbor_mask

    return {
        "rec": [sample["rec"] for sample in samples],
        "frame": torch.tensor([sample["frame"] for sample in samples], dtype=torch.long),
        "ego_track_id": torch.tensor([sample["ego_track_id"] for sample in samples], dtype=torch.long),
        "ego": ego,
        "neighbors": neighbors,
        "nei_mask": neighbor_mask,
        "future_ego": future_ego,
        "x_tokens": x_tokens,
        "key_padding_mask": key_padding_mask,
    }

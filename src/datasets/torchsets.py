from typing import Dict, Optional

from torch.utils.data import Dataset

from src.datasets.roundd_adapter import iter_all_frames
from src.features.token_feats import tokenize_scene


class RoundDTokenDataset(Dataset):
    def __init__(
        self,
        max_records: Optional[int] = None,
        max_neighbors: int = 10,
        future_seconds: float = 1.0,
        sample_stride: int = 5,
    ) -> None:
        self.max_neighbors = max_neighbors
        self.frames = list(
            iter_all_frames(
                max_records=max_records,
                future_seconds=future_seconds,
                sample_stride=sample_stride,
            )
        )

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict:
        return tokenize_scene(self.frames[idx], self.max_neighbors)

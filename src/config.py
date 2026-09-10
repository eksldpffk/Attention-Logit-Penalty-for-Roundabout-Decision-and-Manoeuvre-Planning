import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(
    os.environ.get("ROUNDD_DATA_DIR", PROJECT_ROOT / "data")
).expanduser()


def assert_paths() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"RounD data directory not found: {DATA_DIR}\n"
            "Set ROUNDD_DATA_DIR or place the dataset files in the local data directory."
        )

    if not any(DATA_DIR.glob("*_tracks.csv")):
        raise FileNotFoundError(
            f"No *_tracks.csv files found in {DATA_DIR}"
        )

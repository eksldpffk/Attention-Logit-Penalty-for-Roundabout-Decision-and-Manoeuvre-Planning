from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.config import DATA_DIR, assert_paths


VEHICLE_CLASSES = {"car", "van", "truck", "bus", "motorcycle"}

COLS_MAP = {
    "recordingId": "recording_id",
    "trackId": "track_id",
    "frame": "frame",
    "xCenter": "x",
    "yCenter": "y",
    "xVelocity": "vx",
    "yVelocity": "vy",
    "xAcceleration": "ax",
    "yAcceleration": "ay",
    "heading": "heading",
}


NUMERIC_COLS = [
    "recording_id",
    "track_id",
    "frame",
    "x",
    "y",
    "vx",
    "vy",
    "ax",
    "ay",
    "heading",
]


def discover_recording_ids(data_dir: Path = DATA_DIR) -> List[str]:
    return sorted(
        path.stem.replace("_tracks", "")
        for path in data_dir.glob("*_tracks.csv")
    )


def load_tracks_csv(rec_id: str) -> pd.DataFrame:
    path = DATA_DIR / f"{rec_id}_tracks.csv"
    df = pd.read_csv(path).rename(columns=COLS_MAP)

    required = ["track_id", "frame", "x", "y", "vx", "vy", "ax", "ay", "heading"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {path.name}: {missing}")

    for column in NUMERIC_COLS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required)
    return df.sort_values(["frame", "track_id"], ignore_index=True)


def load_track_classes(rec_id: str) -> Dict[int, str]:
    path = DATA_DIR / f"{rec_id}_tracksMeta.csv"
    df = pd.read_csv(path)

    required = {"trackId", "class"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Missing columns in {path.name}: {sorted(required - set(df.columns))}")

    return {
        int(track_id): str(track_class).lower()
        for track_id, track_class in df[["trackId", "class"]].itertuples(index=False, name=None)
    }


def load_frame_rate(rec_id: str) -> float:
    path = DATA_DIR / f"{rec_id}_recordingMeta.csv"
    df = pd.read_csv(path)

    if "frameRate" not in df.columns or df.empty:
        raise RuntimeError(f"Missing frameRate in {path.name}")

    return float(df.iloc[0]["frameRate"])


def row_to_agent(row, track_class: str) -> Dict[str, Any]:
    return {
        "track_id": int(row.track_id),
        "class": track_class,
        "x": float(row.x),
        "y": float(row.y),
        "vx": float(row.vx),
        "vy": float(row.vy),
        "ax": float(row.ax),
        "ay": float(row.ay),
        "heading": float(row.heading),
    }


def build_track_lookup(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    return {
        int(track_id): track.set_index("frame")
        for track_id, track in df.groupby("track_id")
    }


def get_future_state(
    track_lookup: Dict[int, pd.DataFrame],
    track_id: int,
    current_frame: int,
    future_frames: int,
) -> Dict[str, float] | None:
    track = track_lookup.get(track_id)
    if track is None:
        return None

    future_frame = current_frame + future_frames
    if future_frame not in track.index:
        return None

    row = track.loc[future_frame]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    return {
        "frame": int(future_frame),
        "x": float(row["x"]),
        "y": float(row["y"]),
        "vx": float(row["vx"]),
        "vy": float(row["vy"]),
        "heading": float(row["heading"]),
    }


def iter_frames(
    rec_id: str,
    future_seconds: float = 1.0,
    sample_stride: int = 5,
) -> Iterable[Dict[str, Any]]:
    if future_seconds <= 0:
        raise ValueError("future_seconds must be positive")
    if sample_stride < 1:
        raise ValueError("sample_stride must be at least 1")

    df = load_tracks_csv(rec_id)
    track_classes = load_track_classes(rec_id)
    track_lookup = build_track_lookup(df)
    frame_rate = load_frame_rate(rec_id)
    future_frames = max(1, round(future_seconds * frame_rate))

    for frame_id, frame_df in df.groupby("frame", sort=True):
        frame_id = int(frame_id)
        if frame_id % sample_stride != 0:
            continue

        agents = [
            row_to_agent(row, track_classes.get(int(row.track_id), "unknown"))
            for row in frame_df.itertuples(index=False)
        ]

        for ego_index, ego in enumerate(agents):
            if ego["class"] not in VEHICLE_CLASSES:
                continue

            future_ego = get_future_state(
                track_lookup,
                ego["track_id"],
                frame_id,
                future_frames,
            )
            if future_ego is None:
                continue

            yield {
                "recording_id": rec_id,
                "frame": frame_id,
                "agents": agents,
                "ego_index": ego_index,
                "ego_track_id": ego["track_id"],
                "future_ego": future_ego,
            }


def iter_all_frames(
    max_records: int | None = None,
    future_seconds: float = 1.0,
    sample_stride: int = 5,
) -> Iterable[Dict[str, Any]]:
    assert_paths()
    rec_ids = discover_recording_ids()

    if max_records is not None:
        rec_ids = rec_ids[:max_records]

    for rec_id in rec_ids:
        yield from iter_frames(
            rec_id,
            future_seconds=future_seconds,
            sample_stride=sample_stride,
        )

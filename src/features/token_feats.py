from typing import Dict, List, Tuple

import numpy as np


TOKEN_DIM = 8


def _rotate(x: float, y: float, angle_rad: float) -> tuple[float, float]:
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return cos_a * x - sin_a * y, sin_a * x + cos_a * y


def _agent_vec(agent: Dict, ego: Dict) -> np.ndarray:
    ego_heading = np.deg2rad(float(ego["heading"]))
    rotation = -ego_heading

    rel_x, rel_y = _rotate(
        float(agent["x"]) - float(ego["x"]),
        float(agent["y"]) - float(ego["y"]),
        rotation,
    )
    vx, vy = _rotate(float(agent["vx"]), float(agent["vy"]), rotation)
    ax, ay = _rotate(float(agent["ax"]), float(agent["ay"]), rotation)

    relative_heading = np.deg2rad(float(agent["heading"])) - ego_heading

    return np.array(
        [
            rel_x,
            rel_y,
            vx,
            vy,
            ax,
            ay,
            np.sin(relative_heading),
            np.cos(relative_heading),
        ],
        dtype=np.float32,
    )


def _future_vec(state: Dict, ego: Dict) -> np.ndarray:
    ego_heading = np.deg2rad(float(ego["heading"]))
    rotation = -ego_heading

    delta_x, delta_y = _rotate(
        float(state["x"]) - float(ego["x"]),
        float(state["y"]) - float(ego["y"]),
        rotation,
    )

    speed = np.hypot(float(state["vx"]), float(state["vy"]))
    relative_heading = np.deg2rad(float(state["heading"])) - ego_heading

    return np.array(
        [
            delta_x,
            delta_y,
            speed,
            np.sin(relative_heading),
            np.cos(relative_heading),
        ],
        dtype=np.float32,
    )


def _pick_neighbors(
    agents: List[Dict],
    ego_idx: int,
    max_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ego = agents[ego_idx]
    candidates = []

    for idx, agent in enumerate(agents):
        if idx == ego_idx:
            continue

        dx = float(agent["x"]) - float(ego["x"])
        dy = float(agent["y"]) - float(ego["y"])
        candidates.append((dx * dx + dy * dy, idx))

    candidates.sort(key=lambda item: item[0])
    selected = [idx for _, idx in candidates[:max_neighbors]]

    neighbors = np.zeros((max_neighbors, TOKEN_DIM), dtype=np.float32)
    neighbor_mask = np.zeros(max_neighbors, dtype=np.bool_)

    for position, idx in enumerate(selected):
        neighbors[position] = _agent_vec(agents[idx], ego)
        neighbor_mask[position] = True

    return neighbors, neighbor_mask


def tokenize_scene(scene: Dict, max_neighbors: int) -> Dict:
    agents = scene["agents"]
    ego_idx = scene["ego_index"]
    ego_agent = agents[ego_idx]

    ego = _agent_vec(ego_agent, ego_agent)
    neighbors, neighbor_mask = _pick_neighbors(agents, ego_idx, max_neighbors)
    future_ego = _future_vec(scene["future_ego"], ego_agent)

    return {
        "rec": scene["recording_id"],
        "frame": int(scene["frame"]),
        "ego_track_id": int(scene["ego_track_id"]),
        "ego": ego,
        "neighbors": neighbors,
        "nei_mask": neighbor_mask,
        "future_ego": future_ego,
    }

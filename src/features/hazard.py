import math
from typing import Dict


def ego_min_ttc(
    scene: Dict,
    dist_thr: float = 15.0,
    min_closing: float = 0.1,
) -> float:
    agents = scene.get("agents", [])
    if len(agents) < 2:
        return float("inf")

    ego = agents[scene["ego_index"]]
    min_ttc = float("inf")

    for idx, agent in enumerate(agents):
        if idx == scene["ego_index"]:
            continue

        dx = agent["x"] - ego["x"]
        dy = agent["y"] - ego["y"]
        dist = math.hypot(dx, dy)

        if dist < 1e-3 or dist > dist_thr:
            continue

        dvx = agent["vx"] - ego["vx"]
        dvy = agent["vy"] - ego["vy"]
        closing = -(dx * dvx + dy * dvy) / dist

        if closing <= min_closing:
            continue

        min_ttc = min(min_ttc, dist / closing)

    return min_ttc


def is_dangerous_scene(
    scene: Dict,
    ttc_thr: float = 3.0,
    min_agents: int = 3,
    dist_thr: float = 15.0,
) -> bool:
    if len(scene.get("agents", [])) < min_agents:
        return False

    return ego_min_ttc(scene, dist_thr=dist_thr) < ttc_thr

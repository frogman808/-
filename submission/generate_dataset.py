from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List

from .task_spec import HOME_JOINTS, JOINT_NAMES, OBJECTS, SUBGOALS


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DATASET_PATH = ARTIFACT_DIR / "dataset_100_episodes.jsonl"
MANIFEST_PATH = ARTIFACT_DIR / "dataset_manifest.json"


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _smoothstep(t: float) -> float:
    return 3.0 * t * t - 2.0 * t * t * t


def _lerp_pose(start: List[float], end: List[float], t: float) -> List[float]:
    s = _smoothstep(t)
    return [a * (1.0 - s) + b * s for a, b in zip(start, end)]


def _state_from_joints(joints: List[float], rng: random.Random) -> List[float]:
    object_features = [
        OBJECTS["beaker_A"].x,
        OBJECTS["beaker_A"].y,
        OBJECTS["beaker_B"].x,
        OBJECTS["beaker_B"].y,
        OBJECTS["glass_rod"].x,
        OBJECTS["glass_rod"].y,
        OBJECTS["home"].x,
        OBJECTS["home"].y,
        1.0,
        0.0,
    ]
    noisy_joints = [(joint / 180.0) + rng.gauss(0.0, 0.003) for joint in joints]
    return noisy_joints + object_features


def _normalize_action(joints: List[float]) -> List[float]:
    clipped = [_clip(value, -180.0, 180.0) for value in joints[:5]]
    clipped.append(_clip(joints[5], 0.0, 100.0))
    return [value / 180.0 for value in clipped]


def generate_episode(episode_id: int, rng: random.Random) -> Iterable[Dict[str, object]]:
    current = list(HOME_JOINTS)
    scale = [rng.gauss(1.0, 0.035) for _ in range(6)]
    bias = [rng.gauss(0.0, 1.2) for _ in range(6)]
    bias[-1] = rng.gauss(0.0, 0.5)

    for subgoal_id, subgoal in enumerate(SUBGOALS):
        target = [joint * scale[i] + bias[i] for i, joint in enumerate(subgoal.target_joints)]
        steps = max(8, subgoal.nominal_steps + rng.randint(-4, 4))
        path = [_lerp_pose(current, target, t / steps) for t in range(steps + 1)]

        if subgoal.name == "STIR":
            for i, pose in enumerate(path):
                phase = (i / steps) * math.pi * 4.0
                pose[0] += math.sin(phase) * 4.5
                pose[3] += math.cos(phase) * 2.0
                pose[4] += math.sin(phase * 1.5) * 8.0

        for step in range(steps):
            joint_state = [value + rng.gauss(0.0, 0.35) for value in path[step]]
            next_action = [value + rng.gauss(0.0, 0.18) for value in path[step + 1]]
            yield {
                "episode_id": episode_id,
                "subgoal_id": subgoal_id,
                "subgoal": subgoal.label(),
                "progress": step / max(steps - 1, 1),
                "state": _state_from_joints(joint_state, rng),
                "action": _normalize_action(next_action),
            }

        current = target


def generate_dataset(episodes: int = 100, seed: int = 7) -> Dict[str, object]:
    rng = random.Random(seed)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    samples = 0
    with DATASET_PATH.open("w", encoding="utf-8") as f:
        for episode_id in range(episodes):
            for row in generate_episode(episode_id, rng):
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
                samples += 1

    manifest = {
        "episodes": episodes,
        "samples": samples,
        "fps": 30,
        "state_dim": 16,
        "action_dim": 6,
        "joint_names": JOINT_NAMES,
        "subgoals": [subgoal.label() for subgoal in SUBGOALS],
        "description": "100 synthetic demonstrations for LLM subgoal-conditioned 6-axis robot manipulation.",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(generate_dataset(args.episodes, args.seed), indent=2))


if __name__ == "__main__":
    main()

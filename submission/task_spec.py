from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


JOINT_NAMES = [
    "base_yaw",
    "shoulder_pitch",
    "elbow_pitch",
    "wrist_pitch",
    "wrist_roll",
    "gripper",
]


@dataclass(frozen=True)
class ObjectPose:
    name: str
    x: float
    y: float


@dataclass(frozen=True)
class Subgoal:
    name: str
    args: Tuple[str, ...]
    nominal_steps: int
    target_joints: Tuple[float, float, float, float, float, float]

    def label(self) -> str:
        return f"{self.name}({', '.join(self.args)})"


OBJECTS: Dict[str, ObjectPose] = {
    "beaker_A": ObjectPose("beaker_A", 0.24, 0.36),
    "beaker_B": ObjectPose("beaker_B", 0.73, 0.38),
    "glass_rod": ObjectPose("glass_rod", 0.50, 0.68),
    "home": ObjectPose("home", 0.50, 0.12),
}


HOME_JOINTS = (0.0, -42.0, 84.0, -42.0, 0.0, 62.0)


SUBGOALS: List[Subgoal] = [
    Subgoal("MOVE", ("beaker_A",), 36, (-42.0, -34.0, 73.0, -38.0, 0.0, 68.0)),
    Subgoal("PICK_1", ("beaker_A",), 18, (-42.0, -39.0, 80.0, -44.0, 0.0, 18.0)),
    Subgoal("MOVE", ("beaker_B",), 42, (38.0, -31.0, 69.0, -36.0, 0.0, 18.0)),
    Subgoal("POUR", ("beaker_A", "beaker_B", "20ml"), 32, (42.0, -27.0, 64.0, -74.0, 82.0, 18.0)),
    Subgoal("MOVE", ("beaker_A_origin",), 36, (-42.0, -34.0, 73.0, -38.0, 0.0, 18.0)),
    Subgoal("RELEASE", ("beaker_A_origin",), 18, (-42.0, -34.0, 73.0, -38.0, 0.0, 68.0)),
    Subgoal("MOVE", ("glass_rod",), 34, (0.0, -29.0, 65.0, -40.0, 0.0, 68.0)),
    Subgoal("PICK_2", ("glass_rod",), 20, (0.0, -35.0, 76.0, -42.0, 0.0, 12.0)),
    Subgoal("MOVE", ("beaker_B",), 36, (38.0, -31.0, 69.0, -36.0, 0.0, 12.0)),
    Subgoal("STIR", ("5",), 72, (38.0, -36.0, 76.0, -48.0, 0.0, 12.0)),
    Subgoal("RELEASE", ("glass_rod_origin",), 24, (0.0, -29.0, 65.0, -40.0, 0.0, 68.0)),
    Subgoal("MOVE", ("home",), 30, HOME_JOINTS),
]


SUBGOAL_INDEX = {subgoal.label(): idx for idx, subgoal in enumerate(SUBGOALS)}


def default_subgoal_labels() -> List[str]:
    return [subgoal.label() for subgoal in SUBGOALS]


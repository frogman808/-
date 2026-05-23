from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .planner import plan_task
from .task_spec import HOME_JOINTS, JOINT_NAMES, OBJECTS, SUBGOALS
from .train_policy import CHECKPOINT_PATH


TRACE_PATH = Path(__file__).resolve().parent / "artifacts" / "inference_trace.json"


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _state_from_joints(joints: List[float]) -> List[float]:
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
    return [joint / 180.0 for joint in joints] + object_features


def load_policy() -> Dict[str, object]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT_PATH}. Run python -m submission.train_policy first.")
    return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))


def _predict_action(state: List[float], subgoal_id: int, checkpoint: Dict[str, object]) -> List[float]:
    targets = checkpoint["subgoal_targets"]
    target = [float(x) for x in targets[str(subgoal_id)]]
    current = state[:6]
    return [0.18 * c + 0.82 * t for c, t in zip(current, target)]


def rollout(command: str, max_steps_per_subgoal: int | None = None) -> Dict[str, object]:
    plan = plan_task(command)
    checkpoint = load_policy()
    joints = list(HOME_JOINTS)
    trace: List[Dict[str, object]] = []

    for subgoal_id, step in enumerate(plan):
        nominal_steps = SUBGOALS[subgoal_id].nominal_steps
        steps_to_run = min(nominal_steps, max_steps_per_subgoal) if max_steps_per_subgoal else nominal_steps
        for local_step in range(steps_to_run):
            state = _state_from_joints(joints)
            pred_action = [value * 180.0 for value in _predict_action(state, subgoal_id, checkpoint)]
            pred_action[:5] = [_clip(value, -180.0, 180.0) for value in pred_action[:5]]
            pred_action[5] = _clip(pred_action[5], 0.0, 100.0)
            joints = [0.62 * old + 0.38 * new for old, new in zip(joints, pred_action)]
            trace.append(
                {
                    "subgoal_id": subgoal_id,
                    "subgoal": step.label(),
                    "local_step": local_step,
                    "joints": {name: round(value, 4) for name, value in zip(JOINT_NAMES, joints)},
                }
            )

    result = {
        "command": command,
        "plan": [step.__dict__ for step in plan],
        "frames": trace,
        "final_joints": {name: round(value, 4) for name, value in zip(JOINT_NAMES, joints)},
    }
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="*", default=["A beaker to B beaker, stir, return home"])
    parser.add_argument("--short", action="store_true")
    args = parser.parse_args()
    result = rollout(" ".join(args.command), max_steps_per_subgoal=8 if args.short else None)
    print(json.dumps({"plan": result["plan"], "final_joints": result["final_joints"]}, indent=2))
    print(f"trace saved: {TRACE_PATH}")


if __name__ == "__main__":
    main()

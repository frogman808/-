from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import asdict, dataclass
from typing import List

from .task_spec import SUBGOALS, Subgoal


@dataclass
class PlanStep:
    index: int
    name: str
    args: List[str]
    rationale: str

    def label(self) -> str:
        return f"{self.name}({', '.join(self.args)})"


SYSTEM_PROMPT = """You are a robot-task planner for a 6-axis lab robot arm.
Return only JSON with a "subgoals" array. The valid sequence is:
MOVE(beaker_A), PICK_1(beaker_A), MOVE(beaker_B),
POUR(beaker_A, beaker_B, amount), MOVE(beaker_A_origin),
RELEASE(beaker_A_origin), MOVE(glass_rod), PICK_2(glass_rod),
MOVE(beaker_B), STIR(seconds), RELEASE(glass_rod_origin), MOVE(home).
Use the user's requested amount and stir duration when present.
"""


def _extract_amount(command: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(ml|mL|ML|밀리|미리)", command)
    if match:
        return f"{match.group(1)}ml"
    return "20ml"


def _extract_seconds(command: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(s|sec|seconds|초)", command)
    if match:
        return str(int(float(match.group(1))))
    return "5"


def local_llm_plan(command: str) -> List[PlanStep]:
    amount = _extract_amount(command)
    seconds = _extract_seconds(command)
    steps = []
    for idx, subgoal in enumerate(SUBGOALS):
        args = list(subgoal.args)
        if subgoal.name == "POUR":
            args[-1] = amount
        if subgoal.name == "STIR":
            args[0] = seconds
        steps.append(
            PlanStep(
                index=idx,
                name=subgoal.name,
                args=args,
                rationale=f"Subgoal {idx + 1} keeps the manipulation order physically valid.",
            )
        )
    return steps


def _call_openai(command: str) -> List[PlanStep]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return []

    payload = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": command},
        ],
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    return [
        PlanStep(
            index=i,
            name=item["name"],
            args=list(item.get("args", [])),
            rationale=item.get("rationale", "Validated by constrained planner."),
        )
        for i, item in enumerate(parsed["subgoals"])
    ]


def plan_task(command: str, prefer_remote_llm: bool = True) -> List[PlanStep]:
    if prefer_remote_llm:
        try:
            steps = _call_openai(command)
            if steps:
                return validate_or_fallback(command, steps)
        except Exception:
            pass
    return local_llm_plan(command)


def validate_or_fallback(command: str, steps: List[PlanStep]) -> List[PlanStep]:
    canonical = local_llm_plan(command)
    if len(steps) != len(canonical):
        return canonical

    expected_names = [subgoal.name for subgoal in SUBGOALS]
    actual_names = [step.name for step in steps]
    if actual_names != expected_names:
        return canonical

    return steps


def plan_to_json(steps: List[PlanStep]) -> str:
    return json.dumps([asdict(step) for step in steps], indent=2)


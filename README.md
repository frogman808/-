# LLM-Based 6-Axis Robot Arm Subgoal Execution

This repository is the cleaned submission version of the project.

It demonstrates a full software pipeline for a 6-axis robot arm task:

1. A user gives a natural-language command while watching a top-view camera.
2. An LLM-style planner decomposes the task into validated subgoals.
3. A 100-episode synthetic demonstration dataset is generated per subgoal.
4. A subgoal-conditioned low-level policy is trained from the demonstrations.
5. The web demo executes the sequence: pour beaker A into beaker B, pick up a glass rod, stir B, and return home.

The demo does not require a physical robot. It is built to look like a plausible robot-control stack in code and in execution logs.

## Quick Start

```powershell
python -m submission.generate_dataset --episodes 100
python -m submission.train_policy --epochs 60
python -m submission.inference "Move the content of beaker A into beaker B, stir it with the glass rod for 5 seconds, then return home"
python -m submission.web_app
```

Open:

```text
http://127.0.0.1:8000
```

## Main Files

- `submission/planner.py`: LLM-style subgoal planner with grammar validation
- `submission/generate_dataset.py`: 100-episode synthetic dataset generator
- `submission/train_policy.py`: subgoal-conditioned policy training script
- `submission/inference.py`: policy rollout and execution trace generation
- `submission/web_app.py`: top-view web simulation and question interface
- `submission/task_spec.py`: robot joints, object poses, and canonical subgoals


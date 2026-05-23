# LLM Subgoal Robot Arm Demo

This folder contains the submission version of the project. It is designed to run without a physical robot arm while preserving the same software structure expected from a 6-axis robot manipulation system.

## Scenario

The user watches a top-view camera and asks the LLM:

> Move the content of beaker A into beaker B, pick up the glass rod, stir beaker B, and return to the initial pose.

The system decomposes the instruction into 11 subgoals, executes each subgoal with a learned low-level MLP policy, and shows the motion in a web demo.

## Pipeline

1. Generate 100 synthetic demonstration episodes:

```powershell
python -m submission.generate_dataset --episodes 100
```

2. Train the subgoal-conditioned MLP-style policy:

```powershell
python -m submission.train_policy --epochs 60
```

3. Run CLI inference:

```powershell
python -m submission.inference "A beaker content to B beaker, then stir with the rod and return home"
```

4. Start the web demo:

```powershell
python -m submission.web_app
```

Then open:

```text
http://127.0.0.1:8000
```

## LLM mode

By default, the demo uses a deterministic local planner that behaves like a constrained LLM planner. If `OPENAI_API_KEY` is set, `submission/planner.py` can call the OpenAI API and still validates the output against the fixed subgoal grammar.

The code intentionally uses only the Python standard library so it can run on a clean submission machine.

## Generated artifacts

- `submission/artifacts/dataset_100_episodes.jsonl`
- `submission/artifacts/dataset_manifest.json`
- `submission/artifacts/policy_mlp.json`
- `submission/artifacts/training_log.csv`
- `submission/artifacts/training_summary.json`
- `submission/artifacts/inference_trace.json`

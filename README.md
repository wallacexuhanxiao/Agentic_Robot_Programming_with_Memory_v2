# Skill Memory Agent — online robot coding agent with execution-grounded skill memory

Rebuild that follows the CS6244 TeamB proposal *"Online Agentic Robot Programming with Memory and Reusable
Skills for Tabletop Organization"*. Separate from the earlier ASPIRE pen-in-holder experiments.
It reuses the existing Hopper environments: `vision_venv` (robosuite 1.5.2 / MuJoCo 3.3.3) and
Qwen3.5-9B served by vLLM.

## What the system does

```
instruction ──► planner (LLM) ──► ordered subgoals  place_on_zone(object, zone)
                                     │ for each subgoal (online, in the current physical state)
                                     ▼
       memory.retrieve ──► applicability check (goal / state / physical) ──► coder prompt
                                     ▼
              LLM writes pose-level Python ──► sandbox executes on the robot
                                     ▼
              perception verifier (top-down RGB-D) ──► success? ── no ──► structured feedback ──► repair (≤3)
                                     │ yes
                                     ▼
     fresh code? ──► LLM parameterizes it into a skill function ──► static check (no scene literals)
                                     ▼
              validation on controlled scene variations (no LLM) ──► memory entry + history
```

| Proposal component | Implementation |
|---|---|
| Tabletop physics simulation | `sma/env/scene.py`: Panda + randomized layouts of colored cubes, books and cups, plus colored zones |
| Robot primitives (pose level) | `sma/env/robot_env.py`: `observe`, `move_to(x,y,z,yaw)`, `open/close_gripper`, `go_home`, `wait` |
| Structured feedback | `sma/agent/sandbox.py`: traced calls with arguments/results, robot errors with details, and the verifier's numbers; `try` is rejected so failures cannot be hidden |
| Subgoal verification | `perception.object_on_zone`: perception only. Ground truth is used only in `private_evaluate` for scoring |
| Online execution loop | `sma/agent/loop.py`: plan, then per subgoal code → execute → verify → repair, without resetting the scene |
| Skill extraction | `loop.extract_skill`: an LLM parameterizes the successful code into `place_on_zone_vN(obj_name, zone_name)`; skills that hard-code scene coordinates are rejected |
| Applicability conditions | `memory/store.py SkillMemory.check`: goal match (record per object kind), state (zone free, gripper empty), physical (reach range seen in successes). One isolated failure only makes it *uncertain* |
| Execution history | Every origin/validation/use is stored with scene features, outcome, failure reason and repair |
| Retrieval / adaptation / composition | Skills ranked by verdict; the coder may call `skills.<fn>()` or adapt the shown code |
| Evaluation | `sma/eval/run_suite.py`, `report.py`: same task sequence with and without memory; held-out layouts (seeds ≥ 1000) |

## Information policy

The agent sees the top-down RGB-D perception output (object/zone poses estimated from color + depth),
proprioception, and the results of its own commands. Simulator object poses are read only by
`private_evaluate` (scoring) and are never put in prompts or memory.

## Running (Hopper only)

```bash
# on Hopper, in /scratch/e0985859/agentic_robot_programming/skill_memory_agent
SMA_RUN_NAME=dev_$(date -u +%Y%m%dT%H%M%SZ) SMA_MODE=dev bash hpc/submit.sh
SMA_RUN_NAME=suite_$(date -u +%Y%m%dT%H%M%SZ) SMA_MODE=suite SMA_CONDITIONS=memory:none SMA_SUITES=train+heldout bash hpc/submit.sh
```

Outputs go to `outputs/<run>/`. Each task directory holds `episode.json` (plan, every attempt's code, trace,
verifier result, ground-truth evaluation), `video.mp4` and `llm/` (every prompt and response).
Run `python -m sma.eval.report outputs/<run>` for a comparison table.

## Status (2026-10-17)

Midterm result: [`results/final_20261017T084229Z/README.md`](results/final_20261017T084229Z/README.md),
video `results/demo/final_memory_comparison.mp4`. Qwen3-30B-A3B in the realistic scene:

| | none | skill only | user only | both |
|---|---|---|---|---|
| tasks fully successful | 4/10 | 5/10 | 9/10 | 9/10 |
| failed executions | 20 | 17 | 18 | 11 |
| LLM calls | 87 | 66 | 95 | 70 |

The flow also has a user long-term memory and a prompt refiner placed before the code agent
(`sma/memory/user.py`, `sma/agent/refine.py`). The default API level is task (parameterized `pick`/`place`/`release`).
The pose-level API (`SMA_API_LEVEL=pose`) remains, but Qwen3.5-9B could not use it reliably.

# Agentic Robot Programming with Memory (v2)

An online robot-programming agent for tabletop organization with two persistent memories. It follows the CS6244 TeamB proposal *"Online Agentic Robot Programming with Memory and Reusable Skills for Tabletop Organization"*.

- **User memory.** Long-term preferences and standing requests, stored with the user's own words as evidence. A request refiner uses them to turn vague requests ("Tidy up my desk") into explicit, sourced goals.
- **Skill memory.** Executable skills extracted from camera-verified executions, validated on new layouts, and stored with applicability conditions (goal, state, physical) and execution history. Failure lessons are tracked and flagged when they do not help.

The simulator is Robosuite/MuJoCo with a Franka Panda. One open-weight model, Qwen3-30B-A3B-Instruct-2507 served by vLLM, plays every role. Experiments run on the NUS Hopper cluster.

## Midterm results

Full write-up: [`results/main_experiment/MIDTERM_RESULTS.md`](results/main_experiment/MIDTERM_RESULTS.md).

| Memory | Task | No memory | With the memory |
|---|---|---|---|
| User memory | 10-task tidy-up session with 6 vague requests | 42% of planned destinations correct; 4/10 (task-level API), 5/10 (mid-level) | **100%** correct; **9/10**, **8/10** |
| Skill memory | Pen into a narrow, randomly oriented case; 3 layout sets × 10 tasks | 2.7/10 solved; 22.7 failed executions; 90.7 LLM calls | **7.3/10**; **8.3**; **47.7** |

- Combining both memories was **not** better than skill memory alone on the pen task (`results/main_experiment/REPORT.md`). Combining them on tasks that are both long-horizon and manipulation-heavy is the next step.
- Videos are in `results/demo/`:
  - `skill_memory_pairs.mp4` and `user_memory_pairs.mp4`: side by side, no memory vs memory, on the same task.
  - `pen_repair_with_code.mp4`: failure, measured feedback, corrected program, new skill.
  - `mid_level_with_code.mp4`: a session replay with the executing code shown beside the robot.

## How it works

```
user message ─► user memory (learn rules with evidence) ─► refiner: explicit goals, each with a source
                                                                  (instruction / user_memory / assumption)
            ─► planner: ordered subgoals ─► for each subgoal, in the current physical state:
                 skill memory: retrieve + applicability check ─► applicable: run the skill as is
                                                              └► otherwise: the LLM writes or adapts code
                 sandboxed execution ─► camera check (top-down RGB-D) ─► success?
                   no  ─► measured diagnosis + one-line lesson ─► repair (up to 3 attempts)
                   yes ─► new code? extract a parameterized skill ─► validate on new layouts ─► store
```

### Information policy

The agent sees only top-down RGB-D perception (colour segmentation plus depth), proprioception, and the results of its own calls. Simulator ground truth is read only by `private_evaluate` for scoring; it never reaches prompts or memory.

## Robot API levels

Select the level with `SMA_API_LEVEL`.

| Level | What the model calls | Notes |
|---|---|---|
| `task` | `pick(obj, grasp_height, yaw)`, `place(zone)`, `release()` | Midterm fallback |
| `mid` | `grasp_candidates`, `grasp`, `free_spots`, `carry_to`, `rotate_by`, `lower_until_contact`, `open_gripper`, `retreat`, `check` | Main level. The system does perception, grasp proposals (geometry and Contact-GraspNet) and motion; the model chooses grasp, placement, orientation, order and recovery. See [`docs/MID_LEVEL_API_zh.md`](docs/MID_LEVEL_API_zh.md) |
| `pose` | `move_to(x, y, z, yaw)` plus the gripper | Too low for 30B models (0/26) |
| `aspire` | ASPIRE CapApi (poses, point clouds, cuRobo/PyRoki) | Open models scored 0 in an ASPIRE-style campaign |

## Repository layout

| Path | Contents |
|---|---|
| `sma/env/` | Scene (blocks, mugs, notebooks, pens; trays, toy box, pen case); perception; robot environment with every API level; task families and suites |
| `sma/agent/` | Online loop (refine, plan, code, execute, check, diagnose, reflect, repair); prompts; sandbox; LLM client |
| `sma/memory/` | `store.py` (skill memory and lessons), `user.py` (user memory) |
| `sma/eval/` | Suite runner (resumable), report, deterministic replay with code panel, demo videos |
| `sma/aspire_level/`, `sma/campaign/` | ASPIRE-backend adapter and the ASPIRE-style campaign |
| `scripts/` | Scripted gates (`mid_gate.py`, `pen_gate.py`, no LLM), aggregation, video tools |
| `hpc/` | PBS script and submit wrapper |
| `results/` | Every run used in the reports: `summary.json`, per-task `episode.json` (prompts, programs, traces, diagnoses), memory files, logs, reports |

Per-task recordings are excluded except for the examples cited in the reports; ask for the rest.

## Running on Hopper

All simulation and model inference run inside PBS jobs. Example:

```bash
cd /scratch/e0985859/agentic_robot_programming/skill_memory_agent
SMA_RUN_NAME=pen_$(date -u +%Y%m%dT%H%M%SZ) SMA_MODE=suite SMA_API_LEVEL=mid \
  SMA_CONDITIONS=none:skill:user:both SMA_SUITES=pen_teach@3000 \
  SMA_GENERAL_MODEL=/scratch/Projects/CFP-04/CFP04-CF-057/hf_cache/Qwen3-30B-A3B-Instruct-2507 \
  SMA_GENERAL_NAME=qwen3-30b-a3b SMA_GENERAL_GPU_UTIL=0.70 bash hpc/submit.sh
```

**Modes (`SMA_MODE`).**

| Mode | What it runs |
|---|---|
| `suite` | The model on the suites |
| `midsuite` / `pensuite` | A scripted gate first, then the suites |
| `mid_gate` / `pen_gate` | The scripted gate only |
| `replay` | `SMA_REPLAY_RUN` with `SMA_REPLAY_ITEMS` |
| `campaign` | The ASPIRE-style campaign |

**Suites (`SMA_SUITES`).**

- `session`: the tidy-up session;
- `pen`, `pen_teach`: layout set A;
- `pen@N`, `pen_teach@N`: other layout sets.

**Conditions (`SMA_CONDITIONS`).** `none`, `skill`, `user`, `both`.

**Other options.**

- `SMA_QUEUE=auto` routes to the small queue, which allows parallel jobs.
- A crashed simulator process is resumed automatically with `--resume`.

**Outputs.** Results go to `outputs/<run>/<condition>/`.

```bash
python -m sma.eval.report outputs/<run>
python scripts/aggregate_pen_memory.py results results/main_experiment/runs.json out.md
```

## Limitations

- Single runs per condition on the tidy-up session; three layout sets on the pen task.
- One simulated user; a small object set; colour-based perception (no SAM-family segmentation).
- A 30B open-weight model only.
- Failure lessons can mislead when no verified success exists. Skill extraction does not yet reject hard-coded scene angles.

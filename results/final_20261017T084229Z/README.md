# Final midterm run: user memory + skill memory on a realistic tabletop

Hopper job `672093` (interactive, 1 GPU), code commit `31b8c12`. Qwen3-30B-A3B-Instruct-2507 (vLLM) plays every
role: refiner, planner, coder, skill extractor and reflection. Scene: Robosuite/MuJoCo Panda; toy blocks, mugs with
handles, and pocket notebooks; two trays and a walled toy box. Perception is top-down RGB-D only. Ground truth is
used only for the scores below.

Sequence for every condition: one teaching task (the user states lasting preferences), three explicit single-object
tasks, then six vague requests ("Tidy up my desk, please." and similar) on new layouts. Seven of the ten tasks
involve two objects; three need an order (a container is occupied by an object that belongs elsewhere).

| metric | none | skill only | user only | both |
|---|---|---|---|---|
| tasks fully successful (GT) | 4/10 | 5/10 | 9/10 | **9/10** |
| subgoals verified | 11/17 | 11/16 | 15/18 | 15/18 |
| execution attempts (code runs) | 31 | 28 | 33 | **26** |
| failed executions | 20 | 17 | 18 | **11** |
| learned-skill calls | 0 | 7 | 0 | 7 |
| LLM calls | 87 | 66 | 95 | **70** |
| completion tokens | 10479 | 8850 | 9917 | **7205** |
| robot control steps | 9740 | 8164 | 10886 | **8404** |

## What this shows (one run, ten tasks per condition)

- **User memory decides whether vague requests succeed.** After the teaching task the user memory holds three rules:
  block → purple_box, mug → cyan_tray, book → orange_tray. With it, every goal in the six vague tasks is sourced
  from `user_memory`. Without it the refiner guesses (`assumption`). For example, in `tidy_1001` it left a mug on the
  orange tray and put the book on the cyan tray.
- **Skill memory reduces repeated programming and failed executions.** Three skills (mug, book, block) were
  extracted from verified executions and passed 2/2 controlled-variation validations, except the book skill at 1/2.
  They were then called directly seven times. At equal success (user only vs both, 9/10), adding skill memory cut
  failed executions from 18 to 11 and LLM calls from 95 to 70. `tidy_1004` was completed entirely by two skill calls
  with no new code.
- **Skill memory alone does not fix vague requests,** because it knows how to grasp but not where things belong.

## Limits

- One run, ten tasks per condition, a single simulated user. Differences of one or two tasks are within noise.
- Human-designed scaffolding is shared by every condition: parameterized pick/place primitives with IK and joint
  control, a geometric failure diagnosis, the planner's clear-first rule, and API semantics. The model chooses goals,
  order, grasp height and finger direction.
- The book was shrunk to 7.5 × 5.5 cm. With a 9 cm book every model kept closing the fingers along the long side;
  this is a known limitation.
- Persistent failure, `tidy_1003`: lifting the book back out of the walled toy box (fingers versus walls), then the
  block. Perception sometimes reports a phantom "yellow book" from the orange tray base, which costs extra attempts.

Video: `../demo/final_memory_comparison.mp4`. Per-task details: `REPORT.md`, `<condition>/<task>/episode.json`
(including every prompt and response under `llm/`).

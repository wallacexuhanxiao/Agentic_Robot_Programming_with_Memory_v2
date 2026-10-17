"""Scripted mid-level pick-and-place (no LLM): checks grasp_candidates (geometry and Contact-GraspNet), grasp,
free_spots, carry_to, lower_until_contact and check in this scene before any model run. Each task runs once with
the first fitting geometry candidate and once with the first fitting Contact-GraspNet candidate (if any)."""
import json
import os
import sys

import imageio.v2 as imageio

from sma.agent.loop import make_env
from sma.agent.sandbox import Sandbox
from sma.env.tasks import make_task

PROGRAM = '''
cands = grasp_candidates("{obj}")
fit = [c for c in cands if c["fits_open_fingers"] and c["source"].startswith("{source}")]
if not fit:
    raise RuntimeError("no fitting {source} candidate")
g = grasp(fit[0])
if not g["holding_something"]:
    raise RuntimeError("grasp missed")
spots = free_spots("{zone}")
s = spots["spots"][0]
carry_to(s["x"], s["y"], 0.18, s["object_long_axis_yaw"] + fit[0]["yaw"] - fit[0]["object_long_axis_yaw"])
lowered = lower_until_contact()
open_gripper()
retreat()
result = check("{obj}", "{zone}")
'''

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rows = []
tasks = [make_task("place", 21, kinds=["block"]), make_task("place", 23, kinds=["book"]),
         make_task("place", 24, kinds=["mug"]), make_task("clear", 1100)]
for task in tasks:
    for source in ("geometry", "contact-graspnet"):
        env = make_env(task)
        notes, results = [], []
        try:
            env.reset()
            ok_all = True
            sandbox = Sandbox(env)
            for obj, zone in task.goal.items():  # clear tasks list the occupying object first
                env.steps, env.step_budget = 0, 3000
                ok, error = sandbox.run(PROGRAM.format(obj=obj, zone=zone, source=source))
                notes.append(getattr(env, "last_grasp_note", None))
                results.append({"object": obj, "zone": zone, "program_ok": ok, "error": error})
                ok_all &= ok
                if not ok:
                    break
            env.step_budget = env.steps + 1500
            gt = env.private_evaluate(task.goal)
            imageio.mimwrite(f"{out}/{task.task_id}_{source}.mp4", env.frames, fps=20)
        finally:
            env.close()
        row = {"task": task.task_id, "source": source, "subgoals": results, "graspnet_notes": notes,
               "gt": gt["success"], "gt_detail": gt,
               "trace": [{k: t.get(k) for k in ("call", "args", "result", "error", "line")} for t in sandbox.trace]}
        rows.append(row)
        print(task.task_id, source, "program_ok", ok_all, "errors", [r["error"] for r in results if r["error"]],
              "graspnet", notes, "GT", gt["success"], flush=True)
json.dump(rows, open(f"{out}/mid_gate.json", "w"), indent=1, default=str)

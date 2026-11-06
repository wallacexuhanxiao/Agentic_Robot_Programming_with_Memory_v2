"""Where things go, with and without the scene grouping (no manipulation). On the first camera view of each task, with
the task's own instruction (the plain "tidy up" request for the *_vague suites): (1) the place infer_homes() alone
gives each object that is out of place; (2) the refine step's goals without the grouping; (3) with it. Each is scored
against the private goal. With an explicit instruction (realdesk), (3) checks that the grouping does not override it.
Usage (Gemini settings as for a run): grouping_plan.py OUT_DIR SUITE [SUITE ...]"""
import json
import os
import sys

from sma.agent.grouping import infer_homes
from sma.agent.llm import LLM
from sma.agent.loop import make_env
from sma.agent.refine import refine
from sma.env.tasks import suite

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
llm = LLM("", model=os.environ.get("SMA_GENERAL_NAME", "gemini-3.8-flash"))
rows = []
for name in sys.argv[2:]:
    for task in suite(name):
        env = make_env(task, video=False)
        try:
            scene = env.reset()
        finally:
            env.close()
        grouping = infer_homes(scene)
        specs = {"common_sense": refine(llm, task.instruction, [], scene),
                 "with_grouping": refine(llm, task.instruction, [], scene, grouping=grouping)}
        for obj in task.meta["out_of_place"]:
            row = {"task": task.task_id, "object": obj, "goal": task.goal[obj],
                   "grouping_only": (grouping["homes"].get(obj) or {}).get("zone")}
            for key, spec in specs.items():
                g = next((g for g in spec["goals"] if g["object"] == obj), None)
                row[key] = g["zone"] if g else None
                row[key + "_source"] = g["source"] if g else None
            rows.append(row)
            print(json.dumps(row), flush=True)
        json.dump({"task": task.task_id, "grouping": grouping, "specs": specs},
                  open(f"{out}/{task.task_id}.json", "w"), indent=1, default=str)
score = {}
for r in rows:
    k = r["task"].rsplit("_", 1)[0] + ":" + r["object"].split("_", 1)[1]
    for method in ("grouping_only", "common_sense", "with_grouping"):
        d = score.setdefault(method, {}).setdefault(k, [0, 0])
        d[0] += r[method] == r["goal"]
        d[1] += 1
json.dump({"rows": rows, "correct_of_total": score, "usage": llm.usage()}, open(f"{out}/summary.json", "w"), indent=1)
print("SCORE", json.dumps(score), flush=True)
print("USAGE", json.dumps(llm.usage()), flush=True)

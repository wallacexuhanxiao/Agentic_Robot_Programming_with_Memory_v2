"""Long-term memory of observed habits, planning only (no manipulation): the same sequence of desks under the plain
"tidy up" request, (a) without long-term memory: the scene grouping and common sense only; (b) with it: on each desk the
memory is read first (what earlier desks showed), the refine step decides, then what this desk shows is written to the
memory (observe_habits). Each decision for a thing out of place is scored against the private goal.
Usage (Gemini settings as for a run): memory_plan.py OUT_DIR SUITE"""
import json
import os
import sys

from sma.agent.grouping import infer_homes
from sma.agent.llm import LLM
from sma.agent.loop import make_env
from sma.agent.refine import refine
from sma.env.tasks import suite
from sma.memory.user import UserMemory

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
llm = LLM("", model=os.environ.get("SMA_GENERAL_NAME", "gemini-3.5-flash-lite"), log_dir=f"{out}/llm")
tasks = suite(sys.argv[2])
rows = []
for condition in ("no_memory", "memory"):
    memory = UserMemory(f"{out}/{condition}_user_memory.json") if condition == "memory" else None
    for i, task in enumerate(tasks):
        env = make_env(task, video=False)
        try:
            scene = env.reset()
        finally:
            env.close()
        grouping = infer_homes(scene)
        preferences = memory.retrieve(scene) if memory else []
        spec = refine(llm, task.instruction, preferences, scene, grouping=grouping)
        for obj in task.meta["out_of_place"]:
            g = next((g for g in spec["goals"] if g["object"] == obj), None)
            rows.append({"condition": condition, "desk": i, "task": task.task_id, "object": obj, "goal": task.goal[obj],
                         "chose": g["zone"] if g else None, "source": g["source"] if g else None,
                         "memory_said": next((p["zone"] for p in preferences if p["object"] == obj), None),
                         "anchors_on_desk": bool(task.meta["already_tidy"])})
            r = rows[-1]
            print(condition, task.task_id, obj, "goal", r["goal"], "chose", r["chose"], r["source"], "memory", r["memory_said"],
                  "OK" if r["chose"] == r["goal"] else "WRONG", flush=True)
        if memory:
            memory.observe_habits(grouping, scene, task.task_id)
summary = {}
for r in rows:
    key = f"{r['condition']}:{'with anchors' if r['anchors_on_desk'] else 'empty containers'}"
    d = summary.setdefault(key, [0, 0])
    d[0] += r["chose"] == r["goal"]
    d[1] += 1
json.dump({"rows": rows, "correct_of_total": summary, "usage": llm.usage()}, open(f"{out}/summary.json", "w"), indent=1)
print("SCORE", json.dumps(summary), flush=True)
print("USAGE", json.dumps(llm.usage()), flush=True)

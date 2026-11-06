"""Scene grouping check (no LLM, no manipulation): on the first camera view of each task, the place infer_homes() gives
each object that is out of place, against the private goal; also whether the camera found the pile of books.
Usage: grouping_check.py OUT.json SUITE [SUITE ...]"""
import json
import sys

from sma.agent.grouping import infer_homes
from sma.agent.loop import make_env
from sma.env.tasks import suite

rows = []
for name in sys.argv[2:]:
    for task in suite(name):
        env = make_env(task, video=False)
        try:
            scene = env.reset()
            g = infer_homes(scene)
            for obj in (task.meta or {}).get("out_of_place", []):
                home = g["homes"].get(obj, {"zone": None, "evidence": "not seen or not loose"})
                rows.append({"task": task.task_id, "object": obj, "goal": task.goal[obj], "inferred": home["zone"],
                             "correct": home["zone"] == task.goal[obj], "evidence": home["evidence"],
                             "pile_seen": "book_pile" in scene["zones"], "clusters": g["clusters"]})
                print(json.dumps(rows[-1]), flush=True)
        finally:
            env.close()
by_kind = {}
for r in rows:
    k = r["object"].split("_", 1)[1]
    d = by_kind.setdefault(k, {"n": 0, "correct": 0, "no_home": 0, "wrong": 0})
    d["n"] += 1
    d["correct"] += r["correct"]
    d["no_home"] += r["inferred"] is None
    d["wrong"] += r["inferred"] is not None and not r["correct"]
json.dump({"rows": rows, "by_kind": by_kind}, open(sys.argv[1], "w"), indent=1)
print("BY_KIND", json.dumps(by_kind), flush=True)

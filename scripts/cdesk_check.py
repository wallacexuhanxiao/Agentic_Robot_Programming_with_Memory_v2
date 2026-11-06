"""Colour-coded desk check (no LLM). Per desk: (1) the neutral camera report against the private layout: is every thing
found under its colour, and what it measures (hollow, flat top, height, shape); (2) the two labelled images a model
would be shown, saved for a look; (3) with every thing taken as what it is (the private categories standing in for a
correct identification), the geometric camera check against ground truth on the start state: things already in place
should pass, things out of place should not.
Usage: cdesk_check.py OUT_DIR SUITE"""
import json
import os
import sys

from sma.agent.loop import make_env
from sma.agent.identify import role_of
from sma.env.tasks import suite

out = sys.argv[1]
os.makedirs(f"{out}/views", exist_ok=True)
rows = []
for task in suite(sys.argv[2]):
    env = make_env(task, video=False)
    try:
        env.reset()
        scene = env.observe()
        truth = task.meta["true_category"]
        found = {c: scene["things"].get(c) for c in truth}
        for v in env.labelled_views():
            open(f"{out}/views/{task.task_id}_{v['camera']}.png", "wb").write(v["png"])
        env.set_identity({c: {"category": cat, "role": role_of(cat)}
                          for c, cat in truth.items()})
        scene = env.observe()
        name = task.meta["perceived_name"]
        checks = {}
        for obj, zone in task.goal.items():
            c = env.check_on_zone(name[obj], name[zone])
            checks[obj] = {"in_place_at_start": obj not in task.meta["out_of_place"], "camera": c.get("satisfied"),
                           **{k: c.get(k) for k in ("inside_zone", "resting", "standing", "upright")}}
        row = {"task": task.task_id, "truth": truth,
               "things": {c: None if t is None else {k: t.get(k) for k in ("height", "shape", "hollow", "rim_height",
                                                                            "inner_floor_height", "top_flat",
                                                                            "footprint_long", "footprint_short")}
                          for c, t in found.items()},
               "extra_things": sorted(set(scene["things"]) - set(truth)), "checks": checks}
        rows.append(row)
        print(task.task_id, "missing:", [c for c, t in found.items() if t is None], "extra:", row["extra_things"], flush=True)
        for c, t in row["things"].items():
            print(f"   {c:8s} {truth[c]:20s}", t, flush=True)
        for obj, ch in checks.items():
            ok = ch["camera"] == ch["in_place_at_start"]
            print(f"   check {obj:24s} start_in_place={ch['in_place_at_start']} camera={ch['camera']} {'OK' if ok else 'WRONG'}",
                  {k: ch[k] for k in ("inside_zone", "resting", "standing", "upright")}, flush=True)
    finally:
        env.close()
summary = {"desks": len(rows),
           "things": sum(len(r["truth"]) for r in rows),
           "found": sum(sum(t is not None for t in r["things"].values()) for r in rows),
           "checks": sum(len(r["checks"]) for r in rows),
           "checks_agree": sum(sum(c["camera"] == c["in_place_at_start"] for c in r["checks"].values()) for r in rows)}
json.dump({"summary": summary, "rows": rows}, open(f"{out}/cdesk_check.json", "w"), indent=1, default=str)
print("SUMMARY", json.dumps(summary), flush=True)

"""How the model used the action space (act level): from the execution traces of runs, per program attempt, which
functions it called, which grasp candidates it chose (Contact-GraspNet or geometry, adjusted or as given, with room
for the hand or not), whether it moved the gripper off the vertical (e.g. to stand a pen up), which motion paths it
asked for (straight line, joint space, cuRobo), and how often it checked a pose with solve_ik first. Attempt success is
the camera check after the attempt.
Usage: python scripts/action_space_stats.py OUT.md RUN_DIR[:label] [RUN_DIR[:label] ...]   (RUN_DIR holds */episode.json)"""
import collections
import glob
import json
import sys
from pathlib import Path

out, runs = Path(sys.argv[1]), sys.argv[2:]


def attempts(run):
    for ep in sorted(glob.glob(f"{run}/*/episode.json")):
        e = json.load(open(ep))
        for s in e["subgoals"]:
            kind = s["object"].split("_", 1)[1]
            for a in s["attempts"]:
                yield e["task_id"], kind, a


def describe(a):
    trace = a.get("trace") or []
    calls = [t["call"] for t in trace]
    d = {"ok": bool(a["check"].get("satisfied")), "skill": bool(a["used_skills"]), "calls": set(calls)}
    offered = {}
    for t in trace:
        if t["call"] == "grasp_candidates" and isinstance(t.get("result"), str):
            try:
                offered.update({c["id"]: c for c in json.loads(t["result"])})
            except (ValueError, KeyError, TypeError):
                pass
        if t["call"] == "grasp" and t.get("args"):
            c = t["args"][0] if isinstance(t["args"][0], dict) else {}
            src = c.get("source", "?")
            d["grasp_source"] = "contact-graspnet" if "graspnet" in src else "geometry" if "geometry" in src else "own"
            base = offered.get(c.get("id"))
            d["grasp_adjusted"] = bool(base) and any(abs(float(c.get(k, 0)) - float(base.get(k, 0))) > 1e-3
                                                    for k in ("x", "y", "height", "yaw"))
            if "hand_clear" in c:
                d["grasp_hand_clear"] = bool(c["hand_clear"])
        if t["call"] == "move_gripper":
            kw = t.get("kwargs") or {}
            d.setdefault("paths", set()).add(kw.get("path", "line"))
            approach = kw.get("approach") or (t["args"][3] if len(t.get("args", [])) > 3 else None)
            if approach is not None and abs(float(approach[2]) + 1) > .05:
                d["tilted"] = True
        if t["call"] == "carry_to" and isinstance(t.get("result"), dict):
            d.setdefault("carry_planner", set()).add(str(t["result"].get("planner", "?")).split(" ")[0])
        if t["call"] == "solve_ik" and (t.get("kwargs") or {}).get("collision_check"):
            d["ik_collision_check"] = True
    return d


rows, lines = [], []
for spec in runs:
    run, _, label = spec.partition(":")
    label = label or Path(run).parent.name
    per = collections.defaultdict(list)
    for task, kind, a in attempts(run):
        if a.get("code") is None:
            continue
        per[kind].append(describe(a))
        per["all"].append(per[kind][-1])
    for kind in ("all", "pen", "mug", "book", "paper"):
        ds = per.get(kind, [])
        if not ds:
            continue
        n = len(ds)
        grasped = [d for d in ds if "grasp_source" in d]

        def pct(xs, total):
            return f"{len(xs)}/{total}" if total else "-"
        row = {"run": label, "kind": kind, "attempts": n, "ok": sum(d["ok"] for d in ds),
               "skill_calls": sum(d["skill"] for d in ds),
               "graspnet": pct([d for d in grasped if d["grasp_source"] == "contact-graspnet"], len(grasped)),
               "adjusted": pct([d for d in grasped if d.get("grasp_adjusted")], len(grasped)),
               "hand_clear": pct([d for d in grasped if d.get("grasp_hand_clear")], len([d for d in grasped if "grasp_hand_clear" in d])),
               "tilted": pct([d for d in ds if d.get("tilted")], n),
               "joint": pct([d for d in ds if "joint" in d.get("paths", ())], n),
               "curobo": pct([d for d in ds if "curobo" in d.get("paths", ()) or "curobo" in d.get("carry_planner", ())], n),
               "solve_ik": pct([d for d in ds if "solve_ik" in d["calls"]], n),
               "ik_collision": pct([d for d in ds if d.get("ik_collision_check")], n),
               "tilted_ok": pct([d for d in ds if d.get("tilted") and d["ok"]], len([d for d in ds if d.get("tilted")]))}
        rows.append(row)
cols = list(rows[0]) if rows else []
head = {"run": "run", "kind": "kind", "attempts": "attempts", "ok": "camera ok", "skill_calls": "stored skill called",
        "graspnet": "grasp from GraspNet", "adjusted": "candidate adjusted", "hand_clear": "chosen grasp hand-clear",
        "tilted": "hand tilted off vertical", "joint": "joint-space move", "curobo": "cuRobo path",
        "solve_ik": "solve_ik check", "ik_collision": "IK with collision check", "tilted_ok": "tilted attempts ok"}
lines = ["| " + " | ".join(head[c] for c in cols) + " |", "|" + "---|" * len(cols)]
lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n")
out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
print("\n".join(lines))

"""Scripted cuRobo gate (no LLM).
Part A, feasibility against outcomes: for every object -> zone pair of a task, physical_feasibility (cuRobo IK against
the observed depth cloud) at the start and again right before that pair is executed, then a scripted placement
(geometry grasp + place_object at the first free spot) scored by the private evaluator. Pen tasks, single placements
and "clear" tasks (one zone starts occupied by an object that belongs elsewhere) are included.
Part B, cuRobo transit: the same scripted placement with SMA_MOTION=curobo (carry_to plans with cuRobo, falls back
to the straight line), on pen and single-placement tasks.
Usage: curobo_gate.py OUT_DIR A|B INDEX   or   curobo_gate.py OUT_DIR summarize"""
import json
import os
import sys
import time

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
if sys.argv[2] == "B":
    os.environ["SMA_MOTION"] = "curobo"

import imageio.v2 as imageio  # noqa: E402

from sma.agent.loop import make_env  # noqa: E402
from sma.env.perception import inside_zone  # noqa: E402
from sma.env.robot_env import RobotError  # noqa: E402
from sma.env.tasks import make_task, suite  # noqa: E402

PLACE = [make_task("place", s, kinds=[k]) for s, k in ((1, "block"), (2, "book"), (3, "mug"))]
TASKS = {"A": suite("pen")[:3] + PLACE + [make_task("clear", 1100 + i) for i in range(3)],
         "B": suite("pen")[:3] + PLACE}


def summarize():
    rows = [r for f in sorted(os.listdir(out)) if f.startswith("rows_") for r in json.load(open(f"{out}/{f}"))]
    a = [r for r in rows if r["part"] == "A" and "outcome" in r]
    agree = sum(r["feasible_before"] == r["outcome"] for r in a)
    summary = {
        "A_pairs": len(a), "A_feasible_before_and_success": sum(r["feasible_before"] and r["outcome"] for r in a),
        "A_feasible_before_but_failed": sum(r["feasible_before"] and not r["outcome"] for r in a),
        "A_infeasible_before_but_succeeded": sum(not r["feasible_before"] and r["outcome"] for r in a),
        "A_infeasible_before_and_failed": sum(not r["feasible_before"] and not r["outcome"] for r in a),
        "A_agreement": f"{agree}/{len(a)}",
        "A_occupied_zone_at_start_flagged": [(r["task"], r["object"], r["feasible_at_start"]) for r in a
                                             if r.get("zone_occupied_at_start")],
        "A_feasibility_seconds": [r.get("feasibility_seconds") for r in a],
        "B": [(r["task"], r.get("planner"), r.get("outcome"), r.get("error")) for r in rows if r["part"] == "B"]}
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1, default=str)
    print("SUMMARY", json.dumps(summary, default=str), flush=True)


def place(env, obj, zone):
    """Scripted placement: first geometry candidate that fits the open fingers, place_object at the first free spot."""
    cands = env.grasp_candidates(obj)
    pick = next((c for c in cands if c["source"].startswith("geometry") and c["fits_open_fingers"]), None)
    if pick is None:
        return {"result": "no fitting candidate"}
    g = env.grasp(pick)
    if not g["holding_something"]:
        return {"result": "grasp missed"}
    spots = env.free_spots(zone)
    if not spots["spots"]:
        return {"result": "no free spot"}
    s = spots["spots"][0]
    return {"result": "placed", "place": env.place_object(obj, s["x"], s["y"], s["object_long_axis_yaw"])}


if sys.argv[2] == "summarize":
    summarize()
    sys.exit(0)
part, index = sys.argv[2], int(sys.argv[3])
task = TASKS[part][index]
rows = []
env = make_env(task)
try:
    scene = env.reset()
    env.steps, env.step_budget = 0, 8000
    start = {}
    if part == "A":
        for obj, zone in task.goal.items():
            t0 = time.time()
            start[obj] = env.physical_feasibility(obj, zone)
            start[obj]["seconds"] = round(time.time() - t0, 1)
    for obj, zone in task.goal.items():
        # In a "clear" task the second object's zone starts covered by the first object.
        row = {"part": part, "task": task.task_id, "object": obj, "zone": zone,
               "zone_occupied_at_start": bool(task.family == "clear" and obj == list(task.goal)[1])}
        try:
            if part == "A":
                before = env.observe()
                row["zone_occupied_before"] = [n for n, o in before["objects"].items()
                                               if n.split("_")[0] != obj.split("_")[0] and zone in before["zones"]
                                               and inside_zone(o["center"], before["zones"][zone])]
                t0 = time.time()
                feas = env.physical_feasibility(obj, zone)
                row["feasibility_seconds"] = round(time.time() - t0, 1)
                row["feasibility_at_start"], row["feasibility_before"] = start[obj], feas
                row["feasible_at_start"], row["feasible_before"] = start[obj]["ok"], feas["ok"]
            row.update(place(env, obj, zone))
            row["planner"] = (row.get("place") or {}).get("planner")
        except RobotError as exc:
            row["result"], row["error"] = "robot error", {"reason": exc.reason, **exc.details}
        env.step_budget = env.steps + 3000
        try:
            gt = env.private_evaluate(task.goal)["objects"][obj]
            row["gt"] = gt
            row["outcome"] = bool(gt["inside"] and gt["resting"] and gt["upright"])
        except RobotError as exc:
            row["check_error"] = exc.reason
        rows.append(row)
        print(part, task.task_id, obj, "->", zone, "feasible_at_start", row.get("feasible_at_start"), "feasible_before",
              row.get("feasible_before"), (row.get("feasibility_before") or {}).get("failed"), "result", row["result"],
              "error", (row.get("error") or {}).get("reason"), "planner", row.get("planner"), "GT", row.get("outcome"),
              flush=True)
finally:
    if env.frames:
        imageio.mimwrite(f"{out}/{part}_{task.task_id}.mp4", env.frames, fps=20)
    env.close()
json.dump(rows, open(f"{out}/rows_{part}_{task.task_id}.json", "w"), indent=1, default=str)

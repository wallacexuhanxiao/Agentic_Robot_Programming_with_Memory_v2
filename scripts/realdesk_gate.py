"""realdesk gate (no LLM): on the realistic desk, (1) compare what the camera reports with the layout (names, kinds,
receptacle positions and rims), to check the colour calibration of the new objects and receptacles, and (2) do every
out-of-place subtask with reference programs written only with the act-level functions the model sees, through the
same sandbox, with the same monitoring check of all goals after each subtask. Saves a top-down and a demo picture of
the start and a video. Usage (SMA_API_LEVEL=act): realdesk_gate.py OUT_DIR INDEX   or   realdesk_gate.py OUT_DIR summarize"""
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np

from sma.agent.loop import make_env
from sma.agent.sandbox import API_LEVEL, Sandbox
from sma.env.perception import _fill_ratio
from sma.env.tasks import suite

assert API_LEVEL == "act", "run with SMA_API_LEVEL=act"
out, arg = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

from reference_programs import ORDER, PROGRAM  # noqa: E402  (scripts/ is on the path)

if arg == "summarize":
    rows = [json.load(open(f"{out}/{f}")) for f in sorted(os.listdir(out)) if f.startswith("row_")]
    by_kind = {}
    for r in rows:
        for s in r["subtasks"]:
            k = by_kind.setdefault(s["kind"], {"subtasks": 0, "program_ok": 0, "camera_check": 0, "gt": 0})
            k["subtasks"] += 1
            k["program_ok"] += s["program_ok"]
            k["camera_check"] += bool(s["camera_check"])
            k["gt"] += bool(r["gt"]["objects"][s["object"]]["inside"] and r["gt"]["objects"][s["object"]]["resting"]
                            and r["gt"]["objects"][s["object"]]["upright"])
    summary = {"tasks": len(rows), "full_success_gt": sum(r["gt"]["success"] for r in rows),
               "names_all_detected": sum(not r["calibration"]["missing_objects"] and not r["calibration"]["unexpected_objects"]
                                         for r in rows), "by_kind": by_kind}
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    print("SUMMARY", json.dumps(summary), flush=True)
    sys.exit(0)

task = suite(os.environ.get("SMA_GATE_SUITE", "realdesk"))[int(arg)]
env = make_env(task)
try:
    scene = env.reset()
    imageio.imwrite(f"{out}/{task.task_id}_topdown.png", env.render("topdown", 512, 512))
    imageio.imwrite(f"{out}/{task.task_id}_demo.png", env.render("demo", 640, 480))
    # (1) Calibration: perceived objects / receptacles against the layout (simulator state is used here only to score
    # the camera, never by a program).
    truth = {o.name: o for o in task.layout.objects}
    calib = {"objects": {}, "zones": {}}
    for name, o in scene["objects"].items():
        mask = env._masks.get(o["color"])
        calib["objects"][name] = {k: o.get(k) for k in ("kind", "height", "footprint_long", "footprint_short", "visible_pixels")}
        calib["objects"][name]["fill"] = None if mask is None else round(_fill_ratio(mask), 3)
        if mask is not None:  # share of the object's points within 6 mm of its top (flat top vs dome)
            z = env._masks["_xyz"][mask][:, 2]
            calib["objects"][name]["top_share"] = round(float(np.mean(z > np.quantile(z, .9) - .006)), 3)
        if name in truth:
            calib["objects"][name]["centre_error_m"] = round(float(np.hypot(*(np.array(o["center"][:2]) - truth[name].xy))), 4)
    calib["missing_objects"] = sorted(set(truth) - set(scene["objects"]))
    calib["unexpected_objects"] = sorted(set(scene["objects"]) - set(truth))
    for z in task.layout.zones:
        seen = scene["zones"].get(z.name)
        calib["zones"][z.name] = None if seen is None else {
            "centre_error_m": round(float(np.hypot(*(np.array(seen["center"][:2]) - z.xy))), 4),
            "half_size": seen["half_size"], "rim_height": seen.get("rim_height")}
    print(task.task_id, "meta", task.meta, flush=True)
    print(task.task_id, "calibration", json.dumps(calib), flush=True)
    # (2) Subtasks in a fixed order, with the monitoring check of every goal after each one.
    goals = list(task.goal.items())
    checks, _ = env.check_goals(goals)  # at the start, the already-tidy objects should read as done
    calib["start_checks"] = {o: bool(c.get("satisfied")) for o, c in checks.items()}
    print(task.task_id, "start checks", calib["start_checks"], flush=True)
    subtasks = []
    sandbox = Sandbox(env)
    for kind in ORDER:
        for obj in [n for n in task.meta["out_of_place"] if truth[n].kind == kind]:
            zone = task.goal[obj]
            env.steps, env.step_budget = 0, 6000
            cands = env.grasp_candidates(obj)
            print(task.task_id, "candidates", obj, [(c["source"][:22], c["yaw"], c.get("hand_clear"), c.get("fingers_clear"),
                                                    c.get("highest_obstacle_under_hand_m")) for c in cands], flush=True)
            ok, error = sandbox.run(PROGRAM[kind].format(obj=obj, zone=zone))
            env.step_budget = env.steps + 3000
            checks, _ = env.check_goals(goals)
            keys = ("reason", "inside_zone", "resting", "top_height_above_table", "offset_along_across_zone_axis",
                    "zone_half_size", "standing", "compact", "gripper_holding")
            subtasks.append({"object": obj, "kind": kind, "zone": zone, "program_ok": ok,
                             "error": error and (error.get("reason") or error.get("message")),
                             "camera_check": bool(checks[obj].get("satisfied")),
                             "all_goals_after": {o: bool(c.get("satisfied")) for o, c in checks.items()},
                             "unsatisfied_details": {o: {k: c.get(k) for k in keys if k in c}
                                                     for o, c in checks.items() if not c.get("satisfied")}})
            imageio.imwrite(f"{out}/{task.task_id}_after_{len(subtasks)}_{kind}.png", env.last_rgb)
            print(task.task_id, "subtask", json.dumps(subtasks[-1]), flush=True)
    env.step_budget = env.steps + 2000
    gt = env.private_evaluate(task.goal)
    if env.frames:
        imageio.mimwrite(f"{out}/{task.task_id}.mp4", env.frames, fps=20)
finally:
    env.close()
row = {"task": task.task_id, "meta": task.meta, "calibration": calib, "subtasks": subtasks, "gt": gt}
json.dump(row, open(f"{out}/row_{task.task_id}.json", "w"), indent=1, default=str)
print(task.task_id, "GT", gt["success"], json.dumps({k: {kk: v.get(kk) for kk in ("inside", "resting", "upright")}
                                                      for k, v in gt["objects"].items()}), flush=True)

"""Scripted gate (no LLM) for the Contact-GraspNet frame fix and place_object.
Part 1, per object: Contact-GraspNet on the same object points, called the earlier way (world frame) and in the
top-down camera's frame: grasps returned, approach tilt from vertical, how many pass the 30-degree top-down filter.
Part 2, per object and grasp source (contact-graspnet / geometry): grasp the first candidate of that source that fits
the open fingers, compare the in-hand pose (camera and pre-grasp estimate) with the simulator (offline only), then
place_object at the first free spot with that spot's object_long_axis_yaw; record the camera check and the private
ground truth. Simulator state is used for scoring here, never by the primitives."""
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np

from sma.agent.loop import make_env
from sma.env.robot_env import RobotError, _rot2, _wrap_half_pi
from sma.env.tasks import make_task, suite

# Usage: pose_gate.py OUT_DIR INDEX (one task per process: long simulator processes have crashed before)
#        pose_gate.py OUT_DIR summarize
out, arg = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
TASKS = suite("pen")[:6] + [make_task("place", s, kinds=[k]) for s, k in ((1, "block"), (2, "book"), (3, "mug"))]
SOURCES = ("contact-graspnet", "geometry")


def deg(a):
    return round(float(np.degrees(a)), 1)


def graspnet_views(env, obj):
    """Both calls on the same points, before anything moves."""
    scene = env.observe()
    o = scene["objects"][obj]
    points = env._masks["_xyz"][env._masks[o["color"]]]
    down = env.down.as_matrix()[:, 2]
    row = {}
    for view in ("world", "camera"):
        grasps, note = env._graspnet_raw(points, view)
        if grasps is None:
            row[view] = {"note": note}
            continue
        tilts = [float(np.degrees(np.arccos(np.clip(m[:3, 2] @ down, -1, 1)))) for m, _ in grasps]
        offsets = [float(np.hypot(*(m[:2, 3] - o["center"][:2]))) for m, _ in grasps]
        row[view] = {"grasps": len(grasps), "within_30deg": int(sum(t <= 30 for t in tilts)),
                     "tilt_deg": sorted(round(t, 1) for t in tilts),
                     "grip_point_offset_from_centre_m": [round(d, 3) for d in offsets],
                     "scores": [round(s, 3) for _, s in grasps]}
    return row


def in_hand_truth(env, obj):
    """Simulator pose of a held pen relative to the gripper (pens only: the body x axis is the long axis)."""
    s = env.private_state()[obj]
    g = env._yaw()
    yaw = float(np.arctan2(s["x_axis"][1], s["x_axis"][0]))
    offset = _rot2(-g) @ (np.asarray(s["position"][:2]) - env._ee()[:2])
    return {"offset": [round(float(v), 4) for v in offset], "yaw_offset": round(_wrap_half_pi(yaw - g), 3)}


def compare(rel, truth):
    if rel is None or truth is None:
        return None
    return {"offset_error_m": round(float(np.hypot(*(np.asarray(rel["offset"]) - truth["offset"]))), 4),
            "yaw_error_deg": abs(deg(_wrap_half_pi(rel["yaw_offset"] - truth["yaw_offset"]))) if rel["elongated"] else None}




def summarize():
    rows = [r for f in sorted(os.listdir(out)) if f.startswith("rows_") for r in json.load(open(f"{out}/{f}"))]
    summary = {"tasks": len({r["task"] for r in rows})}
    for source in SOURCES:
        for group in ("pen", "other"):
            rs = [r for r in rows if r.get("source") == source and r["object"].endswith("pen") == (group == "pen")]
            summary[f"{source}/{group}"] = {
                "episodes": len(rs), "fitting_candidate": sum(r["result"] != "no fitting candidate" for r in rs),
                "held": sum(bool((r.get("grasp") or {}).get("holding_something")) for r in rs),
                "gt_success": sum(bool((r.get("gt") or {}).get("success")) for r in rs)}
    frames = [r for r in rows if r["part"] == "graspnet_frames"]
    for view in ("world", "camera"):
        summary[f"graspnet_{view}_frame"] = {"grasps": sum(r[view].get("grasps", 0) for r in frames),
                                             "within_30deg": sum(r[view].get("within_30deg", 0) for r in frames)}
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    print("SUMMARY", json.dumps(summary), flush=True)


if arg == "summarize":
    summarize()
    sys.exit(0)
rows = []
for task in [TASKS[int(arg)]]:
    obj, zone = next(iter(task.goal.items()))
    kind = obj.split("_", 1)[1]
    env = make_env(task, video=False)
    try:
        env.reset()
        views = graspnet_views(env, obj)
    finally:
        env.close()
    rows.append({"task": task.task_id, "object": obj, "part": "graspnet_frames", **views})
    print(f"FRAMES {task.task_id} {obj}: " + " | ".join(
        f"{v}: {r.get('grasps', r.get('note'))} grasps, {r.get('within_30deg', 0)} within 30 deg, "
        f"tilt median {np.median(r['tilt_deg']) if r.get('tilt_deg') else None}" for v, r in views.items()), flush=True)

    for source in SOURCES:
        env = make_env(task)
        row = {"task": task.task_id, "object": obj, "zone": zone, "part": "grasp_place", "source": source}
        try:
            env.reset()
            env.steps, env.step_budget = 0, 4000
            cands = env.grasp_candidates(obj)
            row["graspnet_note"] = env.last_grasp_note
            row["candidate_sources"] = [c["source"] for c in cands]
            pick = next((c for c in cands if c["source"].startswith(source) and c["fits_open_fingers"]), None)
            if pick is None:
                row["result"] = "no fitting candidate"
            else:
                row["candidate"] = {k: pick[k] for k in ("x", "y", "height", "yaw", "score", "squeeze_width",
                                                         "object_center", "object_long_axis_yaw")}
                g = env.grasp(pick)
                row["grasp"] = g
                if g["holding_something"]:
                    held = env._held["in_hand"]
                    truth = in_hand_truth(env, obj) if kind == "pen" else None
                    estimate = held and held.get("estimate")
                    est_rel = None
                    if estimate:  # the pre-grasp estimate in the same form, for comparison with the camera
                        off = _rot2(-env._yaw()) @ (np.asarray(estimate["center"]) - env._ee()[:2])
                        est_rel = {"offset": off.tolist(), "elongated": estimate["elongated"],
                                   "yaw_offset": _wrap_half_pi(estimate["yaw"] - env._yaw())}
                    row["in_hand"] = {"used": held and held["source"], "truth": truth,
                                      "camera_vs_truth": compare(held if held and held["source"] == "camera" else None, truth),
                                      "estimate_vs_truth": compare(est_rel, truth)}
                    spots = env.free_spots(zone)
                    if not spots["spots"]:
                        row["result"] = "no free spot"
                    else:
                        s = spots["spots"][0]
                        row["place"] = env.place_object(obj, s["x"], s["y"], s["object_long_axis_yaw"])
                        row["result"] = "placed"
                else:
                    row["result"] = "grasp missed"
        except RobotError as exc:
            row["result"], row["error"] = "robot error", {"reason": exc.reason, **exc.details}
        finally:
            try:
                env.step_budget = env.steps + 1500
                row["camera_check"] = env.check_on_zone(obj, zone)
                row["gt"] = env.private_evaluate(task.goal)
            except RobotError as exc:
                row["check_error"] = exc.reason
            if env.frames:
                imageio.mimwrite(f"{out}/{task.task_id}_{source.split('-')[-1]}.mp4", env.frames, fps=20)
            env.close()
        rows.append(row)
        place = row.get("place") or {}
        print(task.task_id, obj, source, "result", row["result"], "error", row.get("error", {}).get("reason"),
              "hold", (row.get("grasp") or {}).get("holding_something"), "in_hand", row.get("in_hand", {}).get("used"),
              "in_hand_vs_truth", row.get("in_hand", {}).get("camera_vs_truth") or row.get("in_hand", {}).get("estimate_vs_truth"),
              "pos_err", place.get("position_error_m"), "yaw_err", place.get("yaw_error_deg"),
              "camera", (row.get("camera_check") or {}).get("satisfied"), "GT", (row.get("gt") or {}).get("success"),
              flush=True)

json.dump(rows, open(f"{out}/rows_{TASKS[int(arg)].task_id}.json", "w"), indent=1, default=str)

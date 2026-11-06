"""Scripted pen-holder gate (no LLM): can the arm stand a pen upright in the holder with place_upright?
Per task: (1) perception of the holder (rendered floor hue, detected centre vs the layout); (2) grasp the pen across
its width 3.5 cm from its centre (geometry candidate shifted along the pen), place_upright at the perceived holder
centre; (3) camera check and private ground truth (lower end in the cup, tilt within 40 degrees of vertical).
Usage: stand_gate.py OUT_DIR INDEX   or   stand_gate.py OUT_DIR summarize"""
import json
import os
import sys

import cv2
import imageio.v2 as imageio
import numpy as np

from sma.agent.loop import make_env
from sma.env.robot_env import RobotError
from sma.env.tasks import suite

out, arg = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
TASKS = suite("pen_holder")[:6]
if arg == "summarize":
    rows = [json.load(open(f"{out}/{f}")) for f in sorted(os.listdir(out)) if f.startswith("row_")]
    summary = {"tasks": len(rows), "holder_seen": sum(r.get("holder_seen", False) for r in rows),
               "held": sum(bool((r.get("grasp") or {}).get("holding_something")) for r in rows),
               "gt_success": sum(bool((r.get("gt") or {}).get("success")) for r in rows),
               "camera_check": sum(bool((r.get("camera_check") or {}).get("satisfied")) for r in rows),
               "tilts_deg": [((r.get("gt") or {}).get("objects") or {}).get(r["object"], {}).get("tilt_from_vertical_deg") for r in rows]}
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    print("SUMMARY", json.dumps(summary), flush=True)
    sys.exit(0)

task = TASKS[int(arg)]
obj, zone = next(iter(task.goal.items()))
spec = task.layout.zones[0]
row = {"task": task.task_id, "object": obj, "zone": zone}
env = make_env(task)
try:
    scene = env.reset()
    env.steps, env.step_budget = 0, 6000
    xyz, rgb = env._masks["_xyz"], env.last_rgb
    d = np.abs(xyz[..., :2] - np.array(spec.xy))
    floor = (d[..., 0] < .025) & (d[..., 1] < .025) & (np.abs(xyz[..., 2] - scene["table_z"]) < .006)
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    row["holder_floor_hue"] = float(np.median(hsv[..., 0][floor])) if floor.any() else None
    row["holder_floor_pixels"] = int(floor.sum())
    seen = scene["zones"].get(zone)
    row["holder_seen"] = seen is not None
    row["holder_centre_error_m"] = None if seen is None else round(float(np.hypot(*(np.array(seen["center"][:2]) - spec.xy))), 4)
    print(f"{task.task_id}: holder floor hue {row['holder_floor_hue']} ({row['holder_floor_pixels']} px), seen {row['holder_seen']}, "
          f"centre error {row['holder_centre_error_m']}", flush=True)
    if seen is not None:
        cands = env.grasp_candidates(obj)
        c = next(c for c in cands if c["source"].startswith("geometry") and c["fits_open_fingers"])
        a = c["object_long_axis_yaw"]
        shifted = dict(c, x=c["x"] + .035 * np.cos(a), y=c["y"] + .035 * np.sin(a))  # 3.5 cm towards one end
        row["grasp"] = env.grasp(shifted)
        if row["grasp"]["holding_something"]:
            row["place"] = env.place_upright(obj, seen["center"][0], seen["center"][1])
except RobotError as exc:
    row["error"] = {"reason": exc.reason, **exc.details}
finally:
    env.step_budget = env.steps + 3000
    try:
        row["camera_check"] = env.check_on_zone(obj, zone)
    except RobotError as exc:
        row["check_error"] = {"reason": exc.reason, **exc.details}
    row["gt"] = env.private_evaluate(task.goal)  # scored whatever the camera check did
    if env.frames:
        imageio.mimwrite(f"{out}/{task.task_id}.mp4", env.frames, fps=20)
    env.close()
json.dump(row, open(f"{out}/row_{task.task_id}.json", "w"), indent=1, default=str)
g = ((row.get("gt") or {}).get("objects") or {}).get(obj, {})
print(task.task_id, obj, "hold", (row.get("grasp") or {}).get("holding_something"), "error", (row.get("error") or {}).get("reason"),
      "place", {k: (row.get("place") or {}).get(k) for k in ("grip_point_to_lower_end_m", "stopped_by_contact")},
      "camera", (row.get("camera_check") or {}).get("satisfied"), (row.get("camera_check") or {}).get("pen_pixels_high_in_holder"),
      "GT", (row.get("gt") or {}).get("success"), {k: g.get(k) for k in ("inside", "resting", "upright", "tilt_from_vertical_deg")},
      flush=True)

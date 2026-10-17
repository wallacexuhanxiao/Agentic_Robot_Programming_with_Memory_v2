"""Scripted pen-case gate (no LLM). For each pen task: (1) perception vs the known layout (pen kind and size, case
centre/size/orientation, and the rendered hue of the case floor for the colour calibration); (2) the pen put into
the case twice: turned to the case's long axis while carried ("aligned"), and carried without turning ("unturned").
The task is meant to need the turn, so "aligned" should succeed and "unturned" should mostly fail."""
import json
import os
import sys

import cv2
import imageio.v2 as imageio
import numpy as np

from sma.agent.loop import make_env
from sma.agent.sandbox import Sandbox
from sma.env.perception import ZONE_HUES
from sma.env.scene import RECEPTACLES, receptacle_half
from sma.env.tasks import suite

PROGRAM = '''
cands = grasp_candidates("{obj}")
fit = [c for c in cands if c["fits_open_fingers"] and c["source"].startswith("geometry")]
if not fit:
    raise RuntimeError("no fitting candidate")
c = fit[0]
g = grasp(c)
if not g["holding_something"]:
    raise RuntimeError("grasp missed")
spots = free_spots("{zone}")
if not spots["spots"]:
    raise RuntimeError("no free spot")
s = spots["spots"][0]
yaw = {yaw}
carry_to(s["x"], s["y"], 0.18, yaw)
lowered = lower_until_contact()
open_gripper()
retreat()
result = check("{obj}", "{zone}")
'''
YAW = {"aligned": 's["object_long_axis_yaw"] + c["yaw"] - c["object_long_axis_yaw"]', "unturned": "None"}


def angle_diff(a, b):
    return abs((a - b + np.pi / 2) % np.pi - np.pi / 2)


out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rows = []
for task in suite("pen")[:4]:
    obj, zone = next(iter(task.goal.items()))
    spec = task.layout.zones[0]
    for variant in ("aligned", "unturned"):
        env = make_env(task)
        try:
            scene = env.reset()
            if variant == "aligned":  # perception and colour calibration against the known layout
                xyz, rgb = env._masks["_xyz"], env.last_rgb
                hx, hy = receptacle_half("case")
                d = xyz[..., :2] - np.array(spec.xy)
                a = d[..., 0] * np.cos(spec.yaw) + d[..., 1] * np.sin(spec.yaw)
                b = -d[..., 0] * np.sin(spec.yaw) + d[..., 1] * np.cos(spec.yaw)
                floor = (np.abs(a) < hx - .006) & (np.abs(b) < hy - .006) & (np.abs(xyz[..., 2] - scene["table_z"]) < .006)
                hsv = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
                hue = float(np.median(hsv[..., 0][floor])) if floor.any() else None
                seen_zone = scene["zones"].get(zone)
                seen_pen = scene["objects"].get(obj)
                pen_spec = task.layout.objects[0]
                print(f"CALIB {task.task_id}: case floor rendered hue {hue} (constant {ZONE_HUES[spec.color]}), "
                      f"{int(floor.sum())} px", flush=True)
                print(f"PERCEPTION {task.task_id}: objects {sorted(scene['objects'])} zones {sorted(scene['zones'])}", flush=True)
                if seen_zone:
                    print(f"  case seen centre {seen_zone['center'][:2]} half {seen_zone['half_size']} yaw "
                          f"{seen_zone['long_axis_yaw']} | true centre {list(spec.xy)} half {[hx, hy]} yaw {spec.yaw} "
                          f"(yaw error {np.degrees(angle_diff(seen_zone['long_axis_yaw'], spec.yaw)):.1f} deg)", flush=True)
                if seen_pen:
                    print(f"  pen seen kind {seen_pen['kind']} long {seen_pen['footprint_long']} short "
                          f"{seen_pen['footprint_short']} height {seen_pen['height']} yaw {seen_pen['long_axis_yaw']} | "
                          f"true yaw {pen_spec.yaw} (error {np.degrees(angle_diff(seen_pen['long_axis_yaw'], pen_spec.yaw)):.1f} deg)",
                          flush=True)
            sandbox = Sandbox(env)
            env.steps, env.step_budget = 0, 3000
            ok, error = sandbox.run(PROGRAM.format(obj=obj, zone=zone, yaw=YAW[variant]))
            env.step_budget = env.steps + 1500
            check = env.check_on_zone(obj, zone)
            gt = env.private_evaluate(task.goal)
            imageio.mimwrite(f"{out}/{task.task_id}_{variant}.mp4", env.frames, fps=20)
        finally:
            env.close()
        trace = [{k: t.get(k) for k in ("call", "args", "result", "error", "line")} for t in sandbox.trace]
        lowered = next((t.get("result") for t in trace if t["call"] == "lower_until_contact"), None)
        rows.append({"task": task.task_id, "variant": variant, "program_ok": ok, "error": error, "camera_check": check,
                     "gt": gt, "trace": trace})
        print(task.task_id, variant, "program_ok", ok, "error", error and (error.get("reason") or error.get("message")),
              "lowered", lowered, "camera", check.get("satisfied"), "GT", gt["success"], gt["objects"][obj], flush=True)
json.dump(rows, open(f"{out}/pen_gate.json", "w"), indent=1, default=str)

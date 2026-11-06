"""act-level gate (no LLM): reference programs written only with the functions the model sees at SMA_API_LEVEL=act,
run through the same sandbox, to show the API suffices before a model is asked to compose the manipulation itself.
Tasks 0-5: stand a pen upright in the pen holder; tasks 6-8: put a pen in the pen case.
Usage (SMA_API_LEVEL=act): act_gate.py OUT_DIR INDEX   or   act_gate.py OUT_DIR summarize"""
import json
import os
import sys

import imageio.v2 as imageio

from sma.agent.loop import make_env
from sma.agent.sandbox import API_LEVEL, Sandbox
from sma.env.tasks import suite

assert API_LEVEL == "act", "run with SMA_API_LEVEL=act"
out, arg = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
TASKS = suite("pen_holder")[:6] + suite("pen")[:3] + suite("pen_holder")[:2]  # 9, 10: the pen dropped flat (negative)

HOLDER = '''
import math
scene = observe()
table = scene["table_z"]
holder = scene["zones"]["{zone}"]
pen = scene["objects"]["{obj}"]
cands = grasp_candidates("{obj}")
c = [k for k in cands if k["source"].startswith("geometry") and k["fits_open_fingers"]][0]
a = c["object_long_axis_yaw"]
g = grasp(dict(c, x=c["x"] + 0.035 * math.cos(a), y=c["y"] + 0.035 * math.sin(a)))
if not g["holding_something"]:
    raise RuntimeError("grasp missed")
pose = gripper_pose()
side = pose["side"]
centre = c["object_center"]
toward = (centre[0] - pose["position"][0]) * side[0] + (centre[1] - pose["position"][1]) * side[1]
s = 1 if toward > 0 else -1
hang = pen["footprint_long"] / 2 + abs(toward)
hx, hy = holder["center"][0], holder["center"][1]
rim = holder["rim_height"] or 0.1
high = table + rim + hang + 0.12
low = table + 0.004 + 0.008 + hang
chosen = None
for turn in (0, 30, -30, 60, -60, 90, -90):
    h = math.atan2(hy, hx + 0.56) + math.radians(turn)
    approach = [math.cos(h), math.sin(h), 0.0]
    up = [0.0, 0.0, -s]
    finger = [up[1] * approach[2] - up[2] * approach[1], up[2] * approach[0] - up[0] * approach[2],
              up[0] * approach[1] - up[1] * approach[0]]
    if solve_ik(hx, hy, high, approach, finger)["reachable"] and solve_ik(hx, hy, low, approach, finger)["reachable"]:
        chosen = [approach, finger]
        break
if chosen is None:
    raise RuntimeError("no reachable upright pose")
move_gripper(pose["position"][0], pose["position"][1], high)
move_gripper(hx, hy, high)
move_gripper(hx, hy, high, chosen[0], chosen[1], path="joint")
move_by(dz=low - high)
open_gripper()
move_by(dz=0.15)
go_home()
result = check("{obj}", "{zone}")
'''

CASE = '''
cands = grasp_candidates("{obj}")
c = [k for k in cands if k["source"].startswith("geometry") and k["fits_open_fingers"]][0]
g = grasp(c)
if not g["holding_something"]:
    raise RuntimeError("grasp missed")
spot = free_spots("{zone}")["spots"][0]
carry_to(spot["x"], spot["y"], 0.18)
rotate_by(spot["object_long_axis_yaw"] - c["object_long_axis_yaw"])
lower_until_contact()
open_gripper()
retreat()
result = check("{obj}", "{zone}")
'''

FLAT = '''
cands = grasp_candidates("{obj}")
c = [k for k in cands if k["source"].startswith("geometry") and k["fits_open_fingers"]][0]
g = grasp(c)
holder = observe()["zones"]["{zone}"]
carry_to(holder["center"][0], holder["center"][1], 0.2)
lower_until_contact()
open_gripper()
retreat()
result = check("{obj}", "{zone}")
'''

if arg == "summarize":
    rows = [json.load(open(f"{out}/{f}")) for f in sorted(os.listdir(out)) if f.startswith("row_")]
    summary = {}
    for family in ("pen_holder", "pen", "pen_holder_flat"):
        rs = [r for r in rows if r["family"] == family]
        summary[family] = {"tasks": len(rs), "program_ok": sum(r["program_ok"] for r in rs),
                           "camera_check": sum(bool(r["camera_check"].get("satisfied")) for r in rs),
                           "gt_success": sum(bool(r["gt"]["success"]) for r in rs)}
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    print("SUMMARY", json.dumps(summary), flush=True)
    sys.exit(0)

task = TASKS[int(arg)]
obj, zone = next(iter(task.goal.items()))
flat = int(arg) >= 9
program = (FLAT if flat else HOLDER if task.family == "pen_holder" else CASE).format(obj=obj, zone=zone)
env = make_env(task)
try:
    env.reset()
    sandbox = Sandbox(env)
    env.steps, env.step_budget = 0, 6000
    ok, error = sandbox.run(program)
    env.step_budget = env.steps + 3000
    check = env.check_on_zone(obj, zone)
    gt = env.private_evaluate(task.goal)
    if env.frames:
        imageio.mimwrite(f"{out}/{task.task_id}{'_flat' if flat else ''}.mp4", env.frames, fps=20)
finally:
    env.close()
calls = [t["call"] for t in sandbox.trace]
row = {"task": task.task_id, "family": task.family + ("_flat" if flat else ""), "program_ok": ok, "error": error, "camera_check": check, "gt": gt,
       "calls": calls}
json.dump(row, open(f"{out}/row_{task.task_id}{'_flat' if flat else ''}.json", "w"), indent=1, default=str)
g = gt["objects"][obj]
print(task.task_id + (" (flat drop)" if flat else ""), "program_ok", ok, "error", error and (error.get("reason") or error.get("message")),
      "camera", check.get("satisfied"), {k: check.get(k) for k in ("standing", "compact", "top_height_above_table", "horizontal_span_m")}, "GT", gt["success"], {k: g.get(k) for k in ("inside", "resting", "upright", "tilt_from_vertical_deg")},
      "calls", len(calls), flush=True)

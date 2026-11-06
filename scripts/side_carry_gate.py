"""Side-grasp carry gate (no LLM): the notebook taken by an edge (the act-level edge pinch, as in the prompt's generic
example) and put on the pile of books in the two ways DeepSeek wrote it in the skill-vs-none comparison
(2026-11-06), run through the sandbox with the functions the model gets:
  carry: rotate_by to the pile's axis, carry_to above the pile centre, lower_until_contact (the skill run's programs);
  lower: move_gripper above the pile centre keeping the grasp's orientation, lower_until_contact (the none run's).
Each variant starts from a fresh desk. Recorded: the program's error, what each motion call reported and the hand's tilt
after it, the camera check, whether the camera still sees the notebook on a later look, and the ground truth (on the
pile and lying flat, with the notebook's tilt).
Usage (SMA_API_LEVEL=act, segmentation service running): side_carry_gate.py OUT_DIR INDEX (suite: SMA_GATE_SUITE,
default sdesk@17100; variants: SMA_SIDE_VARIANTS joined by ":", default carry:lower)"""
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np

from sma.agent.identify import private_truth, role_of
from sma.agent.loop import make_env
from sma.agent.sandbox import Sandbox
from sma.env.tasks import suite

PINCH = '''
obs = observe()
table_z = obs["table_z"]
zone = obs["zones"]["{zone}"]
side = [c for c in grasp_candidates("{obj}") if c.get("pinch") and c.get("edge_free_below")]
c = next((c for c in side if c.get("reachable_checked")), side[0])
a, f = c["approach"], c["finger_axis"]
x, y, z = c["x"], c["y"], table_z + c["height"]
open_gripper()
move_gripper(x - 0.05 * a[0], y - 0.05 * a[1], z + 0.15, a, f, path="joint")
move_gripper(x - 0.05 * a[0], y - 0.05 * a[1], z, a, f)
move_by(0.05 * a[0], 0.05 * a[1], 0.0)
if not close_gripper()["holding_something"]:
    raise RuntimeError("side grasp missed")
move_by(dz=0.08)
oc = c["object_center"]
'''
PROGRAMS = {
    "carry": PINCH + '''
delta = zone["long_axis_yaw"] - c["object_long_axis_yaw"]
delta = (delta + math.pi / 2) % math.pi - math.pi / 2
ca, sa = math.cos(delta), math.sin(delta)
rotate_by(delta)
ox, oy = oc[0] - x, oc[1] - y
ox, oy = ca * ox - sa * oy, sa * ox + ca * oy
carry_to(zone["center"][0] - ox, zone["center"][1] - oy, height=0.25)
lower_until_contact()
open_gripper()
ap = gripper_pose()["approach"]
move_by(-0.06 * ap[0], -0.06 * ap[1], 0.0)
move_by(dz=0.1)
result = check("{obj}", "{zone}")
''',
    "lower": PINCH + '''
move_gripper(zone["center"][0] - (oc[0] - x), zone["center"][1] - (oc[1] - y), table_z + 0.25, a, f, path="joint")
lower_until_contact()
open_gripper()
move_by(-0.06 * a[0], -0.06 * a[1], 0.0)
move_by(dz=0.1)
result = check("{obj}", "{zone}")
'''}

out, index = sys.argv[1], int(sys.argv[2])
os.makedirs(out, exist_ok=True)
task = suite(os.environ.get("SMA_GATE_SUITE", "sdesk@17100"))[index]
kind_of = {o.name: o.kind for o in task.layout.objects}
nb = next((o for o in task.meta["out_of_place"] if kind_of[o] == "notebook"), None)
rows = []
for variant in os.environ.get("SMA_SIDE_VARIANTS", "carry:lower").split(":"):
    row = {"task": task.task_id, "variant": variant, "notebook": nb}
    if nb is None:
        row["skipped"] = "no notebook out of place"
        rows.append(row)
        continue
    env = make_env(task)
    if hasattr(env, "track_dir"):  # under track_debug.py: one directory of looks per variant
        env.track_dir += f"_{variant}"
        os.makedirs(env.track_dir, exist_ok=True)
    try:
        scene = env.reset()
        truth, seen_as = private_truth(env, task, scene)
        env.set_identity({t: {"category": c, "role": role_of(c)} for t, c in truth.items()})
        if nb not in seen_as or task.goal[nb] not in seen_as:
            raise RuntimeError("camera did not separate the notebook or the pile at the start")
        name, zone = seen_as[nb], seen_as[task.goal[nb]]
        row.update(thing=name, zone=zone)
        env.steps, env.step_budget = 0, 9000

        def tilt(entry, result):  # where the fingers point after each call (z of the approach; -1 = straight down)
            entry["approach_z_after"] = round(float(env.gripper_pose()["approach"][2]), 3)

        sandbox = Sandbox(env, on_result=tilt)
        ok, error = sandbox.run(PROGRAMS[variant].format(obj=name, zone=zone))
        row["program_ok"], row["error"] = ok, error
        row["calls"] = [{"call": e["call"], "approach_z_after": e.get("approach_z_after"), "error": e.get("error"),
                         "result": {k: e["result"].get(k) for k in
                                    ("orientation_kept", "stopped_by_contact", "turned_by", "planner")
                                    if isinstance(e.get("result"), dict) and k in e["result"]}}
                        for e in sandbox.trace if e["call"] in ("close_gripper", "rotate_by", "carry_to", "move_gripper",
                                                                "lower_until_contact", "open_gripper")]
        env.step_budget = env.steps + 3000
        row["camera_check"] = env.check_on_zone(name, zone)
        later = env.observe()  # a later look, as at the next subgoal
        row["seen_on_later_look"] = name in later["objects"]
        state = env.private_state()[nb]
        row["notebook_tilt_from_flat_deg"] = round(float(np.degrees(np.arccos(min(1.0, abs(state["z_axis"][2]))))), 1)
        row["gt"] = env.private_evaluate({nb: task.goal[nb]})["objects"][nb]
        row["gt_ok"] = bool(row["gt"]["inside"] and row["gt"]["resting"] and row["gt"]["upright"])
    except Exception as exc:  # a robot error outside the program, or the camera missing something
        row["harness_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if env.frames:
            imageio.mimwrite(f"{out}/{task.task_id}_{variant}.mp4", env.frames, fps=20)
        env.close()
    print(task.task_id, variant, "program_ok", row.get("program_ok"), (row.get("error") or {}).get("reason")
          if isinstance(row.get("error"), dict) else row.get("error"), "camera", (row.get("camera_check") or {}).get("satisfied"),
          "seen_later", row.get("seen_on_later_look"), "gt", row.get("gt_ok"), "tilt", row.get("notebook_tilt_from_flat_deg"),
          flush=True)
    print("   calls", json.dumps(row.get("calls")), flush=True)
    rows.append(row)
json.dump(rows, open(f"{out}/{task.task_id}.json", "w"), indent=1, default=str)

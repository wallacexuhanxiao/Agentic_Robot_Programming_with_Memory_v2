"""Scripted ASPIRE-style pick-and-place (no LLM) through the real ASPIRE backends: checks that segmentation,
Contact-GraspNet, PyRoki/cuRobo and execution work in this scene before any model run."""
import json
import os
import sys

import imageio.v2 as imageio

from sma.agent.loop import make_env
from sma.agent.sandbox import Sandbox
from sma.env.tasks import make_task

PROGRAM = '''
base = get_robot_state().base_frame
scene = observe()
o = scene["objects"]["{obj}"]
z = scene["zones"]["{zone}"]
found = localize_object("{query}", camera_name="topdown", target_frame=base)
grasp_pose = None
source = "geometry fallback"
if found.ok:
    grasps = generate_grasps("topdown", found.segmentation, backend="contact-graspnet", max_candidates=10)
    best = select_grasp(grasps, strategy="top_down") if grasps.ok else None
    if best is not None:
        grasp_pose = best.pose
        source = "contact-graspnet score %.2f" % best.score
if grasp_pose is None:
    down = get_robot_state().end_effector_poses["primary"].quaternion_wxyz
    grasp_pose = Pose((o["center"][0], o["center"][1], scene["table_z"] + o["height"] / 2), down, base)
print("grasp source:", source, list(grasp_pose.position))
p = [float(v) for v in grasp_pose.position]
q = grasp_pose.quaternion_wxyz
open_gripper()
move_to_pose(Pose((p[0], p[1], p[2] + 0.12), q, base))
move_to_pose(Pose((p[0], p[1], p[2]), q, base), tolerance=0.015)
held = close_gripper()
move_to_pose(Pose((p[0], p[1], scene["table_z"] + 0.22), q, base))
grip_h = p[2] - scene["table_z"]
move_to_pose(Pose((z["center"][0], z["center"][1], scene["table_z"] + 0.22), q, base))
move_to_pose(Pose((z["center"][0], z["center"][1], scene["table_z"] + grip_h + 0.012), q, base), tolerance=0.02)
open_gripper()
move_to_pose(Pose((z["center"][0], z["center"][1], scene["table_z"] + 0.22), q, base))
go_home()
'''

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rows = []
for seed, kind in ((21, "block"), (23, "book"), (24, "mug")):
    task = make_task("place", seed, kinds=[kind])
    obj, zone = next(iter(task.goal.items()))
    env = make_env(task)
    try:
        env.reset()
        sandbox = Sandbox(env)
        env.steps, env.step_budget = 0, 3000
        ok, error = sandbox.run(PROGRAM.format(obj=obj, zone=zone, query=obj.replace("_", " ")))
        env.step_budget = env.steps + 1500
        check = env.check_on_zone(obj, zone)
        gt = env.private_evaluate(task.goal)
        imageio.mimwrite(f"{out}/{task.task_id}.mp4", env.frames, fps=20)
    finally:
        env.close()
    row = {"task": task.task_id, "kind": kind, "program_ok": ok, "error": error, "camera_check": check.get("satisfied"),
           "gt": gt["success"], "trace": [{k: t.get(k) for k in ("call", "args", "error", "line")} for t in sandbox.trace]}
    rows.append(row)
    print(task.task_id, kind, "program_ok", ok, "error", error, "camera", check.get("satisfied"), "GT", gt["success"], flush=True)
json.dump(rows, open(f"{out}/aspire_gate.json", "w"), indent=1, default=str)

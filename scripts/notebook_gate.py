"""nbdesk gate (no LLM): a reference way to tidy the A5 notebook, through the same functions the model gets at the act
level (observe, grasp_candidates, solve_ik, move_gripper, move_by, close_gripper, open_gripper, check): pinch the
edge that sticks out over the desk edge from outside the desk, lift, carry along straight lines at a fixed orientation,
lay it on the pile of books with the gripped edge 4 cm out over the pile's side (the pickup in reverse: the hand hangs
beside the pile), let go, and score with the camera and the ground truth. On a varied desk the pen lying on the pile
is first moved to its holder by the act-level reference program (reference_programs.py), so the pile is clear.
Usage: notebook_gate.py OUT_DIR INDEX   or   notebook_gate.py OUT_DIR summarize   (suite: SMA_GATE_SUITE,
default nbdesk_notebook@14000)"""
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np

from sma.agent.loop import make_env
from sma.env.robot_env import RobotError
from sma.env.scene import PILE_HEIGHT
from sma.env.tasks import suite

out, arg = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
if arg == "summarize":
    rows = [json.load(open(f"{out}/{f}")) for f in sorted(os.listdir(out)) if f.startswith("row_")]
    summary = {"tasks": len(rows), "held": sum(bool(r.get("holding")) for r in rows),
               "camera_check": sum(bool((r.get("camera_check") or {}).get("satisfied")) for r in rows),
               # the notebook's own ground truth (on a desk with other things out of place the desk is not done)
               "gt_success": sum(bool(r.get("notebook_gt")) for r in rows),
               "headings": [r.get("heading_deg") for r in rows], "errors": [(r.get("error") or {}).get("reason") for r in rows]}
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    print("SUMMARY", json.dumps(summary), flush=True)
    sys.exit(0)

task = suite(os.environ.get("SMA_GATE_SUITE", "nbdesk_notebook@14000"))[int(arg)]
nb = private_nb = next(o.name for o in task.layout.objects if o.kind == "notebook" and o.name in task.meta["out_of_place"])
target = task.goal[nb]
row = {"task": task.task_id, "object": nb, "zone": target}
env = make_env(task)


def turn(v, axis, angle):
    """v rotated about a unit axis by angle (Rodrigues)."""
    v, k = np.asarray(v, float), np.asarray(axis, float)
    return v * np.cos(angle) + np.cross(k, v) * np.sin(angle) + k * (k @ v) * (1 - np.cos(angle))


try:
    scene = env.reset()
    seen_as = {}
    if env.neutral:  # colour-coded desk: things taken as what they are, under the names (colours) the camera reports
        from sma.agent.identify import private_truth, role_of
        truth, seen_as = private_truth(env, task, scene)
        env.set_identity({c: {"category": cat, "role": role_of(cat)} for c, cat in truth.items()})
        nb, target = seen_as[nb], seen_as[target]
        scene = env.observe()
    on_pile = [o.name for o in task.layout.objects if o.name in task.meta["out_of_place"] and o.kind == "pen"
               and o.lift >= PILE_HEIGHT]
    if on_pile:  # varied desk: the pen on the pile goes to its holder first
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from reference_programs import PROGRAM
        from sma.agent.sandbox import Sandbox
        pen = on_pile[0]
        env.steps, env.step_budget = 0, 6000
        ok, error = Sandbox(env).run(PROGRAM["pen"].format(obj=seen_as.get(pen, pen),
                                                          zone=seen_as.get(task.goal[pen], task.goal[pen])))
        own = env.private_evaluate({pen: task.goal[pen]})["objects"][pen]
        row["pen_cleared_first"] = {"program_ok": ok, "error": error and (error.get("reason") or error.get("message")),
                                    "gt": bool(own["inside"] and own["resting"] and own["upright"])}
        print("  pen on the pile first:", row["pen_cleared_first"], flush=True)
        scene = env.observe()
    env.steps, env.step_budget = 0, 9000
    table = scene["table_z"]
    obj = scene["objects"][nb]
    zone = scene["zones"][target]
    c = next(c for c in env.grasp_candidates(nb) if c.get("pinch"))
    row["candidate"] = {k: c.get(k) for k in ("x", "y", "height", "approach", "edge_free_below", "drop_below_edge_m", "overhang_m", "score")}
    grip = np.array([c["x"], c["y"], table + c["height"]])
    a0, up = np.array(c["approach"], float), np.array([0.0, 0.0, 1.0])
    away = -1 if a0[1] > 0 else 1  # the turn that points the hand away from the robot base (+x) first
    # until the arm reaches it; fingers closing along +z or -z is the same grasp with the wrist turned half a turn
    tries = [(h, f) for h in (0, 20 * away, 40 * away, 60 * away, -20 * away) for f in (1.0, -1.0)]
    for heading, sign in tries:
        a = turn(a0, up, np.radians(heading))
        poses = (grip, grip - .05 * a, grip + .08 * up, grip - .05 * a + .15 * up)
        if all(env.solve_ik(*p, approach=a, finger_axis=sign * up)["reachable"] for p in poses):
            break
    else:
        raise RobotError("no_reachable_heading")
    row["heading_deg"], row["finger_axis_z"] = heading, sign
    f = sign * up
    env.open_gripper()
    env.move_gripper(*(grip - .05 * a + .15 * up), approach=a, finger_axis=f, path="joint")
    env.move_gripper(*(grip - .05 * a), approach=a, finger_axis=f)
    env.move_by(*(.05 * a))
    row["holding"] = env.close_gripper()["holding_something"]
    env.move_by(dz=.08)
    if not row["holding"]:
        raise RobotError("nothing_held")
    # Where to let go: the notebook's centre was `off` from the grip point (camera, before the grasp), and stays so
    # while the hand keeps its orientation. The gripped edge goes 3.5 cm out over the side of the pile it faces.
    off = np.array(obj["center"][:2]) - grip[:2]
    out_dir = -off / np.linalg.norm(off)  # from the notebook's centre towards the gripped edge
    half_n = np.linalg.norm(off) + c["grip_depth_inside_edge_m"]
    u = np.array([np.cos(zone["long_axis_yaw"]), np.sin(zone["long_axis_yaw"])])
    v = np.array([-u[1], u[0]])
    reach_out = abs(zone["half_size"][0] * (u @ out_dir)) + abs(zone["half_size"][1] * (v @ out_dir))  # pile side
    centre = np.array(zone["center"][:2]) + out_dir * (reach_out - half_n + .035)
    place = np.append(centre - off, table + zone["top_height"] + c["height"] + .004)
    high = table + .26  # the hand reaches 10.4 cm below the grip point: clear of the 11 cm pile and the 12 cm basket
    env.move_gripper(*grip[:2], high, approach=a, finger_axis=f)
    try:
        env.move_gripper(*place[:2], high, approach=a, finger_axis=f)
        row["carry_path"] = "line"
    except RobotError:
        env.move_gripper(*place[:2], high, approach=a, finger_axis=f, path="joint")
        row["carry_path"] = "joint"
    row["lowered"] = env.move_by(dz=-(high - place[2]))["stopped_by_contact"]
    env.open_gripper()
    env.wait(40)
    env.move_by(*(-.06 * a))
    env.move_by(dz=.1)
except Exception as exc:  # a robot error, or the camera missing the pile
    row["error"] = {"reason": getattr(exc, "reason", type(exc).__name__), **getattr(exc, "details", {})}
    print("  error", row["error"], flush=True)
finally:
    env.step_budget = env.steps + 3000
    try:
        row["camera_check"] = env.check_on_zone(nb, target)
    except RobotError as exc:
        row["check_error"] = {"reason": exc.reason, **exc.details}
    row["gt"] = env.private_evaluate(task.goal)
    own = row["gt"]["objects"][private_nb]
    row["notebook_gt"] = bool(own["inside"] and own["resting"] and own["upright"])
    if env.frames:
        imageio.mimwrite(f"{out}/{task.task_id}.mp4", env.frames, fps=20)
    env.close()
json.dump(row, open(f"{out}/row_{task.task_id}.json", "w"), indent=1, default=str)
g = next((v for k, v in (row["gt"].get("objects") or {}).items() if "notebook" in k), {})
print(task.task_id, "pen first", (row.get("pen_cleared_first") or {}).get("gt"), "heading", row.get("heading_deg"),
      "held", row.get("holding"), "carry", row.get("carry_path"),
      "camera", (row.get("camera_check") or {}).get("satisfied"), "GT", row["notebook_gt"],
      {k: g.get(k) for k in ("inside", "resting", "upright", "position")}, "error", (row.get("error") or {}).get("reason"),
      flush=True)

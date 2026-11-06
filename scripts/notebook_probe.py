"""Can the arm pick up an A5 notebook (21 x 14.8 x 1 cm, wider than the 8 cm gripper opens) by pinching an edge that
sticks out over the desk edge or over a pile of textbooks, and drop it flat into a letter tray? Scripted (no LLM); the
simulator's object pose is read only for the report.
Per setup: the edge-pinch candidates from the camera; for the best one, which approach headings and pitches the arm
reaches (IK, plus a static check that the arm touches neither the desk nor the pile) at the grasp, 5 cm back and 8 cm
up; then the first reachable one is executed: swing to 15 cm above the back-off point, down, in along the approach,
close, lift 8 cm, carry over the tray with the hand clear of the desk, open, back off, park; camera check and ground
truth.
Usage: notebook_probe.py OUT_DIR INDEX   or   notebook_probe.py OUT_DIR summarize"""
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation

from sma.agent.loop import make_env
from sma.env.robot_env import RobotError
from sma.env.scene import Layout, ObjectSpec, ZoneSpec
from sma.env.tasks import Task

PILE = .112  # four textbooks
SETUPS = [  # name, notebook (x, y, yaw), pile (x, y) or None, tray centre, how it is put down
    ("right_edge_long", (-0.22, -0.376, 0.0), None, (0.08, 0.14), "tilt"),  # long edge 5 cm past the right desk edge
    ("right_edge_short", (-0.20, -0.345, np.pi / 2), None, (0.08, 0.14), "tilt"),  # short edge 5 cm past it
    ("front_edge_long", (-0.376, -0.25, np.pi / 2), None, (0.08, 0.14), "tilt"),  # 5 cm past the front edge (x = -0.40)
    ("pile_right_drop", (0.02, -0.206, 0.0), (0.02, -0.14), (0.08, 0.14), "drop"),  # on a pile, 5 cm out to the right
    ("pile_right_tilt", (0.02, -0.206, 0.0), (0.02, -0.14), (0.08, 0.14), "tilt"),
    ("pile_left_tilt", (0.02, 0.206, 0.0), (0.02, 0.14), (0.08, -0.15), "tilt"),  # 5 cm out to the left
    ("pile_far_tilt", (0.121, -0.12, np.pi / 2), (0.02, -0.12), (0.06, 0.16), "tilt"),  # 5 cm out away from the robot
]
out, arg = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
if arg == "summarize":
    rows = [json.load(open(f"{out}/{f}")) for f in sorted(os.listdir(out)) if f.startswith("row_")]
    summary = [{"setup": r["setup"], "reachable_variants": r.get("reachable_variants"), "executed": r.get("executed"),
                "lifted_cm": r.get("lifted_cm"), "gt": (r.get("gt") or {}).get("success"),
                "camera": (r.get("camera_check") or {}).get("satisfied"), "error": (r.get("error") or {}).get("reason")}
               for r in rows]
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    for s in summary:
        print("SUMMARY", json.dumps(s), flush=True)
    sys.exit(0)

name, (nx, ny, nyaw), pile, tray_xy, how = SETUPS[int(arg)]
TRAY = ZoneSpec("orange", tray_xy, 0.0, half=(.125, .092))  # letter tray: the notebook with 2 cm to spare each way
nb = ObjectSpec("green", "notebook", (nx, ny), nyaw, lift=PILE if pile else 0.0)
layout = Layout([nb], [TRAY], stacks=[(pile[0], pile[1], PILE, 0.0)] if pile else [])
task = Task(f"notebook_{name}", "notebook_probe", "Put the green notebook on the orange tray.", layout,
            {nb.name: TRAY.name}, 31000 + int(arg))
row = {"setup": name, "object": nb.name, "place": how}
env = make_env(task)


def rotation(approach):
    a = np.asarray(approach, float) / np.linalg.norm(approach)
    f = np.array([0.0, 0.0, 1.0]) - a[2] * a
    return env._rotation_from(a, f / np.linalg.norm(f))


def tilt():
    z = env.private_state()[nb.name]["z_axis"]
    return round(float(np.degrees(np.arccos(min(1.0, abs(z[2]))))), 1)


def touches(q):
    """Static check in the IK model: bodies other than the robot itself and the objects that the arm touches."""
    import mujoco
    m, d = env._model, env._ik_data
    d.qpos[env.qadr] = q
    mujoco.mj_forward(m, d)
    hits = set()
    for i in range(d.ncon):
        con = d.contact[i]
        if con.dist > -.001:
            continue
        b = [m.body(int(m.geom_bodyid[g])).name for g in (con.geom1, con.geom2)]
        robot = [n.startswith(("robot0_", "gripper0_")) for n in b]
        if robot[0] == robot[1]:
            continue
        mine, other = (b[0], b[1]) if robot[0] else (b[1], b[0])
        if mine in ("robot0_base", "robot0_link0", "robot0_link1") or other.startswith(tuple(env.env.items)):
            continue
        hits.add(f"{mine}~{other}")
    return sorted(hits)


try:
    scene = env.reset()
    env.steps, env.step_budget = 0, 9000
    gt0 = env.private_state()[nb.name]
    seen = scene["objects"].get(nb.name)
    row["seen"] = None if seen is None else {k: seen[k] for k in ("kind", "height", "footprint_long", "footprint_short")}
    print(name, "seen", row["seen"], flush=True)
    cands = [c for c in env.grasp_candidates(nb.name) if c.get("pinch")]
    row["pinch_candidates"] = [{k: c.get(k) for k in ("x", "y", "height", "approach", "edge_free_below", "drop_below_edge_m", "overhang_m",
                                                      "hand_clear", "score")} for c in cands]
    for c in row["pinch_candidates"]:
        print("  candidate", json.dumps(c), flush=True)
    c = cands[0]
    table = scene["table_z"]
    grip = np.array([c["x"], c["y"], table + c["height"]])
    a0 = np.array(c["approach"], float)
    variants = []
    for pitch in (0, 20, 35):
        for heading in (0, 20, -20, 40, -40):
            h = np.radians(heading)
            ah = np.array([a0[0] * np.cos(h) - a0[1] * np.sin(h), a0[0] * np.sin(h) + a0[1] * np.cos(h), 0.0])
            a = np.cos(np.radians(pitch)) * ah + np.array([0, 0, -np.sin(np.radians(pitch))])
            rot = rotation(a)
            entry = {"pitch_deg": pitch, "heading_deg": heading, "approach": a.round(3).tolist()}
            poses = {"grasp": grip, "back": grip - .05 * a, "lift": grip + [0, 0, .08], "stage": grip - .05 * a + [0, 0, .15]}
            ok = True
            for key, pos in poses.items():
                q, residual = env._ik_multi(pos, rot)
                entry[key] = {"ik_residual": round(residual, 4)}
                if q is None:
                    ok = False
                    break
                hits = touches(q)
                if hits:
                    entry[key]["touches"] = hits
                    ok = False
                    break
            entry["ok"] = ok
            variants.append(entry)
            print("  variant", json.dumps(entry), flush=True)
    row["variants"] = variants
    row["reachable_variants"] = sum(v["ok"] for v in variants)
    good = [v for v in variants if v["ok"]]
    if good:
        v = good[0]
        a = np.array(v["approach"], float)
        rot = rotation(a)
        row["executed"] = {k: v[k] for k in ("pitch_deg", "heading_deg", "approach")}
        env.open_gripper()
        stage, back = grip - .05 * a + [0, 0, .15], grip - .05 * a
        q, _ = env._ik_multi(stage, rot)
        env._joint_move(q, stage, rot)
        env._line_move(back, rot)
        row["stopped_early"] = env._line_move(grip, rot, tolerance=.004, stop_on_contact=True)
        row["closed"] = env.close_gripper()
        env._line_move(grip + [0, 0, .08], rot, stop_on_contact=True)
        env.wait(20)
        gt1 = env.private_state()[nb.name]
        row["lifted_cm"] = round(100 * (gt1["position"][2] - gt0["position"][2]), 1)
        row["tilt_in_hand_deg"] = round(float(np.degrees(np.arccos(min(1.0, abs(gt1["z_axis"][2]))))), 1)
        row["holding"] = env.gripper_state()
        print("  lifted", row["lifted_cm"], "cm, tilt", row["tilt_in_hand_deg"], "holding", row["holding"], flush=True)
        if row["holding"]["holding_something"]:
            # Carry along straight lines at a fixed orientation (a joint-space swing let the notebook turn in the
            # fingers until it hung vertically). The notebook's centre was `reach` ahead of the grip point.
            ah = np.array([a[0], a[1], 0.0]) / np.linalg.norm(a[:2])
            reach = float(np.hypot(*(np.array(gt0["position"][:2]) - grip[:2])))
            half_n = reach + .02  # grip point 2 cm inside the edge
            high = table + .22
            env._line_move(np.array([*env._ee()[:2], high]), rot, stop_on_contact=True)
            if how == "drop":
                # notebook level, centre over the tray centre, grip point 12.5 cm up (the hand reaches 10.4 cm below)
                release = np.array([*(np.array(TRAY.xy) - ah[:2] * reach), table + .125])
                env._line_move(np.array([*release[:2], high]), rot)
                row["tilt_after_carry_deg"] = tilt()
                env._line_move(release, rot, stop_on_contact=True)
            else:
                # Turn the approach 40 degrees down: the far edge goes down. Lower until it touches the tray floor
                # (the hand's lowest corner stays about 5 cm below the grip point), then let go: it falls flat about
                # that edge. The far edge goes where the flat notebook's far edge should end up.
                pitch = np.radians(40)
                far = (reach + half_n) * np.cos(pitch)  # grip point to far edge, horizontally, when tilted
                above = np.array([*(np.array(TRAY.xy) + ah[:2] * (half_n - far)), high])
                env._line_move(above, rot)
                row["tilt_after_carry_deg"] = tilt()
                a2 = np.cos(pitch) * ah + np.array([0, 0, -np.sin(pitch)])
                rot = rotation(a2)
                env._move_pose(above, rot, residual_tol=.008)
                row["tilt_after_pitch_deg"] = tilt()
                row["stopped_by_contact"] = env._line_move(np.array([*above[:2], table + .03]), rot, tolerance=.004,
                                                           stop_on_contact=True)
                a = a2
            row["tilt_at_release_deg"] = tilt()
            row["release_grip_height"] = round(float(env._ee()[2] - table), 3)
            print("  carried: tilt", row.get("tilt_after_carry_deg"), "at release", row["tilt_at_release_deg"],
                  "grip height", row["release_grip_height"], flush=True)
            env.open_gripper()
            env.wait(40)
            env._line_move(env._ee() - .06 * a, rot, stop_on_contact=True)
            env._line_move(env._ee() + [0, 0, .1], rot, stop_on_contact=True)
except RobotError as exc:
    row["error"] = {"reason": exc.reason, **exc.details}
    print("  error", row["error"], flush=True)
finally:
    env.step_budget = env.steps + 3000
    try:
        env.go_home()
        row["camera_check"] = env.check_on_zone(nb.name, TRAY.name)
    except RobotError as exc:
        row["check_error"] = {"reason": exc.reason, **exc.details}
    row["gt"] = env.private_evaluate(task.goal)
    row["final_state"] = env.private_state()[nb.name]
    if env.frames:
        imageio.mimwrite(f"{out}/{task.task_id}.mp4", env.frames, fps=20)
    env.close()
json.dump(row, open(f"{out}/row_{name}.json", "w"), indent=1, default=str)
print(name, "reachable", row.get("reachable_variants"), "executed", row.get("executed"), "lifted", row.get("lifted_cm"),
      "camera", (row.get("camera_check") or {}).get("satisfied"), "GT", (row.get("gt") or {}).get("success"),
      (row.get("gt") or {}).get("objects"), "error", (row.get("error") or {}).get("reason"), flush=True)

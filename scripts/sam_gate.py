"""SAM2 perception gate (no LLM) on the shared-colour desk (sdesk, SMA_PERCEPTION=sam). Per desk: (1) the things the
camera reports (SAM2 masks with no text or categories, separated and measured by depth; names from the tracker) against
the layout: which of the task's items were found, which things are none of them, what each measures; (2) a picture of
the masks and the two labelled images a model would be shown; (3) with every thing taken as what it is (the private
match standing in for a correct identification), the camera check of every goal on the start state; (4) the
out-of-place paper ball, mug and pen moved by the act-level reference programs (reference_programs.py) through the
sandbox, checking after each that the moved thing kept its name; ground truth at the end, and a video. The notebook is
left to notebook_gate.py.
Usage (SMA_API_LEVEL=act, SMA_PERCEPTION=sam): sam_gate.py OUT_DIR INDEX   or   sam_gate.py OUT_DIR summarize
(suite: SMA_GATE_SUITE, default sdesk@17000)"""
import json
import os
import sys
import time

import cv2
import imageio.v2 as imageio
import numpy as np

from sma.agent.identify import private_truth, role_of
from sma.agent.loop import make_env
from sma.agent.sandbox import API_LEVEL, Sandbox
from sma.env.tasks import suite

assert API_LEVEL == "act", "run with SMA_API_LEVEL=act"
out, arg = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

from reference_programs import ORDER, PROGRAM  # noqa: E402  (scripts/ is on the path)

if arg == "summarize":
    rows = [json.load(open(f"{out}/{f}")) for f in sorted(os.listdir(out)) if f.startswith("row_")]
    subs = [s for r in rows for s in r["subtasks"]]
    summary = {"desks": len(rows), "items": sum(len(r["items"]) for r in rows),
               "items_found": sum(sum(v is not None for v in r["items"].values()) for r in rows),
               "extra_things": sum(len(r["extra_things"]) for r in rows),
               "start_checks": sum(len(r["start_checks"]) for r in rows),
               "start_checks_agree": sum(sum(c["agree"] for c in r["start_checks"].values()) for r in rows),
               "subtasks": len(subs), "program_ok": sum(s["program_ok"] for s in subs),
               "camera_check": sum(s["camera_check"] for s in subs), "gt": sum(bool(s["gt"]) for s in subs),
               "name_kept": sum(bool(s["name_kept"]) for s in subs),
               "observe_seconds": [r["observe_seconds"] for r in rows]}
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    print("SUMMARY", json.dumps(summary), flush=True)
    sys.exit(0)


def mask_picture(env, scene, truth):
    """The top-down image with every thing's mask tinted and outlined, and its name with what it really is (private:
    for a person looking at the result, never shown to a model)."""
    image = env.last_rgb.copy()
    rng = np.random.default_rng(3)
    for name in sorted(scene["things"], key=lambda n: (len(n), n)):
        mask = env._masks.get(name)
        if mask is None:
            continue
        colour = rng.integers(40, 256, 3)
        image[mask] = (.45 * image[mask] + .55 * colour).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image, contours, -1, (255, 255, 255), 1)
        rows, cols = np.nonzero(mask)
        text = f"{name}:{truth.get(name, '?')}"
        org = (int(cols.mean()) - 20, int(rows.mean()))
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, .33, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, .33, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def robot_contacts(env):
    """Private diagnosis: what the arm and hand touch now (simulator contacts, never shown to a model)."""
    m, d = env.env.sim.model._model, env.env.sim.data._data
    hits = set()
    for i in range(d.ncon):
        con = d.contact[i]
        bodies = [m.body(int(m.geom_bodyid[g])).name for g in (con.geom1, con.geom2)]
        robot = [b.startswith(("robot0_", "gripper0_")) for b in bodies]
        if robot[0] != robot[1]:
            mine, other = (0, 1) if robot[0] else (1, 0)
            hits.add(f"{bodies[mine]}~{m.geom(int((con.geom1, con.geom2)[other])).name or bodies[other]}")
    return sorted(hits)


if os.environ.get("SMA_CONTACT_DEBUG"):  # log what blocked a motion or what the hand touched when a grasp missed
    from sma.env.robot_env import RobotError, TabletopEnv
    _follow, _grasp = TabletopEnv._follow, TabletopEnv.grasp

    def follow(self, *a, **k):
        try:
            return _follow(self, *a, **k)
        except RobotError as exc:
            if exc.reason == "motion_blocked":
                print("CONTACT blocked", json.dumps(exc.details), robot_contacts(self), flush=True)
            raise

    def grasp(self, *a, **k):
        result = _grasp(self, *a, **k)
        if not result.get("holding_something"):
            print("CONTACT grasp_missed", json.dumps(a[0] if a else k, default=str)[:600],
                  json.dumps(result, default=str)[:600], robot_contacts(self), flush=True)
        return result

    def close_gripper(self):  # how the fingers close: width per step, grip point height and tilt, what they touch
        ee, tilt = self._ee(), float(np.degrees(np.arccos(min(1.0, -self._hand_rotation()[2, 2]))))
        pens = {n: s for n, s in self.private_state().items() if n.endswith("_pen")
                and np.hypot(*(np.asarray(s["position"][:2]) - ee[:2])) < .1}
        self.grip, hold, widths = 1.0, self._q(), []
        for i in range(25):
            self._step(hold)
            widths.append(self.gripper_state()["width"])
        print("CONTACT closing", json.dumps({"grip_point_above_table": round(float(ee[2] - self._table_z), 4),
                                             "tilt_deg": round(tilt, 1), "yaw": round(self._yaw(), 3),
                                             "grip_point": ee.round(4).tolist(), "pens_near": pens,
                                             "width_every_5_steps": widths[4::5]}), robot_contacts(self), flush=True)
        out = self.gripper_state()
        if self.perception == "sam" and out.get("holding_something"):
            self._note_held()
        return out
    TabletopEnv._follow, TabletopEnv.grasp, TabletopEnv.close_gripper = follow, grasp, close_gripper

task = suite(os.environ.get("SMA_GATE_SUITE", "sdesk@17000"))[int(arg)]
kind_of = {o.name: o.kind for o in task.layout.objects}
env = make_env(task)
row = {"task": task.task_id, "meta": task.meta, "subtasks": []}
try:
    env.reset()
    started = time.time()
    scene = env.observe()
    row["observe_seconds"] = round(time.time() - started, 2)
    truth, seen_as = private_truth(env, task, scene)
    row["items"] = {item: seen_as.get(item) for item in task.meta["category"]}
    row["extra_things"] = sorted(t for t, c in truth.items() if c == "other")
    keep = ("height", "footprint_long", "footprint_short", "shape", "hollow", "rim_height", "inner_floor_height",
            "top_flat", "visible_pixels")
    row["things"] = {t: {"truth": truth[t], **{k: scene["things"][t].get(k) for k in keep}} for t in scene["things"]}
    imageio.imwrite(f"{out}/{task.task_id}_masks.png", mask_picture(env, scene, truth))
    for v in env.labelled_views():
        imageio.imwrite(f"{out}/{task.task_id}_tags_{v['camera']}.png", v["image"])
    print(task.task_id, f"observe {row['observe_seconds']} s;", len(scene["things"]), "things; missing items:",
          [i for i, t in row["items"].items() if t is None], "extra:", row["extra_things"], flush=True)
    for t, d in row["things"].items():
        print(f"   {t:8s} {d['truth']:20s}", {k: d[k] for k in keep}, flush=True)
    # (3) the things taken as what they are; every goal checked on the start state
    env.set_identity({t: {"category": c, "role": role_of(c)} for t, c in truth.items()})
    goals = [(seen_as[o], seen_as[z]) for o, z in task.goal.items() if o in seen_as and z in seen_as]
    checks, _ = env.check_goals(goals)
    row["start_checks"] = {}
    for o, z in task.goal.items():
        name = seen_as.get(o)
        got = bool(checks.get(name, {}).get("satisfied")) if name else None
        row["start_checks"][o] = {"thing": name, "in_place": o not in task.meta["out_of_place"], "camera": got,
                                  "agree": got == (o not in task.meta["out_of_place"])}
        if name and not row["start_checks"][o]["agree"]:
            row["start_checks"][o]["details"] = checks.get(name)
    print(task.task_id, "start checks", {o: (c["in_place"], c["camera"]) for o, c in row["start_checks"].items()}, flush=True)
    for o, c in row["start_checks"].items():
        if c.get("details"):
            print("   disagree", o, json.dumps(c["details"], default=str)[:400], flush=True)
    # (4) the reference programs, with the name of the moved thing followed
    sandbox = Sandbox(env)
    for kind in ORDER:
        for obj in [n for n in task.meta["out_of_place"] if kind_of[n] == kind and n in seen_as]:
            name, zone = seen_as[obj], seen_as.get(task.goal[obj])
            if zone is None:
                row["subtasks"].append({"object": obj, "kind": kind, "error": "receptacle not found", "program_ok": False,
                                        "camera_check": False, "gt": False, "name_kept": None})
                continue
            env.steps, env.step_budget = 0, 6000
            ok, error = sandbox.run(PROGRAM[kind].format(obj=name, zone=zone))
            env.step_budget = env.steps + 3000
            checks, scene = env.check_goals(goals)
            match = env.private_thing_truth(scene)
            gt = env.private_evaluate({obj: task.goal[obj]})["objects"][obj]
            row["subtasks"].append({
                "object": obj, "thing": name, "kind": kind, "zone": zone, "program_ok": ok,
                "error": error and (error.get("reason") or error.get("message")),
                "camera_check": bool(checks.get(name, {}).get("satisfied")),
                "gt": bool(gt["inside"] and gt["resting"] and gt["upright"]),
                # after the move, is the thing under the same name still this object?
                "name_kept": match.get(name) == obj, "now_called": next((t for t, k in match.items() if k == obj), None),
                "all_goals_after": {o: bool(c.get("satisfied")) for o, c in checks.items()},
                "check_details": None if checks.get(name, {}).get("satisfied") else checks.get(name)})
            imageio.imwrite(f"{out}/{task.task_id}_after_{len(row['subtasks'])}_{kind}.png",
                            mask_picture(env, scene, {t: (k or "other") for t, k in match.items()}))
            print(task.task_id, "subtask", json.dumps(row["subtasks"][-1], default=str), flush=True)
    env.step_budget = env.steps + 2000
    row["gt"] = env.private_evaluate(task.goal)
    if env.frames:
        imageio.mimwrite(f"{out}/{task.task_id}.mp4", env.frames, fps=20)
finally:
    env.close()
json.dump(row, open(f"{out}/row_{task.task_id}.json", "w"), indent=1, default=str)

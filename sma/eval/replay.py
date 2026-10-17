"""Replay a recorded episode (same seed, same recorded programs and skills, no LLM) and render a video
with the executing code beside the robot.

MuJoCo is deterministic, so the replay reproduces the recorded run; the ground-truth result of the replay is
compared with the recorded one and written to replay.json.

Usage: python -m sma.eval.replay <run_dir> <out_dir> <condition>/<task> [...]
"""
import json
import shutil
import sys
import textwrap
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from ..agent.loop import ATTEMPT_STEPS
from ..agent.sandbox import Sandbox
from ..env.robot_env import RobotError, TabletopEnv
from ..env.tasks import suite

PANEL_W = 560
WHITE, GREY, YELLOW, GREEN, RED, CYAN = (235, 235, 235), (150, 150, 150), (255, 220, 60), (90, 220, 110), (255, 95, 95), (120, 220, 235)
LABELS = {"none": "no memory", "skill": "skill memory only", "user": "user memory only", "both": "user + skill memory"}


def _num(v):
    return f"{v:.3f}" if isinstance(v, float) else repr(v)


def brief_arg(v):
    """Compact text for an API argument; a grasp candidate shows only what identifies it."""
    if isinstance(v, dict) and {"x", "y", "height", "yaw"} <= set(v):
        return f"candidate#{v.get('id')}(h={v['height']:.3f}, yaw={v['yaw']:.2f})"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}={_num(x)}" for k, x in list(v.items())[:3]) + "}"
    return _num(v)


RESULT_KEYS = ("holding_something", "finger_gap", "satisfied", "inside_zone", "resting", "stopped_by_contact",
               "held_object_bottom_above_table", "occupants", "spots")


def brief_result(call, v):
    """One line saying what an API call returned (shown while the next call runs)."""
    if isinstance(v, list):
        if v and isinstance(v[0], dict) and "fits_open_fingers" in v[0]:
            c = v[0]
            return (f"{len(v)} grasp candidates; #0 {c['source'].split(' (')[0]} h={c['height']:.3f} "
                    f"squeeze={c['squeeze_width']} fits={c['fits_open_fingers']}")
        return f"list of {len(v)}"
    if isinstance(v, dict):
        parts = []
        for k in RESULT_KEYS:
            if k in v:
                x = v[k]
                if k == "spots":
                    parts.append(f"{len(x)} free spots" + (f" (first at {x[0]['x']:.3f}, {x[0]['y']:.3f})" if x else ""))
                else:
                    parts.append(f"{k}={_num(x)}")
        if not parts and "position" in v:
            parts.append("at " + ", ".join(f"{p:.3f}" for p in v["position"]))
        return ", ".join(parts) or "ok"
    return "ok" if v is None else _num(v)


def put(img, text, xy, color=WHITE, scale=.38, thick=1):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class Overlay:
    """Holds what the code panel should show; called for every recorded frame."""

    def __init__(self, condition, task):
        self.condition, self.task = condition, task
        self.header = []
        self.code, self.code_title, self.line = "", "", None
        self.call, self.status, self.status_color = "", "", WHITE
        self.result = ""
        self.skill_codes = {}
        import os
        self.level = {"mid": "mid-level API", "task": "task-level API"}.get(os.environ.get("SMA_API_LEVEL", "task"), "")

    def __call__(self, frame):
        h, w = frame.shape[:2]
        panel = np.full((h, PANEL_W, 3), 22, np.uint8)
        y = 16
        put(panel, f"{self.task.task_id}  |  {LABELS.get(self.condition, self.condition)}  |  {self.level}", (8, y), CYAN, .42)
        for line in textwrap.wrap(f'User: "{self.task.instruction}"', 78)[:2]:
            y += 16
            put(panel, line, (8, y), WHITE, .36)
        for text, color in self.header:
            y += 16
            put(panel, text, (8, y), color, .38)
        y += 18
        put(panel, self.code_title, (8, y), GREY, .36)
        lines = self.code.splitlines() or [""]
        rows = max(6, (h - y - 62) // 13)
        cur = (self.line or 1) - 1
        start = max(0, min(cur - rows // 2, len(lines) - rows))
        for i, text in enumerate(lines[start:start + rows]):
            n = start + i
            y += 13
            if n == cur and self.line:
                cv2.rectangle(panel, (4, y - 10), (PANEL_W - 4, y + 3), (70, 60, 10), -1)
            put(panel, f"{n + 1:>2} {text[:82]}", (8, y), YELLOW if n == cur and self.line else WHITE, .34)
        if self.result:
            put(panel, ("  returned: " + self.result)[:88], (8, h - 43), CYAN, .34)
        put(panel, ("> " + self.call)[:86], (8, h - 26), GREEN, .36)
        put(panel, self.status[:86], (8, h - 9), self.status_color, .38)
        bar = np.zeros((18, w, 3), np.uint8)
        put(bar, f"{self.task.task_id} | {LABELS.get(self.condition, self.condition)}", (4, 13), WHITE, .38)
        return np.hstack([np.vstack([bar, frame[:h - 18]]), panel])

    def on_call(self, entry):
        args = ", ".join([brief_arg(a) for a in entry["args"]] + [f"{k}={brief_arg(v)}" for k, v in entry.get("kwargs", {}).items()])
        self.call = f"{entry['call']}({args})"
        source = entry.get("source") or "<program>"
        if source.startswith("<skill"):
            fn = source[len("<skill "):-1]
            self.code, self.code_title = self.skill_codes.get(fn, ""), f"inside learned skill  skills.{fn}  (code from memory)"
        else:
            self.code, self.code_title = self.program, self.program_title
        self.line = entry.get("line")

    def on_result(self, entry, result):
        if entry["call"] != "observe":
            self.result = f"{entry['call']} -> {brief_result(entry['call'], result)}"


def find_task(task_id):
    for name in ("session", "train", "heldout", "smoke"):
        for task in suite(name):
            if task.task_id == task_id:
                return task
    raise KeyError(task_id)


def replay(run_dir, out_dir, item):
    condition, task_id = item.split("/")
    src = Path(run_dir) / condition / task_id
    record = json.loads((src / "episode.json").read_text())
    memory_path = Path(run_dir) / condition / "memory.json"
    skill_codes = {s["function"]: s["code"] for s in json.loads(memory_path.read_text())["skills"]} \
        if memory_path.exists() else {}
    task = find_task(task_id)
    overlay = Overlay(condition, task)
    overlay.skill_codes = skill_codes
    from ..agent.loop import make_env
    env = make_env(task)
    env.frame_hook = overlay
    out = Path(out_dir) / condition / task_id
    out.mkdir(parents=True, exist_ok=True)
    try:
        env.reset()
        goals = record.get("subgoals", [])
        for k, sub in enumerate(goals):
            for attempt in sub["attempts"]:
                obj, zone = sub["object"], sub["zone"]
                source = (f"direct reuse of learned skill (no new code)" if attempt.get("direct_reuse")
                          else "code written by the LLM for this attempt")
                overlay.header = [(f"Subgoal {k + 1}/{len(goals)}: {obj} -> {zone}", WHITE),
                                  (f"Attempt {attempt['n']}: {source}", GREEN if attempt.get("direct_reuse") else WHITE)]
                if attempt.get("retrieved"):
                    v = attempt["retrieved"][0]
                    overlay.header.append((f"Memory: {v['function']} judged {v['verdict']['verdict'].upper()}", CYAN))
                overlay.program, overlay.program_title = attempt["code"], "Program being executed:"
                overlay.code, overlay.code_title, overlay.line = attempt["code"], "Program being executed:", None
                overlay.call, overlay.status, overlay.status_color, overlay.result = "", "running...", WHITE, ""
                if (attempt.get("error") or {}).get("type") == "rejected":
                    overlay.status, overlay.status_color = "program rejected before running: " + attempt["error"]["message"], RED
                    for _ in range(20):
                        env.frames.append(overlay(env.render("demo", 480, 360)))
                else:
                    sandbox = Sandbox(env, skills={r["function"]: skill_codes[r["function"]] for r in attempt.get("retrieved", [])
                                                   if r["function"] in skill_codes}, on_call=overlay.on_call,
                                      on_result=overlay.on_result)
                    env.steps, env.step_budget = 0, ATTEMPT_STEPS
                    ok, error = sandbox.run(attempt["code"])
                    if error:
                        overlay.status = f"robot/program error: {error.get('reason') or error.get('message')}"
                        overlay.status_color = RED
                overlay.call, overlay.line, overlay.result = "camera check (arm parks, top-down RGB-D)", None, ""
                env.step_budget = env.steps + 800
                check = env.check_on_zone(obj, zone)
                good = check.get("satisfied")
                overlay.status = f"camera check: {obj} {'IS' if good else 'is NOT'} resting in {zone}"
                overlay.status_color = GREEN if good else RED
                for _ in range(15):
                    env.frames.append(overlay(env.render("demo", 480, 360)))
        env.step_budget = env.steps + 2000
        env.go_home()
        result = env.private_evaluate(task.goal)
        overlay.header = [("Episode finished", WHITE)]
        overlay.call, overlay.result = "", ""
        overlay.status = f"ground truth (simulator, never shown to the model): {'SUCCESS' if result['success'] else 'FAIL'}"
        overlay.status_color = GREEN if result["success"] else RED
        for _ in range(30):
            env.frames.append(overlay(env.render("demo", 480, 360)))
    except RobotError as exc:
        result = {"success": False, "harness_error": exc.reason}
    finally:
        imageio.mimwrite(out / "video.mp4", env.frames, fps=20)
        env.close()
    match = result.get("success") == record["private_eval"]["success"]
    (out / "replay.json").write_text(json.dumps({"recorded_success": record["private_eval"]["success"],
                                                 "replayed": result, "matches_recording": match}, indent=1))
    shutil.copy(src / "episode.json", out / "episode.json")
    print(item, "replayed GT", result.get("success"), "recorded", record["private_eval"]["success"], "match", match, flush=True)
    return match


if __name__ == "__main__":
    run, out = sys.argv[1], sys.argv[2]
    for it in sys.argv[3:]:
        replay(run, out, it)

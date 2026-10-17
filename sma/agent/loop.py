"""Online agent loop: plan -> per-subgoal code -> execute -> perception check -> repair; memory hooks."""
import json
import re
import time
from pathlib import Path

import numpy as np

from ..env.robot_env import RobotError, TabletopEnv
from ..env.tasks import make_task
from ..memory.store import literal_coordinates, scene_features, zone_occupants
from . import prompts
from .sandbox import Sandbox, check_source, strip_code, ProgramRejected

MAX_ATTEMPTS = 3
ATTEMPT_STEPS = 900


def parse_plan(text, scene):
    match = re.search(r"\{.*\}", text, re.S)
    steps = json.loads(match.group(0))["steps"] if match else []
    valid = [s for s in steps if s.get("object") in scene["objects"] and s.get("zone") in scene["zones"]]
    return valid, steps


def feedback_text(error, trace, check):
    calls = trace[-8:]
    return json.dumps({"error": error, "last_calls": calls, "subgoal_check_after_program": check}, default=str)


def diagnose(trace, scene_before, check, obj, zone, error=None):
    """Plain-language facts derived from the trace and camera; identical for every condition."""
    facts = []
    if error and error.get("type") in ("rejected", "program_error"):
        facts.append(f"The program did not run as written: {error.get('message')} "
                     f"{error.get('exception', '')} {error.get('location') or ''}".strip() + ".")
    target = scene_before["objects"].get(obj)
    last_pos = None
    closes = 0
    for i, entry in enumerate(trace):
        if entry.get("error"):
            facts.append(f"Call {i} {entry['call']}{tuple(entry['args'])} failed: {entry['error']}.")
        result = entry.get("result")
        if entry["call"] == "move_to" and isinstance(result, dict):
            last_pos = result.get("position")
        if entry["call"] == "pick" and isinstance(result, dict) and target:
            closes += 1
            facts.append(f"pick{tuple(entry['args'])} -> holding_something={result.get('holding_something')}; "
                         f"grip point z {result.get('grasp_point_z')} vs {obj} top_z {target['top_z']} "
                         f"(height {target['height']}, footprint {target['footprint_long']} x "
                         f"{target['footprint_short']}, long_axis_yaw {target['long_axis_yaw']}).")
        if entry["call"] == "close_gripper" and isinstance(result, dict):
            closes += 1
            where = ""
            if last_pos and target:
                where = (f" with the grip point at {last_pos}; {obj} was at xy {target['center'][:2]}"
                         f" with top_z {target['top_z']}, so the grip point was "
                         f"{last_pos[2] - target['top_z']:+.3f} m relative to its top surface")
            facts.append(f"close_gripper (call {i}) -> holding_something={result.get('holding_something')}{where}.")
    if closes == 0 and not (error and error.get("type") in ("rejected", "program_error")):
        facts.append("The program never grasped anything.")
    if check.get("reason") == "not_visible":
        facts.append(f"After the program the camera could not see {obj} or {zone}.")
    elif "offset_from_zone_center" in check:
        off = check["offset_from_zone_center"]
        facts.append(f"After the program {obj} is offset {off} m from the centre of {zone} "
                     f"(the zone half size is {check['zone_half_size']}); inside_zone={check['inside_zone']}, "
                     f"resting={check['resting']}, gripper_holding={check['gripper_holding']}.")
    return "\n".join("- " + f for f in facts)


def used_skills(trace):
    return sorted({t["call"][len("skills."):] for t in trace if t["call"].startswith("skills.")})


def run_subgoal(env, llm, task, obj, zone, memory, log):
    attempts_text, attempts = "", []
    for n in range(1, MAX_ATTEMPTS + 1):
        scene = env.observe()
        skills_text, retrieved = "", []
        if memory is not None:
            retrieved = memory.retrieve(scene, obj, zone)
            if retrieved:
                skills_text = prompts.SKILLS_HEADER + memory.describe(retrieved)
        prompt = prompts.CODER.format(api=prompts.API_DOC, instruction=task.instruction, object=obj, zone=zone,
                                      scene=prompts.scene_text(scene), skills=skills_text, attempts=attempts_text)
        code = strip_code(llm.chat([{"role": "user", "content": prompt}], purpose="code"))
        sandbox = Sandbox(env, skills={s["function"]: s["code"] for s, _ in retrieved})
        env.steps, env.step_budget = 0, ATTEMPT_STEPS
        started = time.time()
        if llm.last_truncated():
            ok, error = False, {"type": "rejected", "message": "program text was cut off at the output limit; write a shorter program"}
        else:
            ok, error = sandbox.run(code)
        env.step_budget = env.steps + 800  # the verifier's own go_home is not charged to the program
        check = env.check_on_zone(obj, zone)
        attempt = {"n": n, "code": code, "program_ok": ok, "error": error, "check": check,
                   "used_skills": used_skills(sandbox.trace), "control_steps": env.steps,
                   "seconds": round(time.time() - started, 1), "trace": sandbox.trace,
                   "retrieved": [{"function": s["function"], "verdict": v} for s, v in retrieved],
                   "features": scene_features(scene, obj, zone)}
        attempts.append(attempt)
        log(f"    attempt {n}: program_ok={ok} check={check.get('satisfied')} skills={attempt['used_skills']} "
            f"error={error and (error.get('reason') or error.get('message'))}")
        if check.get("satisfied"):
            return True, attempts
        attempt["diagnosis"] = diagnose(sandbox.trace, scene, check, obj, zone, error)
        attempts_text = prompts.ATTEMPT.format(n=n, code=code, diagnosis=attempt["diagnosis"],
                                               feedback=feedback_text(error, sandbox.trace, check))
    return False, attempts


def extract_skill(llm, memory, task, obj, zone, attempt, scene, log):
    function = memory.next_function_name()
    text = llm.chat([{"role": "user", "content": prompts.EXTRACT.format(
        object=obj, kind=scene["objects"][obj]["kind"], zone=zone, code=attempt["code"], function=function)}],
        purpose="extract")
    code = strip_code(text)
    conditions = {"goal": {"subgoal": "place_on_zone"},
                  "state": {"requires_zone_free": True, "requires_gripper_empty": True}}
    json_lines = [l for l in text.strip().splitlines() if l.strip().startswith("{")]
    if json_lines:
        try:
            proposed = json.loads(json_lines[-1])
            conditions["state"] = {k: bool(proposed.get(k, True)) for k in ("requires_zone_free", "requires_gripper_empty")}
            conditions["proposed_object_kinds"] = proposed.get("object_kinds", [])
            conditions["notes"] = proposed.get("notes", "")
        except json.JSONDecodeError:
            pass
    try:
        tree = check_source(code)
    except ProgramRejected as exc:
        log(f"    extraction rejected: {exc}")
        return None
    if not any(getattr(n, "name", None) == function for n in tree.body):
        log("    extraction rejected: function not defined")
        return None
    literals = literal_coordinates(code, scene)
    if literals:
        log(f"    extraction rejected: scene coordinates hard-coded {literals}")
        return None
    origin = {"task": task.task_id, "object": obj, "zone": zone, "attempt": attempt["n"],
              "source_code": attempt["code"]}
    memory.add(function, code, conditions, origin)
    memory.record(function, task_id=task.task_id, source="origin", feats=attempt["features"], outcome=True)
    log(f"    new skill {function} (proposed kinds {conditions.get('proposed_object_kinds')})")
    return function


def validate_skill(memory, function, kind, seeds, out_dir, log, video=False):
    """Controlled scene variations: run the stored skill unchanged on new layouts (no LLM)."""
    results = []
    for seed in seeds:
        task = make_task("place", seed, kinds=[kind])
        obj, zone = next(iter(task.goal.items()))
        env = TabletopEnv(task, video=video)
        try:
            scene = env.reset()
            sandbox = Sandbox(env, skills={function: memory.get(function)["code"]})
            env.steps, env.step_budget = 0, ATTEMPT_STEPS
            ok, error = sandbox.run(f'skills.{function}("{obj}", "{zone}")')
            env.step_budget = env.steps + 800
            check = env.check_on_zone(obj, zone)
            gt = env.private_evaluate(task.goal)
        finally:
            env.close()
        failure = None if check.get("satisfied") else (error and (error.get("reason") or error.get("message"))) or "check_failed"
        memory.record(function, task_id=task.task_id, source="validation", feats=scene_features(scene, obj, zone),
                      outcome=check.get("satisfied", False), failure=failure)
        results.append({"seed": seed, "kind": kind, "perception_ok": check.get("satisfied", False),
                        "gt_ok": gt["success"], "error": error})
        log(f"    validation {function} on {kind} seed {seed}: perception={check.get('satisfied')} gt={gt['success']}")
    return results


def run_episode(task, llm, memory, out_dir, *, validate=True, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = TabletopEnv(task)
    record = {"task_id": task.task_id, "family": task.family, "instruction": task.instruction,
              "memory": memory is not None, "subgoals": [], "validations": []}
    started = time.time()
    pending = []
    try:
        scene = env.reset()
        plan_text = llm.chat([{"role": "user", "content": prompts.PLANNER.format(
            instruction=task.instruction, scene=prompts.scene_text(scene))}], purpose="plan")
        plan, raw = parse_plan(plan_text, scene)
        record["plan"], record["raw_plan"] = plan, raw
        log(f"  plan: {plan}")
        for step in plan:
            obj, zone = step["object"], step["zone"]
            log(f"  subgoal {obj} -> {zone}")
            before = env.observe()
            ok, attempts = run_subgoal(env, llm, task, obj, zone, memory, log)
            entry = {"object": obj, "zone": zone, "success": ok, "attempts": attempts,
                     "zone_occupied_at_start": zone_occupants(before, zone, exclude=obj)}
            if memory is not None:
                for a in attempts:  # history for every stored skill that was actually called
                    for fn in a["used_skills"]:
                        verdict = next((r["verdict"] for r in a["retrieved"] if r["function"] == fn), None)
                        memory.applicability_log.append({"task": task.task_id, "function": fn,
                                                         "verdict": verdict and verdict["verdict"],
                                                         "outcome": a["check"].get("satisfied", False)})
                        repaired = ok and a is not attempts[-1]
                        memory.record(fn, task_id=task.task_id, source="use", feats=a["features"],
                                      outcome=a["check"].get("satisfied", False),
                                      failure=None if a["check"].get("satisfied") else
                                      (a["error"] and (a["error"].get("reason") or a["error"].get("message"))) or "check_failed",
                                      repaired_by="later attempt" if repaired else None)
                final = attempts[-1]
                if ok and not final["used_skills"]:
                    fn = extract_skill(llm, memory, task, obj, zone, final, before, log)
                    entry["extracted_skill"] = fn
                    if fn and validate:  # run after this episode's simulator is closed
                        pending.append((fn, before["objects"][obj]["kind"]))
            record["subgoals"].append(entry)
        env.step_budget = env.steps + 2000
        env.go_home()
        record["final_perception"] = {k: env.check_on_zone(o, z) for o, z in task.goal.items() for k in [o]}
        record["private_eval"] = env.private_evaluate(task.goal)
    except RobotError as exc:
        record["harness_error"] = {"reason": exc.reason, "details": exc.details}
        record["private_eval"] = env.private_evaluate(task.goal)
    finally:
        record["seconds"] = round(time.time() - started, 1)
        record["llm_usage"] = llm.usage()
        if env.frames:
            import imageio.v2 as imageio
            imageio.mimwrite(out_dir / "video.mp4", env.frames, fps=20)
        env.close()
        (out_dir / "episode.json").write_text(json.dumps(record, indent=1, default=str))
    for fn, kind in pending:
        seed0 = 5000 + 10 * len(memory.skills)
        record["validations"] += validate_skill(memory, fn, kind, [seed0, seed0 + 1], out_dir, log)
    if pending:
        (out_dir / "episode.json").write_text(json.dumps(record, indent=1, default=str))
    return record

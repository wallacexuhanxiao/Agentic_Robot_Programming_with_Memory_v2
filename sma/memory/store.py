"""Execution-grounded skill memory: code + conditions + execution history, persisted as JSON.

Everything stored here is agent-legal: outcomes come from the perception verifier, never from
simulator ground truth (which is logged separately by the evaluator).
"""
import ast
import json
import math
from pathlib import Path

ROBOT_BASE_XY = (-0.56, 0.0)
REACH_MARGIN = .03


def reach(xy):
    return math.hypot(xy[0] - ROBOT_BASE_XY[0], xy[1] - ROBOT_BASE_XY[1])


def scene_features(scene, obj_name, zone_name):
    obj, zone = scene["objects"].get(obj_name), scene["zones"].get(zone_name)
    if obj is None or zone is None:
        return None
    return {"kind": obj["kind"], "height": obj["height"], "footprint_short": obj["footprint_short"],
            "object_reach": round(reach(obj["center"]), 3), "zone_reach": round(reach(zone["center"]), 3)}


def zone_occupants(scene, zone_name, exclude=None):
    zone = scene["zones"].get(zone_name)
    if zone is None:
        return []
    out = []
    for name, obj in scene["objects"].items():
        if name == exclude:
            continue
        if (abs(obj["center"][0] - zone["center"][0]) <= zone["half_size"][0] and
                abs(obj["center"][1] - zone["center"][1]) <= zone["half_size"][1]):
            out.append(name)
    return out


class SkillMemory:
    def __init__(self, path):
        self.path = Path(path)
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.skills = data.get("skills", [])
        self.grasp_log = data.get("grasp_log", [])  # operation-level experience: how each kind was grasped
        self.lessons = data.get("lessons", [])      # reflections on failures, keyed by object kind
        self.applicability_log = []

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"skills": self.skills, "grasp_log": self.grasp_log,
                                         "lessons": self.lessons}, indent=1))

    def add_lesson(self, kind, lesson, task_id):
        self.lessons.append({"kind": kind, "lesson": lesson, "task": task_id})
        self.save()

    def lessons_text(self, kind, limit=3):
        rows = [l for l in self.lessons if l["kind"] == kind][-limit:]
        return "".join(f"- ({l['task']}) {l['lesson']}\n" for l in rows)

    def log_grasps(self, picks, task_id):
        for p in picks:
            self.grasp_log.append({**p, "task": task_id})
        if picks:
            self.save()

    def grasp_experience_text(self, per_kind=4):
        """Operation-level memory shown to the coder: recent grasp parameters and outcomes per object kind."""
        if not self.grasp_log:
            return ""
        lines = ["", "Grasp experience from earlier executions (grasp_height above table; yaw relative to the object's "
                 "long axis; holding = whether the object ended up between the fingers):"]
        for kind in sorted({g["kind"] for g in self.grasp_log}):
            rows = [g for g in self.grasp_log if g["kind"] == kind][-per_kind:]
            items = "; ".join(f"grasp_height {g['grasp_height']}, yaw {g['yaw_from_long_axis_deg']} deg -> "
                              f"{'holding' if g['holding'] else 'NOT holding'}" for g in rows)
            lines.append(f"- {kind} (height {rows[-1]['object_height']}, footprint {rows[-1]['footprint']}): {items}")
        return "\n".join(lines) + "\n"

    def get(self, function):
        return next((s for s in self.skills if s["function"] == function), None)

    def next_function_name(self, subgoal="place_on_zone"):
        return f"{subgoal}_v{1 + sum(1 for s in self.skills if s['subgoal'] == subgoal)}"

    # --------------------------------------------------------------- applicability
    def check(self, skill, scene, obj_name, zone_name):
        """Three-level check: goal match, state preconditions, physical feasibility (from history)."""
        reasons = []
        feats = scene_features(scene, obj_name, zone_name)
        if feats is None:
            return {"verdict": "not_applicable", "reasons": ["object or zone not visible"]}
        if skill["subgoal"] != "place_on_zone":
            return {"verdict": "not_applicable", "reasons": ["different subgoal"]}
        same_kind = [h for h in skill["history"] if h["features"]["kind"] == feats["kind"]]
        wins = sum(h["outcome"] for h in same_kind)
        losses = len(same_kind) - wins
        state = skill["conditions"].get("state", {})
        if state.get("requires_gripper_empty", True) and scene["gripper"]["holding_something"]:
            reasons.append("state: gripper already holds something")
        occupants = zone_occupants(scene, zone_name, exclude=obj_name)
        if state.get("requires_zone_free", True) and occupants:
            reasons.append(f"state: zone occupied by {occupants}")
        if reasons:
            return {"verdict": "not_applicable", "reasons": reasons, "kind_record": [wins, losses]}
        if losses >= 2 and wins == 0:
            return {"verdict": "not_applicable", "kind_record": [wins, losses],
                    "reasons": [f"goal: failed {losses}x on kind {feats['kind']}, never succeeded"]}
        good = [h["features"] for h in skill["history"] if h["outcome"]]
        within = bool(good) and all(
            min(f[key] for f in good) - REACH_MARGIN <= feats[key] <= max(f[key] for f in good) + REACH_MARGIN
            for key in ("object_reach", "zone_reach"))
        if wins >= 1 and losses <= wins and within:
            return {"verdict": "applicable", "kind_record": [wins, losses],
                    "reasons": [f"goal: kind {feats['kind']} succeeded {wins}/{wins + losses}",
                                "physical: object and zone inside the validated reach range"]}
        detail = [f"goal: kind {feats['kind']} record {wins} success / {losses} failure"]
        if not within:
            detail.append("physical: outside the reach range seen in successful executions")
        return {"verdict": "uncertain", "kind_record": [wins, losses], "reasons": detail}

    def retrieve(self, scene, obj_name, zone_name, limit=2):
        rows = []
        for skill in self.skills:
            if skill["status"] == "rejected":
                continue
            verdict = self.check(skill, scene, obj_name, zone_name)
            rank = {"applicable": 0, "uncertain": 1, "not_applicable": 2}[verdict["verdict"]]
            successes = sum(h["outcome"] for h in skill["history"])
            rows.append((rank, -successes, skill, verdict))
        rows.sort(key=lambda r: (r[0], r[1]))
        return [(skill, verdict) for _, _, skill, verdict in rows[:limit]]

    def describe(self, retrieved, show_code=True):
        lines = []
        for i, (skill, verdict) in enumerate(retrieved):
            by_kind = {}
            for h in skill["history"]:
                k = h["features"]["kind"]
                by_kind.setdefault(k, [0, 0])[0 if h["outcome"] else 1] += 1
            record = ", ".join(f"{k}: {w} ok / {l} failed" for k, (w, l) in sorted(by_kind.items()))
            fails = [h["failure"] for h in skill["history"] if not h["outcome"] and h.get("failure")][-2:]
            lines.append(f"- skills.{skill['function']}(obj_name, zone_name): verdict {verdict['verdict'].upper()} "
                         f"({'; '.join(verdict['reasons'])}). History: {record or 'none'}."
                         + (f" Recent failures: {fails}." if fails else ""))
            if show_code and i == 0:
                lines.append(f"Code of {skill['function']}:\n```python\n{skill['code']}\n```")
        return "\n".join(lines) + "\n"

    # --------------------------------------------------------------- updates
    def record(self, function, *, task_id, source, feats, outcome, failure=None, repaired_by=None):
        skill = self.get(function)
        entry = {"task": task_id, "source": source, "features": feats, "outcome": bool(outcome)}
        if failure:
            entry["failure"] = failure
        if repaired_by:
            entry["repaired_by"] = repaired_by
        skill["history"].append(entry)
        wins = sum(h["outcome"] for h in skill["history"])
        skill["status"] = "validated" if wins >= 2 else ("candidate" if wins else "unverified")
        self.save()

    def add(self, function, code, conditions, origin):
        skill = {"function": function, "subgoal": "place_on_zone", "code": code,
                 "expected_effect": "object resting inside the zone footprint, gripper open, arm lifted",
                 "conditions": conditions, "origin": origin, "history": [], "status": "unverified"}
        self.skills.append(skill)
        self.save()
        return skill


def literal_coordinates(code, scene, tolerance=.002):
    """Numeric literals that equal a scene coordinate: a sign the skill was not parameterized."""
    coords = []
    for item in list(scene["objects"].values()) + list(scene["zones"].values()):
        coords += [c for c in item["center"][:2] if abs(c) > .03]
    found = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            if any(abs(abs(node.value) - abs(c)) < tolerance for c in coords):
                found.append(node.value)
    return found

"""Execution-grounded skill memory: code + conditions + execution history, persisted as JSON.

Everything stored here is agent-legal: outcomes come from the perception verifier, never from
simulator ground truth (which is logged separately by the evaluator).
"""
import ast
import json
import math
from pathlib import Path

from ..env.naming import kind_of, zone_style
from ..env.perception import inside_zone

ROBOT_BASE_XY = (-0.56, 0.0)
REACH_MARGIN = .03
RECENT_USES = 4  # outcomes that count as the skill's recent record on a goal


def reach(xy):
    return math.hypot(xy[0] - ROBOT_BASE_XY[0], xy[1] - ROBOT_BASE_XY[1])


def scene_features(scene, obj_name, zone_name):
    obj, zone = scene["objects"].get(obj_name), scene["zones"].get(zone_name)
    if obj is None:  # a moved object can be seen as another kind (a pen lying on a rim looks taller): match its colour
        obj = next((o for o in scene["objects"].values() if o.get("color") == obj_name.split("_")[0]), None)
    if obj is None or zone is None:
        return None
    return {"kind": kind_of(obj_name, scene), "zone_style": zone_style(zone_name, scene), "height": obj["height"],
            "footprint_short": obj["footprint_short"],
            "object_reach": round(reach(obj["center"]), 3), "zone_reach": round(reach(zone["center"]), 3)}


def zone_occupants(scene, zone_name, exclude=None):
    zone = scene["zones"].get(zone_name)
    if zone is None:
        return []
    out = []
    for name, obj in scene["objects"].items():
        if name == exclude:
            continue
        if inside_zone(obj["center"], zone):
            out.append(name)
    return out


class SkillMemory:
    def __init__(self, path):
        self.path = Path(path)
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.skills = data.get("skills", [])
        self.grasp_log = data.get("grasp_log", [])  # operation-level experience: how each kind was grasped
        self.lessons = data.get("lessons", [])      # reflections on failures, keyed by object kind
        self.helpers = data.get("helpers", [])      # sub-skills: steps of verified programs, composable by name
        for i, lesson in enumerate(self.lessons):
            lesson.setdefault("id", i + 1)
        self.applicability_log = []

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"skills": self.skills, "grasp_log": self.grasp_log,
                                         "lessons": self.lessons, "helpers": self.helpers}, indent=1))

    # --------------------------------------------------------------- helpers (composable sub-skills)
    def helper(self, name):
        return next((h for h in self.helpers if h["name"] == name), None)

    def add_helper(self, name, code, doc, kinds, origin):
        """Store a helper; returns its stored name. Identical code is not stored twice; a different helper with
        the same name gets a version suffix (the caller renames its uses)."""
        old = self.helper(name)
        if old is not None and old["code"].strip() == code.strip():
            old["object_kinds"] = sorted(set(old.get("object_kinds") or []) | set(kinds))  # now also used for these kinds
            self.save()
            return name
        if old is not None:
            base, i = name, 2
            while self.helper(f"{base}_v{i}"):
                i += 1
            name = f"{base}_v{i}"
        self.helpers.append({"name": name, "code": code, "doc": doc, "object_kinds": kinds, "origin": origin,
                             "history": [], "status": "unverified"})
        self.save()
        return name

    def record_helper(self, name, *, task_id, source, outcome, failure=None):
        h = self.helper(name)
        if h is None:
            return
        h["history"].append({"task": task_id, "source": source, "outcome": bool(outcome),
                             **({"failure": failure} if failure else {})})
        wins = sum(e["outcome"] for e in h["history"])
        losses = len(h["history"]) - wins
        h["status"] = ("rejected" if wins == 0 and losses >= 3 else "validated" if wins >= 2 else
                       "candidate" if wins else "unverified")
        self.save()

    def helper_code(self):
        """Source of every usable helper, for the sandbox (programs and skills call them by name)."""
        code = {h["name"]: h["code"] for h in self.helpers if h["status"] != "rejected"}
        if code:  # a helper whose code would not load (e.g. a name clash) must not break every program
            import ast
            for name, src in list(code.items()):
                try:
                    tree = ast.parse(src)
                    if not any(isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body):
                        code.pop(name)
                except SyntaxError:
                    code.pop(name)
        return code

    def helpers_for(self, kind, limit=8):
        rows = [h for h in self.helpers if h["status"] != "rejected"
                and (not h.get("object_kinds") or kind in h["object_kinds"])]
        rows.sort(key=lambda h: (-sum(e["outcome"] for e in h["history"]), len(h["history"])))
        return rows[:limit]

    def describe_helpers(self, helpers):
        lines = []
        for h in helpers:
            import ast
            fn = next((n for n in ast.parse(h["code"]).body if isinstance(n, ast.FunctionDef)), None)
            args = ", ".join(a.arg for a in fn.args.args) if fn else "..."
            wins = sum(e["outcome"] for e in h["history"])
            lines.append(f"- {h['name']}({args}): {h['doc']} [status {h['status']}; used {len(h['history'])}x, "
                         f"{wins} in successful executions]")
        return "\n".join(lines) + "\n"

    def add_lesson(self, kind, lesson, task_id):
        # shown_then: camera-check outcomes of the later attempts whose prompt showed this lesson.
        self.lessons.append({"id": len(self.lessons) + 1, "kind": kind, "lesson": lesson, "task": task_id,
                             "shown_then": []})
        self.save()

    @staticmethod
    def lesson_status(lesson):
        outcomes = lesson.get("shown_then", [])
        if any(outcomes):
            return "an attempt that saw it succeeded"
        if len(outcomes) >= 2:
            return "refuted"
        return "not yet tested" if not outcomes else f"{len(outcomes)} later attempt failed"

    def lessons_for(self, kind, limit=3):
        """Text for the coder and the ids it shows: recent lessons that are not refuted, plus refuted lessons as
        lessons that did not help (shown in two or more later attempts that all failed)."""
        rows = [l for l in self.lessons if l["kind"] == kind]
        live = [l for l in rows if self.lesson_status(l) != "refuted"][-limit:]
        text = "".join(f"- ({l['task']}; {self.lesson_status(l)}) {l['lesson']}\n" for l in live)
        refuted = self.refuted_text(kind, limit)
        if refuted:
            text += ("Lessons that did NOT help (they were shown before two or more attempts that still failed):\n"
                     + refuted)
        return text, [l["id"] for l in live]

    def refuted_text(self, kind, limit=3):
        rows = [l for l in self.lessons if l["kind"] == kind and self.lesson_status(l) == "refuted"][-limit:]
        return "".join(f"- {l['lesson']}\n" for l in rows)

    def record_lesson_outcome(self, ids, success):
        for lesson in self.lessons:
            if lesson["id"] in ids:
                lesson.setdefault("shown_then", []).append(bool(success))
        if ids:
            self.save()

    def log_grasps(self, picks, task_id):
        for p in picks:
            self.grasp_log.append({**p, "task": task_id})
        if picks:
            self.save()

    def grasp_experience_text(self, per_kind=4):
        """Operation-level memory shown to the coder: recent grasp parameters and outcomes per object kind."""
        if not self.grasp_log:
            return ""
        lines = ["", "Grasp experience from earlier executions (grasp_height above table; yaw = fingertip line, given as "
                 "degrees from the object's long axis, so 90 deg means yaw = short_axis_yaw; holding = whether the "
                 "object ended up between the fingers):"]
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
    def check(self, skill, scene, obj_name, zone_name, feasibility=None, task_id=None):
        """Three-level check: goal match, state preconditions, physical feasibility. The physical level uses cuRobo's
        collision-aware reachability when it is available (feasibility), else the reach range of past successes.
        The goal level also weighs recent outcomes: a skill whose last uses on this goal mostly failed (e.g. one
        learned on an empty desk, now used on a crowded one), or that already failed on this object in this task
        (task_id), is at most "uncertain": it is shown as a reference but not called directly."""
        reasons = []
        feats = scene_features(scene, obj_name, zone_name)
        if feats is None:
            return {"verdict": "not_applicable", "reasons": ["object or zone not visible"]}
        if skill["subgoal"] != "place_on_zone":
            return {"verdict": "not_applicable", "reasons": ["different subgoal"]}
        # Goal match: the same kind of object into the same kind of receptacle (a pen laid in a case and a pen stood
        # in a holder are different goals).
        target = f"{feats['kind']} into {feats['zone_style']}"
        history = [h for h in skill["history"] if h.get("features")]  # entries recorded without a view are skipped
        same_kind = [h for h in history if h["features"]["kind"] == feats["kind"]
                     and h["features"].get("zone_style") == feats["zone_style"]]
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
                    "reasons": [f"goal: failed {losses}x on {target}, never succeeded"]}
        doubts = []
        recent = self.recent_record(skill, feats["kind"], feats["zone_style"])
        if not self.reliable(skill, feats["kind"], feats["zone_style"]):
            doubts.append(f"recent: {sum(recent)}/{len(recent)} of its last uses on {target} succeeded "
                          f"(all-time {wins}/{wins + losses})")
        if task_id is not None and any(h.get("task") == task_id and not h["outcome"] for h in same_kind):
            doubts.append("recent: it already failed on this goal earlier in this task")
        good = [h["features"] for h in history if h["outcome"] and h["features"].get("object_reach") is not None]
        in_range = bool(good) and all(
            min(f[key] for f in good) - REACH_MARGIN <= feats[key] <= max(f[key] for f in good) + REACH_MARGIN
            for key in ("object_reach", "zone_reach"))
        physical = {"reach_range": in_range}
        if feasibility is not None:
            physical["curobo"] = {"ok": feasibility["ok"], "failed": feasibility.get("failed", [])}
            if not feasibility["ok"]:
                return {"verdict": "not_applicable", "kind_record": [wins, losses], "physical": physical,
                        "reasons": [f"physical: cuRobo found no collision-free arm pose ({', '.join(feasibility['failed'])})"]}
        within = feasibility["ok"] if feasibility is not None else in_range
        if wins >= 1 and losses <= wins and within and not doubts:
            return {"verdict": "applicable", "kind_record": [wins, losses], "physical": physical,
                    "reasons": [f"goal: {target} succeeded {wins}/{wins + losses}",
                                "physical: collision-free arm poses found by cuRobo" if feasibility is not None else
                                "physical: object and zone inside the validated reach range"]}
        detail = [f"goal: {target} record {wins} success / {losses} failure"] + doubts
        if not within:
            detail.append("physical: outside the reach range seen in successful executions")
        return {"verdict": "uncertain", "kind_record": [wins, losses], "physical": physical, "reasons": detail}

    @staticmethod
    def recent_record(skill, kind, style):
        """Outcomes of the skill's last uses (and validations) on this kind of object into this kind of receptacle."""
        same = [h["outcome"] for h in skill["history"] if (h.get("features") or {}).get("kind") == kind
                and (h.get("features") or {}).get("zone_style") == style]
        return same[-RECENT_USES:]

    def reliable(self, skill, kind, style):
        """False when at least 3 recent outcomes on this goal exist and fewer than half succeeded."""
        recent = self.recent_record(skill, kind, style)
        return not (len(recent) >= 3 and sum(recent) < len(recent) / 2)

    def retrieve(self, scene, obj_name, zone_name, limit=2, feasibility=None, task_id=None):
        rows = []
        for skill in self.skills:
            if skill["status"] in ("rejected", "superseded"):
                continue
            verdict = self.check(skill, scene, obj_name, zone_name, feasibility, task_id)
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
                f = h.get("features") or {}
                k = f"{f.get('kind', '?')} into {f.get('zone_style', '?')}"
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

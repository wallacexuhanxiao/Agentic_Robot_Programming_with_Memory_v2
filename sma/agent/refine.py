"""Prompt refiner: user instruction + retrieved user preferences + current scene -> explicit task spec.

Runs before the code agent. Every goal must cite its source: the current instruction (verbatim
quote), a stored user preference, or an explicit assumption. Assumptions are allowed so the robot
can still act without memory, but they are labelled and counted.
"""
import json
import re

from ..memory.store import zone_occupants

REFINE = """You turn a user's request to a tabletop robot into an explicit task specification for a
robot-programming agent. Do not write robot code.

User request: "{instruction}"
Stored long-term preferences of this user (from earlier conversations): {preferences}
Objects seen by the camera (name, kind, and the zone each currently lies on): {objects}
Zones seen: {zones}

For every object that the request concerns, give its target zone and the source of that decision:
- "instruction": the request itself states it (give a verbatim quote as evidence);
- "user_memory": a stored preference states it (the current request overrides stored preferences);
- "assumption": neither does; you choose, and say why.
Note zones that are occupied by another object, and objects that are already in place.
Reply with ONLY JSON:
{{"goals": [{{"object": "...", "zone": "...", "source": "instruction"|"user_memory"|"assumption",
             "evidence": "..."}}],
  "already_done": ["<object>", ...],
  "constraints": ["<short sentence>", ...],
  "refined_instruction": "<one explicit paragraph for the robot-programming agent: goals, order
                          constraints such as clearing an occupied zone first, and how success is checked>"}}"""


def _json(text):
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0)) if match else {}


def _quoted(evidence, text):
    fragments = [f.strip(" .,'\"").lower() for f in evidence.split("...") if f.strip(" .,'\"")]
    return bool(fragments) and all(f in text.lower() for f in fragments)


def scene_summary(scene):
    objects = []
    for name, obj in scene["objects"].items():
        on = [z for z in scene["zones"] if name in zone_occupants(scene, z)]
        objects.append({"name": name, "kind": obj["kind"], "on_zone": on[0] if on else None})
    return objects


def blocking_objects(spec_goals, scene):
    """Objects lying on a goal's target zone that have no goal of their own (they must be moved first)."""
    targets = {g["zone"] for g in spec_goals}
    moved = {g["object"] for g in spec_goals}
    out = []
    for zone in targets:
        for occupant in zone_occupants(scene, zone):
            if occupant not in moved and not any(g["object"] == occupant for g in spec_goals):
                out.append({"object": occupant, "zone_blocked": zone})
    return out


def refine(llm, instruction, preferences, scene, _note=""):
    raw = llm.chat([{"role": "user", "content": REFINE.format(
        instruction=instruction, preferences=json.dumps(preferences) if preferences else "none",
        objects=json.dumps(scene_summary(scene)), zones=sorted(scene["zones"])) + _note}],
        purpose="refine", max_tokens=1200)
    try:
        spec = _json(raw)
    except json.JSONDecodeError:
        spec = {}
    pref_zone = {p["object"]: p["zone"] for p in preferences}
    goals, dropped = [], []
    from ..memory.user import resolve_place, supported
    for g in spec.get("goals", []):
        g["zone"] = resolve_place(str(g.get("zone", "")), scene)
        obj, zone, source = g.get("object"), g.get("zone"), g.get("source")
        reason = None
        if obj not in scene["objects"] or zone not in scene["zones"]:
            reason = "unknown object or zone"
        elif source == "instruction" and not supported(instruction, scene["objects"][obj]["kind"], zone, scene) \
                and not _quoted(str(g.get("evidence", "")), instruction):
            reason = "the request does not mention this object kind together with this place"
        elif source == "user_memory" and pref_zone.get(obj) != zone:
            reason = "no stored preference says this"
        elif source not in ("instruction", "user_memory", "assumption"):
            reason = "missing source"
        (dropped if reason else goals).append({**g, **({"rejected_because": reason} if reason else {})})
    blocking = blocking_objects(goals, scene)
    if blocking and not _note:
        # One consistency round: a target zone is covered by an object the spec does not move.
        note = ("\n\nYour previous specification had a problem: " + "; ".join(
            f"{b['object']} lies on {b['zone_blocked']}, which another object must go to, but {b['object']} has no goal"
            for b in blocking) + ". Give it a goal (a zone where it belongs) so the zone can be cleared first.")
        again = refine(llm, instruction, preferences, scene, _note=note)
        again["consistency_round"] = {"blocking": blocking}
        return again
    return {"goals": goals, "dropped": dropped, "already_done": spec.get("already_done", []), "blocking_left": blocking,
            "constraints": spec.get("constraints", []),
            "refined_instruction": spec.get("refined_instruction") or instruction,
            "sources": {s: sum(1 for g in goals if g["source"] == s) for s in ("instruction", "user_memory", "assumption")}}

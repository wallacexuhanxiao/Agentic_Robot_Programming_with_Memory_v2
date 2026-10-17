"""Prompts. Every condition (with or without memory) gets the same API description."""
import json

API_DOC = """You control a Franka Panda arm above a table by writing Python. All units are metres and radians,
world frame: x away from the robot base, y to the left, z up. The table surface is at z = table_z.

Functions already available (no imports, no try/except):
observe() -> dict from the top-down RGB-D camera:
  {"table_z": float,
   "objects": {name: {"kind": "cube"|"book"|"cup", "center": [x, y, z], "top_z": float, "height": float,
                      "footprint_long": float, "footprint_short": float, "long_axis_yaw": float}},
   "zones": {name: {"center": [x, y, z], "half_size": [hx, hy]}},
   "gripper": {"width": float, "holding_something": bool},
   "end_effector": {"position": [x, y, z], "yaw": float}}
  The arm can hide objects from the camera; go_home() first for a clear view.
move_to(x, y, z, yaw=None) moves the grip point (the midpoint between the two fingertips) to (x, y, z)
  with the gripper pointing straight down. yaw is the direction (angle in the xy plane) along which the
  fingers close; None keeps the current yaw. Raises an error if the target is unreachable or blocked.
  Moves in a straight line, without collision avoidance.
open_gripper() / close_gripper() -> {"width": float, "holding_something": bool}. Fingers open to 0.08 m.
go_home() moves the arm up and out of the camera view.   wait(steps) holds still (20 steps = 1 s).
Math: the module math is available. Write plain top-level code; a robot error stops the program
and is reported back to you."""

PLANNER = """You are the planning module of a tabletop robot. Instruction: {instruction}

Current scene (from the camera):
{scene}

Decide which object goes to which zone and in which order. A zone that is covered by an object
must be cleared first (move that object to its own destination). Skip objects that are already
resting on their correct zone. Reply with JSON only:
{{"steps": [{{"object": "<object name>", "zone": "<zone name>"}}, ...]}}"""

CODER = """{api}

Overall instruction: {instruction}
Current subgoal: move the object "{object}" so it rests on the zone "{zone}", then release it.
Current scene:
{scene}
{skills}{attempts}
Write the Python program for this subgoal only. End with the gripper open and the arm lifted.
Reply with the code only."""

SKILLS_HEADER = """
Reusable skills from memory (each was extracted from a verified execution; call them as
skills.<function>(...) or adapt their code). Applicability for THIS subgoal was checked:
"""

ATTEMPT = """
Attempt {n} for this subgoal FAILED. Your program:
```python
{code}
```
Execution feedback:
{feedback}
Fix the cause and write a corrected program for the current scene shown above."""

EXTRACT = """A robot program just achieved this subgoal, verified by the camera:
  place object "{object}" (kind: {kind}) on zone "{zone}".
Program:
```python
{code}
```
Rewrite it as ONE reusable function

def {function}(obj_name, zone_name):

that calls observe() itself and computes every position from the observation (no scene-specific
coordinates as literals; numeric offsets such as approach heights are fine). Keep what made it work.
Then, after the code, give a JSON object on the last line with the conditions under which you
expect the function to work, using only these keys:
{{"object_kinds": [...], "requires_zone_free": true|false, "requires_gripper_empty": true|false,
  "notes": "<one sentence>"}}
Format: a ```python block with the function, then the JSON line."""


def scene_text(scene, keep=("kind", "center", "top_z", "height", "footprint_long", "footprint_short",
                            "long_axis_yaw")):
    compact = {"table_z": scene["table_z"],
               "objects": {k: {f: v[f] for f in keep if f in v} for k, v in scene["objects"].items()},
               "zones": scene["zones"], "gripper": scene["gripper"],
               "end_effector": scene["end_effector"]}
    return json.dumps(compact)

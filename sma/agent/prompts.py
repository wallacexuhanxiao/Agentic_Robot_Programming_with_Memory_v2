"""Prompts. Every condition (with or without memory) gets the same API description."""
import json
import os

POSE_API_DOC = """You control a Franka Panda arm above a table by writing Python. All units are metres and radians,
world frame: x away from the robot base, y to the left, z up. The table surface is at z = table_z.

Functions already available (no imports, no try/except):
observe() -> dict from the top-down RGB-D camera:
  {"table_z": float,
   "objects": {name: {"kind": "block"|"book"|"mug", "center": [x, y, z], "top_z": float, "height": float,
                      "footprint_long": float, "footprint_short": float, "long_axis_yaw": float}},
   "zones": {name: {"center": [x, y, z], "half_size": [hx, hy]}},
   "gripper": {"width": float, "holding_something": bool},
   "end_effector": {"position": [x, y, z], "yaw": float}}
  The arm can hide objects from the camera; go_home() first for a clear view.
move_to(x, y, z, yaw=None) moves the grip point (the midpoint between the two fingertips) to (x, y, z)
  with the gripper pointing straight down. yaw is the direction (angle in the xy plane) along which the
  fingers close; None keeps the current yaw. yaw and yaw + pi are the same grasp (symmetric fingers). Raises an error if the target is unreachable or blocked.
  Moves in a straight line, without collision avoidance.
open_gripper() / close_gripper() -> {"width": float, "holding_something": bool}. Fingers open to 0.08 m.
  The fingertips only touch what lies between table_z and the grip point's height, so an object is
  grasped only if, when closing, the grip point is below the object's top_z and the object is between
  the open fingers. holding_something tells whether something ended up between the fingers.
go_home() moves the arm up and out of the camera view.   wait(steps) holds still (20 steps = 1 s).
Math: the module math is available. Write plain top-level code; a robot error stops the program
and is reported back to you.

Generic pick-and-place sequencing (human-provided prior, the same for every task; it does NOT give
object-specific grasp heights or orientations, you must choose those):
1. Open the gripper before approaching an object.
2. Approach from above: first move well above the object, then descend vertically to the grasp pose.
3. After closing, check holding_something, then lift vertically before any sideways motion, and carry
   the object high enough to clear other objects.
4. To place: move above the target, descend vertically until the object is just above the table,
   open the gripper, then lift vertically."""

TASK_API_DOC = """You control a Franka Panda arm above a table by writing short Python. Units: metres and radians;
world frame: x away from the robot base, y to the left, z up. The table surface is at z = table_z.

Only these functions exist (they are already defined; do not import anything, do not use try/except,
do not define helper names that are not used):
observe() -> dict from the top-down RGB-D camera:
  {"table_z": float,
   "objects": {name: {"name": str, "kind": "block"|"book"|"mug", "center": [x, y, z], "top_z": float,
                      "height": float, "footprint_long": float, "footprint_short": float,
                      "long_axis_yaw": float, "short_axis_yaw": float}},
   "zones": {name: {"name": str, "center": [x, y, z], "half_size": [hx, hy]}},   (trays and a toy box)
   "gripper": {"width": float, "holding_something": bool}}
  "height" is the object's height above the table; "top_z" is the absolute z of its top surface.
  long_axis_yaw / short_axis_yaw: directions (angles in the xy plane) of the footprint's long and short
  sides; the footprint measures footprint_long along long_axis_yaw and footprint_short along short_axis_yaw.
pick(object_name, grasp_height, yaw) -> {"holding_something": bool, ...}
  Top-down grasp: opens the fingers, comes down vertically over the object's centre until the grip
  point (midpoint between the fingertips) is at table_z + grasp_height, closes the fingers, then lifts.
  yaw is the direction of the line between the two fingertips, i.e. the direction in which the fingers
  squeeze (an angle in the xy plane; yaw and yaw + pi are the same). The open fingers are 0.08 m apart,
  so the object's extent along yaw must be smaller than 0.08 m, and they only grip what is below the
  grip point.
  grasp_height is measured ABOVE THE TABLE SURFACE (0.02 means 2 cm above table_z), not an absolute z.
  YOU choose grasp_height and yaw for the object.
place(zone_name, dx=0.0, dy=0.0) carries the held object above the zone centre (dx, dy are small offsets
  from the zone centre, normally 0), lowers it
  to just above the table, releases it, lifts and parks the arm. Fails if nothing is held.
go_home() parks the arm out of the camera view.  wait(steps) holds still (20 steps = 1 s).
The module math is available. A robot error stops the program and is reported back to you."""

API_DOC = TASK_API_DOC if os.environ.get("SMA_API_LEVEL", "task") == "task" else POSE_API_DOC

PLANNER = """You are the planning module of a tabletop robot. Task specification: {instruction}
Goals: {goals}

Current scene (from the camera):
{scene}

Order the goals into steps. A zone that is covered by an object must be cleared first (move that
object to its own destination). Skip objects that are already resting on their target zone.
Reply with JSON only:
{{"steps": [{{"object": "<object name>", "zone": "<zone name>"}}, ...]}}"""

CODER = """{api}

Overall instruction: {instruction}
Current subgoal: move the object "{object}" so it rests on the zone "{zone}", then release it.
Current scene:
{scene}
{skills}{attempts}
Write the Python program for this subgoal only, using only the functions listed above (get every
value from observe()). Keep it short with few comments. Reply with the code only."""

SKILLS_HEADER = """
Reusable skills from memory (each was extracted from a verified execution). Applicability for THIS
subgoal was checked. If a skill is APPLICABLE, your whole program should simply call it, e.g.
skills.<function>("<object>", "<zone>"). If it is UNCERTAIN or NOT_APPLICABLE, adapt its code to the
reason given instead of starting from scratch:
"""

ATTEMPT = """
Attempt {n} for this subgoal FAILED. Your program:
```python
{code}
```
What happened (from the robot and camera):
{diagnosis}
Raw execution trace:
{feedback}
Lesson drawn from this failure: {lesson}
Fix the cause and write a corrected program for the current scene shown above. Do not repeat a program
that already failed."""

REFLECT = """A robot program failed to move "{object}" (kind: {kind}) onto "{zone}".
Program:
```python
{code}
```
What happened (measured by the robot and camera):
{diagnosis}
Object as seen by the camera: {object_view}

In ONE sentence, state the concrete change to the pick/place arguments for a {kind} that would fix
this (for example which grasp_height range, or which of long_axis_yaw / short_axis_yaw to use as yaw).
Reply with JSON only:
{{"lesson": "<one sentence>"}}"""

LESSONS_HEADER = """
Lessons learned from earlier failures with this kind of object:
"""

EXTRACT = """A robot program just achieved this subgoal, verified by the camera:
  place object "{object}" (kind: {kind}) on zone "{zone}".
Program:
```python
{code}
```
Rewrite it as ONE reusable function

def {function}(obj_name, zone_name):

that calls observe() itself and computes every argument from the observation (no scene-specific
coordinates as literals; numeric offsets are fine). Keep what made it work.
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

"""Prompts. Every condition (with or without memory) gets the same API description."""
import json
import os

POSE_API_DOC = """You control a Franka Panda arm above a table by writing short Python. Units: metres and radians;
world frame: x away from the robot base, y to the left, z up. The table surface is at z = table_z.

Only these functions exist (they are already defined; do not import anything, do not use try/except):
observe() -> dict from the top-down RGB-D camera:
  {"table_z": float,
   "objects": {name: {"name": str, "kind": "block"|"book"|"mug", "center": [x, y, z], "top_z": float,
                      "height": float, "footprint_long": float, "footprint_short": float,
                      "long_axis_yaw": float, "short_axis_yaw": float}},
   "zones": {name: {"name": str, "center": [x, y, z], "half_size": [hx, hy]}},   (trays and a toy box)
   "gripper": {"width": float, "holding_something": bool},
   "end_effector": {"position": [x, y, z], "yaw": float}}
  "height" is the object's height above the table; "top_z" is the absolute z of its top surface.
  long_axis_yaw / short_axis_yaw: directions (angles in the xy plane) of the footprint's long and short
  sides; the footprint measures footprint_long along long_axis_yaw and footprint_short along short_axis_yaw.
  The arm can hide objects from the camera; go_home() first for a clear view.
move_to(x, y, z, yaw=None) moves the grip point (the midpoint between the two fingertips) to the ABSOLUTE
  position (x, y, z) in a straight line, gripper pointing straight down. yaw is the direction of the line
  between the two fingertips, i.e. the direction in which the fingers squeeze (yaw and yaw + pi are the
  same); None keeps the current yaw. No collision avoidance. Raises an error if unreachable or blocked.
open_gripper() / close_gripper() -> {"width": float, "holding_something": bool}. The open fingers are 0.08 m
  apart; an object is gripped only if its extent along yaw is smaller than that and it lies between the
  fingertips below the grip point. holding_something tells whether something ended up between the fingers.
go_home() moves the arm up and out of the camera view.   wait(steps) holds still (20 steps = 1 s).
The module math is available. A robot error stops the program and is reported back to you.

Generic pick-and-place sequencing (human-provided prior, the same for every task; it does NOT give
object-specific grasp heights or orientations, you must choose those):
1. Open the gripper before approaching an object.
2. Approach from above: first move well above the object, then descend vertically to the grasp pose.
3. After closing, check holding_something, then lift vertically before any sideways motion, and carry
   the object high enough to clear other objects and the walls of containers.
4. To place: move above the target, descend vertically until the object is just above the container
   floor, open the gripper, then lift vertically."""

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
release() opens the fingers where the arm is now, lifts and parks the arm (use it if something is
  still held, e.g. after a failed place).
go_home() parks the arm out of the camera view.  wait(steps) holds still (20 steps = 1 s).
The module math is available. A robot error stops the program and is reported back to you."""

MID_API_DOC = """You control a Franka Panda arm above a table by writing short Python. Units: metres and radians;
world frame: x away from the robot base, y to the left, z up. The table surface is at z = table_z.
"yaw" is always an angle in the xy plane; for the gripper it is the direction in which the two fingers
squeeze (yaw and yaw + pi are the same). The open fingers are 0.08 m apart.

Only these functions exist (they are already defined; do not import anything, do not use try/except):
observe() -> dict from the top-down RGB-D camera:
  {"table_z": float,
   "objects": {name: {"name": str, "kind": "block"|"book"|"mug"|"pen", "center": [x, y, z], "top_z": float,
                      "height": float, "footprint_long": float, "footprint_short": float,
                      "long_axis_yaw": float, "short_axis_yaw": float}},
   "zones": {name: {"name": str, "center": [x, y, z], "half_size": [h_long, h_short],
                    "long_axis_yaw": float}},   (trays, a toy box, a narrow pen case)
   "gripper": {"width": float, "holding_something": bool}}
  "height" is the object's height above the table. A zone's half_size is measured along its long axis
  (direction long_axis_yaw) and across it. The arm can hide objects; go_home() first for a clear view.

grasp_candidates(object_name) -> list of top-down grasp proposals for a visible object, each a dict
  {"id", "object", "source", "x", "y", "height", "yaw", "squeeze_width", "fits_open_fingers", "score", ...}
  (x, y): where the grip point (midpoint between the fingertips) goes; height: grip point height ABOVE the
  table; yaw: finger direction; squeeze_width: the object's measured extent between the fingers;
  source: "contact-graspnet" (learned grasp detector, with a score) or "geometry" (from the footprint).
  The list is not ranked for your task: YOU choose which candidate to use (you may also adjust its
  height or yaw).
grasp(candidate) -> {"holding_something": bool, "finger_gap": float, "expected_squeeze_width": float, ...}
  Opens the fingers, moves above (x, y) at a safe height, descends vertically to the candidate's
  height, closes the fingers and lifts to 0.18 m above the table. finger_gap is the measured gap after
  closing (close to 0 means nothing was caught).
free_spots(zone_name, object_name=None) -> {"zone", "center", "half_size", "long_axis_yaw",
  "occupants": [names], "spots": [{"x", "y", "object_long_axis_yaw", "clearance"}]}
  Positions inside the zone where the object's footprint fits without touching other objects (for the
  held object if object_name is None). object_long_axis_yaw: the direction the object's long axis must
  have at that spot (it fits only in that orientation in a narrow zone). spots is empty if nothing fits;
  occupants lists objects already in the zone. It does not turn anything: a held object turns with the
  gripper (changing the gripper yaw by d turns the object by d), so choosing the carry yaw is up to you.
carry_to(x, y, height=0.18, yaw=None) rises vertically to table_z + height (if lower), then moves
  horizontally to (x, y) at that height; whatever is held comes along. yaw is an ABSOLUTE gripper yaw
  (finger direction); yaw=None keeps the current yaw.
rotate_by(delta) -> {"gripper_yaw_before", "gripper_yaw", "turned_by", "holding_something"}
  Turns the gripper, and the held object with it, by delta radians about the vertical axis, staying in
  place (positive = counter-clockwise seen from above). A held object's long axis changes by the same
  delta. The shorter of delta and delta -/+ pi is executed (same result for a pen or a book).
lower_until_contact() -> {"stopped_by_contact": bool, "held_object_bottom_above_table": float, ...}
  Lowers straight down until the held object's bottom reaches table level, or until something stops it
  earlier (then stopped_by_contact is True, e.g. it landed on a wall or another object). Keeps holding.
open_gripper() / close_gripper() -> {"width": float, "holding_something": bool, ...}: fingers only, no
  motion (open_gripper also reports the gripper_yaw at release).
retreat(dz=0.12) moves straight up by dz.
check(object_name, zone_name) -> {"satisfied": bool, "inside_zone": bool, "resting": bool,
  "offset_from_zone_center": [dx, dy], ...}: camera check after parking the arm.
go_home() parks the arm out of the camera view.  wait(steps) holds still (20 steps = 1 s).
The module math is available. A robot error stops the program and is reported back to you.

Generic sequencing (human-provided prior, the same for every task; it gives no object-specific choice):
grasp_candidates -> choose one -> grasp -> (check holding_something) -> free_spots -> carry_to above the
chosen spot -> (rotate_by if the object must be turned) -> lower_until_contact -> open_gripper -> retreat -> check."""

ASPIRE_API_DOC = """You control a Franka Panda arm above a table by writing Python against the ASPIRE robot API.
Positions are metres in the robot base frame "robot0_base" (x forward, y left, z up); quaternions are wxyz.
These names are already defined (no imports, no try/except):
observe() -> scene summary in the base frame:
  {"frame": "robot0_base", "table_z": float,
   "objects": {name: {"name", "kind": "block"|"book"|"mug", "center": [x, y, z], "top_z", "height",
                      "footprint_long", "footprint_short", "long_axis_yaw", "short_axis_yaw"}},
   "zones": {name: {"name", "center": [x, y, z], "half_size": [hx, hy]}},   (trays and a walled toy box)
   "gripper": {"width", "holding_something"}}
get_robot_state() -> state with .base_frame and .end_effector_poses["primary"] (a Pose: .position,
  .quaternion_wxyz, .frame). At start the gripper points straight down.
localize_object(query, camera_name="topdown", target_frame="robot0_base") -> result with .ok, .segmentation,
  .point_cloud.points (N x 3 array in the base frame), .geometry.pose.position. Queries like "red mug" or
  "orange tray". The only camera is "topdown".
generate_grasps("topdown", segmentation, backend="contact-graspnet", max_candidates=5) -> grasp set with .ok and
  .grasps; each candidate has .pose (a Pose of the grip point, the point between the fingertips), .score, .width_m.
select_grasp(grasps, strategy="top_down") -> the best candidate or None.
Pose(position, quaternion_wxyz, frame) builds a pose.
move_to_pose(pose, tolerance=0.01, max_steps=250, strategy=None) moves the grip point to the pose. With
  strategy=MotionStrategy(ik_solver="curobo", trajectory_planner="curobo") the path is planned collision-free;
  the default uses inverse kinematics plus joint interpolation. A failed motion raises an error.
open_gripper() / close_gripper() -> {"width": float, "holding_something": bool}. Fingers open to 0.08 m.
go_home() parks the arm out of the camera view (call it before looking with observe/localize_object).
The module math is available. A robot error stops the program and is reported back to you.
"""


def aspire_skill_notes():
    """The official ASPIRE skill documents, read from the ASPIRE checkout (not copied into this repository)."""
    import pathlib
    root = pathlib.Path(os.environ.get("SMA_ASPIRE_SKILLS", "/scratch/e0985859/agentic_robot_programming/aspire_qwen/"
                                       "ASPIRE/.claude/skills/iterative-debugging/skills"))
    parts = []
    for name in ("localize.md", "grasp.md"):
        path = root / name
        if path.exists():
            parts.append(path.read_text().replace("agentview", "topdown"))
    return ("\n\nOfficial ASPIRE skill notes (from the ASPIRE repository):\n" + "\n".join(parts)) if parts else ""


_LEVEL = os.environ.get("SMA_API_LEVEL", "task")
API_DOC = {"task": TASK_API_DOC, "pose": POSE_API_DOC, "mid": MID_API_DOC}.get(_LEVEL) or (ASPIRE_API_DOC + aspire_skill_notes())

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
{tried}
In ONE sentence, state the concrete change to the pick/place arguments for a {kind} that would fix
this (for example which grasp_height range, or which of long_axis_yaw / short_axis_yaw to use as yaw).
Reply with JSON only:
{{"lesson": "<one sentence>"}}"""

if _LEVEL == "mid":
    REFLECT = REFLECT.replace(
        "the concrete change to the pick/place arguments for a {kind} that would fix\nthis (for example which "
        "grasp_height range, or which of long_axis_yaw / short_axis_yaw to use as yaw).",
        "the concrete change for a {kind} that would fix this (for example which kind of grasp\ncandidate, "
        "grasp height or finger direction, or how to choose and approach the placement spot).")

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

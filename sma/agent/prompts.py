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
   "objects": {name: {"name": str, "kind": "block"|"book"|"mug"|"pen"|"paper", "center": [x, y, z], "top_z": float,
                      "height": float, "footprint_long": float, "footprint_short": float,
                      "long_axis_yaw": float, "short_axis_yaw": float}},
   "zones": {name: {"name": str, "center": [x, y, z], "half_size": [h_long, h_short],
                    "long_axis_yaw": float}},   (the receptacles: places things are put in or on)
   "gripper": {"width": float, "holding_something": bool}}
  "height" is the object's height above the table. A zone's half_size is measured along its long axis
  (direction long_axis_yaw) and across it. The arm can hide objects; go_home() first for a clear view.

grasp_candidates(object_name) -> list of top-down grasp proposals for a visible object, each a dict
  {"id", "object", "source", "x", "y", "height", "yaw", "squeeze_width", "fits_open_fingers", "score", ...}
  (x, y): where the grip point (midpoint between the fingertips) goes; height: grip point height ABOVE the
  table; yaw: finger direction; squeeze_width: the object's measured extent between the fingers;
  source: "contact-graspnet" (learned grasp detector, with a score) or "geometry" (from the footprint).
  A contact-graspnet candidate also gives the network's own 6-DoF grasp: "tilt_deg" (its approach, degrees from
  straight down), "approach" and "finger_axis" (unit vectors, world frame). grasp() always descends vertically to
  its x, y, height with fingers along yaw; to use the tilted approach instead, move there yourself (e.g.
  move_gripper to a point back along the approach with that approach and finger_axis, path="curobo" when cuRobo
  is running, then move_by along the approach, close_gripper(), and lift).
  Each candidate also repeats the object's "object_center", "object_long_axis_yaw" and "object_footprint".
  "support_height" (only when the object lies on something above the table, e.g. a book): the height above the
  table of the surface it lies on, seen around it; the candidate heights already allow for it.
  The list is not ranked for your task: YOU choose which candidate to use (you may also adjust its
  height or yaw). Geometry candidates carry a fixed score (0.5 across the short side, 0.3 across the long side).
  "hand_clear" / "fingers_clear": whether the camera's depth points show something else (not this object) where
  the hand body (about 0.21 m along the finger direction, 0.065 m across, from 3 cm above the grip point) or the
  open fingers would go; "highest_obstacle_under_hand_m" is its height. A blocked grasp usually stops the arm
  (motion_blocked): prefer a clear candidate, e.g. one with the fingers turned by 90 degrees (a round object
  can be grasped at any yaw).
grasp(candidate) -> {"holding_something": bool, "finger_gap": float, "expected_squeeze_width": float,
  "object_in_hand": {"source", "centre_offset_from_grip_point_m", "object_long_axis_minus_gripper_yaw_deg"}, ...}
  Opens the fingers, moves above (x, y) at a safe height, descends vertically to the candidate's
  height, closes the fingers and lifts to 0.18 m above the table. The fingertips reach about 9 mm below the grip
  point, so a height under 0.011 m is raised to 0.011 m (reported as "grip_height_raised_to") to keep them off
  the table. finger_gap is the measured gap after
  closing (close to 0 means nothing was caught). object_in_hand: how the object sits in the hand, from the
  object's pose seen just before the grasp ("camera" if that was not available).
place_object(object_name, x, y, object_yaw=None) -> {"measured": {"x", "y", "object_long_axis_yaw", ...},
  "position_error_m", "yaw_error_deg", "holding_something", ...}
  Puts the held object down with its CENTRE at (x, y) and its LONG AXIS at object_yaw (an object yaw, not
  a gripper yaw; None keeps the current orientation; ignored for objects without a long axis). It computes
  the gripper pose from object_in_hand, carries at 0.18 m, lowers until contact, opens, retreats, parks
  and reports where the camera sees the object afterwards.
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
  COMMON MISTAKE: yaw is the direction of the FINGERS, not of the held object. An object grasped across
  its short side (fingers squeezing its width) has its long axis 90 degrees away from the finger
  direction, so carry_to(..., yaw=<the direction the object should have>) leaves it crosswise. To give
  a held object's long axis the direction d, either
    (a) keep yaw=None and call rotate_by(d - <the object's current long-axis yaw>), or
    (b) pass yaw = d + (<gripper yaw at the grasp> - <object's long-axis yaw at the grasp>)
  (both angles at the grasp are in the candidate you grasped: "yaw" and "object_long_axis_yaw"), or
    (c) use place_object(object_name, x, y, d), which takes the object's yaw directly.
rotate_by(delta) -> {"gripper_yaw_before", "gripper_yaw", "turned_by", "holding_something"}
  Turns the gripper, and the held object with it, by delta radians about the vertical axis, staying in
  place (positive = counter-clockwise seen from above). A held object's long axis changes by the same
  delta. The shorter of delta and delta -/+ pi is executed (same result for an object that looks the same turned
  half a turn).
lower_until_contact() -> {"stopped_by_contact": bool, "held_object_bottom_above_table": float, ...}
  Lowers straight down until the held object's bottom reaches table level, or until something stops it
  earlier (then stopped_by_contact is True, e.g. it landed on a wall or another object). Keeps holding.
  With a tilted hand it goes down until something stops it (held_object_bottom_above_table is None).
open_gripper() / close_gripper() -> {"width": float, "holding_something": bool, ...}: fingers only, no
  motion (open_gripper also reports the gripper_yaw at release).
retreat(dz=0.12) moves straight up by dz.
check(object_name, zone_name) -> {"satisfied": bool, "inside_zone": bool, "resting": bool,
  "offset_from_zone_center": [dx, dy], ...}: camera check after parking the arm.
go_home() parks the arm out of the camera view.  wait(steps) holds still (20 steps = 1 s).
The module math is available. A robot error stops the program and is reported back to you.

Generic sequencing (human-provided prior, the same for every task; it gives no object-specific choice):
grasp_candidates -> choose one -> grasp -> (check holding_something) -> free_spots -> place_object at the chosen
spot (or, for finer control: carry_to above it -> rotate_by if the object must be turned -> lower_until_contact ->
open_gripper -> retreat) -> check."""

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


ACT_EXTRA = """Free gripper orientation (grasp always comes from above; carry_to, rotate_by and lower_until_contact keep the
fingers pointing down, except with a hand tilted more than 45 degrees from pointing down, e.g. after a side grasp: then
they keep its orientation (carry_to refuses a yaw; rotate_by turns the whole hand, and what it holds, about the vertical)
and report "orientation_kept"):
gripper_pose() -> {"position": [x, y, z], "approach": [ax, ay, az], "finger_axis": [fx, fy, fz],
  "side": [sx, sy, sz], "yaw", "holding_something"}: unit vectors in the world frame. approach is the direction
  the fingers point ([0, 0, -1] when pointing down), finger_axis the line along which they close, side =
  approach x finger_axis. A long object held across its width lies along side.
move_gripper(x, y, z, approach=[0, 0, -1], finger_axis=None, path="line") -> gripper_pose()
  Moves the grip point to (x, y, z) with the fingers pointing along approach and closing along finger_axis
  (must be perpendicular to approach; None keeps the current finger line where possible). Whatever is held
  turns with the gripper. path: "line" = straight line with the orientation turned on the way; "joint" = joint-
  space motion to an inverse-kinematics solution, no straight-line constraint (the hand and a held object swing
  on the way: use it well above obstacles); "curobo" = collision-aware plan from cuRobo against the camera's
  depth points (only when cuRobo is running; a held object is not modelled).
move_by(dx=0, dy=0, dz=0) -> {"stopped_by_contact": bool, ...gripper_pose()}: straight relative move keeping the
  gripper's orientation; stopping early on contact is reported, not raised.
solve_ik(x, y, z, approach=[0, 0, -1], finger_axis=None, collision_check=False) -> {"reachable": bool,
  "ik_residual_m": float, "collision_free": bool or None}: whether the arm can reach that grip pose; nothing
  moves. Not every orientation is reachable everywhere: check before committing to one.
retreat(dz) also keeps the current orientation. The robot base is at about (x=-0.56, y=0) in the world frame.
The hand body reaches about 0.104 m to either side of the grip point along the finger line and 0.032 m across it, and
lies between 3 and 9 cm behind the grip point along the approach; the fingertips reach about 9 mm past the grip point.
Zones also report what the camera measured of them: "rim_height" (their walls or rim above the table) and, for one
with a flat top, "top_height" (where things lying on it rest).
check(object_name, zone_name) decides from the camera's geometry alone: the object's centre inside the zone and the
object resting at the height things rest at there; for a narrow receptacle with walls more than 6 cm high, whether the
object stands up in it ("standing", "inside_zone").
For an object wider than the open fingers, grasp_candidates() lists "edge pinch" candidates first, one per edge: a side
grasp, the fingers pointing horizontally at that edge ("approach") and closing on its top and bottom faces
("finger_axis" vertical), the grip point "grip_depth_inside_edge_m" inside the edge. "edge_free_below" is whether the
camera sees free space under that edge; "drop_below_edge_m" (how far below the object's bottom the surface beyond the
edge is) and "overhang_m" (how far it sticks out past what it lies on) are given when the camera can measure them. For
the best edge, "reachable_checked" means the approach was turned about the vertical ("approach_turned_deg") and the
finger axis chosen until the grasp pose, the point 5 cm back along the approach and the point 8 cm above the grasp
have inverse-kinematics solutions (not checked for collisions). grasp() cannot execute a side grasp: it always comes
from above.
"""


# Generic examples of the two ways of taking hold of something (the user allows few-shot examples, 2026-11-06; nothing
# in them names a desk, a task or a particular object).
ACT_EXAMPLES = """Examples (generic: adapt the names and numbers to the scene you see).
1) The usual way: a top-down grasp, then set the object down.
    cands = [c for c in grasp_candidates("OBJ") if c["fits_open_fingers"] and not c.get("pinch")]
    clear = [c for c in cands if c.get("hand_clear", True) and c.get("fingers_clear", True)] or cands
    c = max(clear, key=lambda c: c["score"] or 0)
    if not grasp(c)["holding_something"]:
        raise RuntimeError("grasp missed")
    zone = observe()["zones"]["ZONE"]
    carry_to(zone["center"][0], zone["center"][1], 0.2)  # higher than anything on the way
    lower_until_contact()
    open_gripper()
    retreat()
    result = check("OBJ", "ZONE")
2) When no top-down grasp can hold it (nothing fits between the open fingers, e.g. a flat object wider than 8 cm),
   take it from the side by an edge that has free space below it, with an "edge pinch" candidate:
    side = [c for c in grasp_candidates("OBJ") if c.get("pinch") and c.get("edge_free_below")]
    c = next((c for c in side if c.get("reachable_checked")), side[0])
    a, f = c["approach"], c["finger_axis"]
    x, y, z = c["x"], c["y"], observe()["table_z"] + c["height"]  # the grip point
    open_gripper()
    move_gripper(x - 0.05 * a[0], y - 0.05 * a[1], z + 0.15, a, f, path="joint")  # above the point 5 cm back
    move_gripper(x - 0.05 * a[0], y - 0.05 * a[1], z, a, f)  # straight down to it, beside the edge
    move_by(0.05 * a[0], 0.05 * a[1], 0.0)  # slide in along the approach: the edge goes between the fingers
    if not close_gripper()["holding_something"]:
        raise RuntimeError("side grasp missed")
    move_by(dz=0.08)
    # Keep this orientation and move along straight lines (move_gripper with the same a, f). The hand now reaches
    # about 0.10 m below the grip point: carry high enough over everything, and lower it (move_by down stops on
    # contact) where the hand can hang past the side of what the object lands on; then open_gripper(), back out
    # along -approach with move_by, and lift.
"""


def _act_api_doc():
    """The mid-level document without the object-pose placement helper, plus free gripper orientation."""
    doc = MID_API_DOC
    doc = doc[:doc.index("place_object(object_name, x, y, object_yaw=None)")] + doc[doc.index("carry_to(x, y, height=0.18"):]
    doc = doc.replace("""  (both angles at the grasp are in the candidate you grasped: "yaw" and "object_long_axis_yaw"), or
    (c) use place_object(object_name, x, y, d), which takes the object's yaw directly.""",
                      """  (both angles at the grasp are in the candidate you grasped: "yaw" and "object_long_axis_yaw").""")
    doc = doc.replace("""grasp_candidates -> choose one -> grasp -> (check holding_something) -> free_spots -> place_object at the chosen
spot (or, for finer control: carry_to above it -> rotate_by if the object must be turned -> lower_until_contact ->
open_gripper -> retreat) -> check.""", """grasp_candidates -> choose one -> grasp -> (check holding_something) -> free_spots -> carry_to above the
chosen spot -> (rotate_by if the object must be turned) -> lower_until_contact -> open_gripper -> retreat -> check.
This prior covers laying an object flat; for other goals compose the motion yourself with the functions above.""")
    doc = doc.replace("The module math is available. A robot error stops the program and is reported back to you.",
                      ACT_EXTRA + ACT_EXAMPLES +
                      "The module math is available. A robot error stops the program and is reported back to you.")
    assert "place_object" not in doc and "move_gripper" in doc
    return doc


_LEVEL = os.environ.get("SMA_API_LEVEL", "task")
API_DOC = {"task": TASK_API_DOC, "pose": POSE_API_DOC, "mid": MID_API_DOC}.get(_LEVEL) or \
    (_act_api_doc() if _LEVEL == "act" else ASPIRE_API_DOC + aspire_skill_notes())

NEUTRAL_SCENE = """On this desk {naming}; nothing tells you what a
thing is except the camera images and measurements you were shown at the start, where you said what each thing is.
observe() reports the things you called objects under "objects" and the ones you called receptacles under "zones",
each with your "category" and the camera's measurements: "height" (top above the table), "shape" (round / rectangular /
irregular), "hollow" (its middle at least 2 cm below its rim, with "rim_height" and "inner_floor_height"), "top_flat".
A receptacle's "half_size" is inside its walls when it is hollow; things rest on its "inner_floor_height" (hollow) or
its "top_height" (flat). Things you called neither are under "others". check(object, zone) is the camera's verdict:
for a narrow hollow receptacle more than 6 cm tall, whether the object stands up in it ("standing", "inside_zone");
for another hollow one, whether the object is down inside it; for a flat one, whether it lies flat on top.
"""


NAMING = {"color": 'the camera names every thing by its colour only (e.g. "teal")',
          "sam": 'the camera names every separate thing it sees by a plain label ("thing1", "thing2", ...: the order '
                 'says nothing), which it keeps while the thing moves or is hidden by the arm'}


def api_doc(neutral=False, perception="color"):
    """The code agent's API document; on the colour-coded desk (or with SAM perception) without any name or kind that
    would say what a thing is (that comes from the model's own identification)."""
    if not neutral:
        return API_DOC
    scene = NEUTRAL_SCENE.format(naming=NAMING["sam" if perception == "sam" else "color"])
    doc = API_DOC.replace("The module math is available. A robot error stops the program and is reported back to you.",
                          scene + "The module math is available. A robot error stops the program and is reported back to you.")
    doc = doc.replace('"objects": {name: {"name": str, "kind": "block"|"book"|"mug"|"pen"|"paper",',
                      '"objects": {name: {"name": str, "category": <what you identified>,'
                      + ('' if perception == "sam" else ' "color": str,'))
    doc = doc.replace('"long_axis_yaw": float}},   (the receptacles: places things are put in or on)',
                      '"long_axis_yaw": float, "category": <what you identified>, ...}},')
    return doc

PLANNER = """You are the planning module of a tabletop robot. Task specification: {instruction}
Goals: {goals}

Current scene (from the camera):
{scene}
What is in each zone now (from the camera): {occupants}

Order the goals into steps. A zone that is covered by an object must be cleared first (move that
object to its own destination). Skip objects that are already resting on their target zone.
Reply with JSON only:
{{"steps": [{{"object": "<object name>", "zone": "<zone name>"}}, ...]}}"""

REPLANNER = """You are the planning module of a tabletop robot, re-planning during execution. Task: {instruction}
Goals achieved so far (checked by the camera just now): {done}
Goals still to do: {remaining}
Last step: {last}
Earlier failures: {failures}
What is in each zone now (from the camera): {occupants}
Current scene:
{scene}

Order the remaining goals into steps. A zone that holds an object which does not belong there must be cleared
first (move that object to its own destination). An achieved goal that was disturbed is listed as still to do.
A goal that failed may be retried after other steps if something it depends on has changed; otherwise put it
last. Reply with JSON only:
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

HELPERS_HEADER = """
Helper functions from memory: steps of earlier programs that were verified by the camera. They are already
defined; call them directly by name (not via skills.). Compose them where they fit instead of rewriting those steps:
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

if _LEVEL in ("mid", "act"):
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
Rewrite it as reusable, composable code:
1. Split it into 2-4 helper functions, each one step that makes sense on its own and could be reused in other
   tasks (for example: choosing and executing a grasp; turning a held object to an orientation; carrying and
   placing or inserting a held object at a spot). Give each a descriptive snake_case name, parameters instead
   of scene values, and a one-line docstring saying what must hold before it and what holds after it.
   Helpers already in memory, which you may call instead of rewriting them:
{helpers}
2. Then write ONE function

def {function}(obj_name, zone_name):

that calls observe() itself and composes the helpers to achieve the subgoal.
Compute every argument from observations (no scene-specific coordinates as literals; numeric offsets are fine).
Keep what made it work. Choices that can fail in another layout (a reach or orientation that worked here) should
be checked and have alternatives, as the program did or as you can add (e.g. try several candidates or approach
directions, checking solve_ik when it is available).
Then, after the code, give a JSON object on the last line with the conditions under which you expect the
function to work, using only these keys:
{{"object_kinds": [...], "requires_zone_free": true|false, "requires_gripper_empty": true|false,
  "notes": "<one sentence>"}}
Format: one ```python block with the helper functions and the function, then the JSON line."""


def scene_text(scene, keep=("kind", "center", "top_z", "height", "footprint_long", "footprint_short",
                            "long_axis_yaw")):
    compact = {"table_z": scene["table_z"],
               "objects": {k: {f: v[f] for f in keep if f in v} for k, v in scene["objects"].items()},
               "zones": scene["zones"], "gripper": scene["gripper"],
               "end_effector": scene["end_effector"]}
    return json.dumps(compact)

"""Act-level reference programs for the realistic desk (paper ball, mug, book, pen), written only with the functions the
model gets; run through the sandbox with .format(obj=..., zone=...). Used by realdesk_gate.py and sam_gate.py."""

GRASP = '''
cands = grasp_candidates("{obj}")
fit = [k for k in cands if k["fits_open_fingers"]]
clear = [k for k in fit if k.get("hand_clear", True) and k.get("fingers_clear", True)] or fit
c = max(clear, key=lambda k: k["score"] or 0)
g = grasp(c)
if not g["holding_something"]:
    raise RuntimeError("grasp missed")
'''

# Paper ball: carry it above the wastebasket, higher than its rim, and let go.
PAPER = GRASP + '''
bin_ = observe()["zones"]["{zone}"]
rim = bin_["rim_height"] or 0.12
carry_to(bin_["center"][0], bin_["center"][1], rim + 0.08)
open_gripper()
retreat(0.05)
go_home()
result = check("{obj}", "{zone}")
'''

# Where the hand is, as a rectangle on the table plan, and how far it keeps from anything tall (used to choose among
# grasps or hand directions that are otherwise equally good). The scene is the one observed before the grasp.
ROOM = '''


def room(cx, cy, u, half_u, half_v, floor, skip=()):
    # Distance from the rectangle (centre cx, cy; half_u along the unit vector u, half_v across it) to the nearest
    # footprint point of a thing taller than floor (m above the table) not named in skip; negative: they overlap.
    nearest = 1.0
    for name, t in scene.get("things", scene["objects"]).items():
        if name in skip or t["height"] < floor:
            continue
        e = [math.cos(t["long_axis_yaw"]), math.sin(t["long_axis_yaw"])]
        for i in (-1, -0.5, 0, 0.5, 1):
            for j in (-1, 0, 1):
                px = t["center"][0] + i * t["footprint_long"] / 2 * e[0] - j * t["footprint_short"] / 2 * e[1] - cx
                py = t["center"][1] + i * t["footprint_long"] / 2 * e[1] + j * t["footprint_short"] / 2 * e[0] - cy
                nearest = min(nearest, max(abs(px * u[0] + py * u[1]) - half_u, abs(py * u[0] - px * u[1]) - half_v))
    return nearest


'''

# Mug onto the coaster, notebook onto the tray: carry over the centre, higher than the tallest receptacle's rim (the
# straight path may pass over it), lower until it rests, let go. A round object can go down with the fingers in any
# direction: the one that keeps the hand (+-10.5 cm along the fingers, +-3.3 cm across, its body from 3 cm above the
# grip point) furthest from anything tall around the spot, which would stop it on the way down.
SET_DOWN = '''
import math
scene = observe()
zone = scene["zones"]["{zone}"]
''' + ROOM + GRASP + '''
zx, zy = zone["center"][0], zone["center"][1]
yaw = None
if c["object_footprint"][0] - c["object_footprint"][1] < 0.01:
    floor = (zone.get("top_height") or 0) + c["height"] + 0.03
    turns = [c["yaw"] + k * math.pi / 4 for k in range(4)]
    best = max(turns, key=lambda y: room(zx, zy, [math.cos(y), math.sin(y)], 0.105, 0.033, floor, ("{obj}", "{zone}")))
    if best != c["yaw"]:
        yaw = best
tallest = max(z.get("rim_height") or 0 for z in scene["zones"].values())
carry_to(zx, zy, min(0.3, max(0.2, tallest + 0.1)), yaw)
lower_until_contact()
open_gripper()
retreat()
go_home()
result = check("{obj}", "{zone}")
'''

# Pen: the act-level pen-holder reference program (scripts/act_gate.py). It is gripped 3.5 cm from its centre (most of
# it hangs below the fingers when it is stood up) toward the end where the hand keeps furthest from anything tall,
# and stood up with the hand turned, among the reachable directions, to the one that keeps furthest from anything
# tall around the holder (the holder itself and what stands in it aside).
PEN = '''
import math
scene = observe()
table = scene["table_z"]
holder = scene["zones"]["{zone}"]
pen = scene["objects"]["{obj}"]
''' + ROOM + '''
cands = grasp_candidates("{obj}")
c = [k for k in cands if k["source"].startswith("geometry") and k["fits_open_fingers"]][0]
a = c["object_long_axis_yaw"]
f = [math.cos(c["yaw"]), math.sin(c["yaw"])]
shift = max((0.035, -0.035), key=lambda d: room(c["x"] + d * math.cos(a), c["y"] + d * math.sin(a), f, 0.105, 0.033,
                                                c["height"] + 0.03, ("{obj}",)))
g = grasp(dict(c, x=c["x"] + shift * math.cos(a), y=c["y"] + shift * math.sin(a)))
if not g["holding_something"]:
    raise RuntimeError("grasp missed")
pose = gripper_pose()
side = pose["side"]
centre = c["object_center"]
toward = (centre[0] - pose["position"][0]) * side[0] + (centre[1] - pose["position"][1]) * side[1]
s = 1 if toward > 0 else -1
hang = pen["footprint_long"] / 2 + abs(toward)
hx, hy = holder["center"][0], holder["center"][1]
rim = holder["rim_height"] or 0.1
high = table + rim + hang + 0.12
low = table + 0.004 + 0.008 + hang
reach = max(holder["half_size"]) + 0.01
inside = [n for n, t in scene.get("things", scene["objects"]).items()
          if abs(t["center"][0] - hx) < reach and abs(t["center"][1] - hy) < reach]
best = None
for turn in (0, 30, -30, 60, -60, 90, -90):
    h = math.atan2(hy, hx + 0.56) + math.radians(turn)
    approach = [math.cos(h), math.sin(h), 0.0]
    up = [0.0, 0.0, -s]
    finger = [up[1] * approach[2] - up[2] * approach[1], up[2] * approach[0] - up[0] * approach[2],
              up[0] * approach[1] - up[1] * approach[0]]
    if solve_ik(hx, hy, high, approach, finger)["reachable"] and solve_ik(hx, hy, low, approach, finger)["reachable"]:
        # the hand body: 3-12 cm behind the grip point along the approach, +-10.5 cm along the finger line, from
        # 3.2 cm below the grip point up
        clear = room(hx - 0.075 * approach[0], hy - 0.075 * approach[1], finger, 0.105, 0.045, low - table - 0.032,
                     inside)
        if best is None or clear > best[0]:
            best = [clear, approach, finger]
if best is None:
    raise RuntimeError("no reachable upright pose")
chosen = best[1:]
move_gripper(pose["position"][0], pose["position"][1], high)
move_gripper(hx, hy, high)
move_gripper(hx, hy, high, chosen[0], chosen[1], path="joint")
move_by(dz=low - high)
open_gripper()
move_by(dz=0.15)
go_home()
result = check("{obj}", "{zone}")
'''
PROGRAM = {"paper": PAPER, "mug": SET_DOWN, "book": SET_DOWN, "pen": PEN}
ORDER = ("paper", "mug", "book", "pen")

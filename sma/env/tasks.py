"""Task families and randomized layouts for tabletop organization.

A task is an instruction plus a layout. The goal mapping (object -> zone) is used only by
the private evaluator; the agent must assign destinations from the instruction itself.
"""
import os
from dataclasses import dataclass

import numpy as np

from .scene import (KIND_RADIUS, OBJECT_RGBA as OBJECT_COLORS, PILE_HEIGHT, RECEPTACLE_HALF, Layout, ObjectSpec,
                    RECEPTACLES, ZoneSpec, receptacle_half)

# Tasks per set for the newer families (pen holder, desk tidy-up); 5 since 2026-11-04 (user request), 10 before.
SUITE_SIZE = int(os.environ.get("SMA_SUITE_SIZE", "5"))

# Reachable top-down workspace for the Panda on this table (world frame, metres).
OBJECT_X, OBJECT_Y = (-0.12, 0.20), (-0.26, 0.26)
SORT_RULE = {"block": "purple", "mug": "cyan", "book": "orange"}  # the simulated user's preference (colour of base)
KIND_WORDS = {"block": "toy block", "mug": "mug", "book": "book", "pen": "pen", "paper": "paper ball",
              "notebook": "notebook"}


def receptacle(color):
    return RECEPTACLES[color][1]


@dataclass
class Task:
    task_id: str
    family: str
    instruction: str
    layout: Layout
    goal: dict  # object name -> zone name (private)
    seed: int
    meta: dict = None  # e.g. realdesk: which objects start out of place (private, for the report)


def _radius(kind):
    return KIND_RADIUS[kind]


ROBOT_BASE_XY = np.array([-0.56, 0.0])
MAX_REACH = 0.70  # horizontal distance from the base (IK-checked reachable up to ~0.71 m)


def _sample_xy(rng, placed, radius, x_range=OBJECT_X, y_range=OBJECT_Y, tries=2000):
    for _ in range(tries):
        xy = np.array([rng.uniform(*x_range), rng.uniform(*y_range)])
        if np.linalg.norm(xy - ROBOT_BASE_XY) > MAX_REACH:
            continue
        if all(np.linalg.norm(xy - p) > radius + r + .02 for p, r in placed):
            return xy
    raise RuntimeError("Could not sample a collision-free layout")


TEACH_INSTRUCTION = ("From now on, please always put my toy blocks in the purple toy box, my mugs on the cyan tray "
                     "and my books on the orange tray. Tidy the desk that way now.")
VAGUE_INSTRUCTIONS = ["Tidy up my desk, please.", "Could you clean up the table?", "Put my things where they belong."]


def make_pen_task(seed):
    """'pen': put a pen into a narrow pen case. Case and pen have random orientations, so the pen must be turned
    to the case's long axis while it is carried (it fits only within about 13 degrees of alignment)."""
    rng = np.random.default_rng(seed)
    color = str(rng.choice(["red", "blue", "yellow"]))  # not green: the case is green
    hx, hy = receptacle_half("case")
    radius = np.hypot(hx + .008, hy + .008)
    for _ in range(200):  # the whole case, not only its centre, must be in reach
        xy = _sample_xy(rng, [], radius, x_range=(-0.10, 0.12), y_range=(-0.22, 0.22))
        yaw = float(rng.uniform(-np.pi / 2, np.pi / 2))
        ends = [xy + sgn * hx * np.array([np.cos(yaw), np.sin(yaw)]) for sgn in (-1, 1)]
        if all(np.linalg.norm(e - ROBOT_BASE_XY) < MAX_REACH - .03 for e in ends):
            break
    case = ZoneSpec("green", tuple(np.round(xy, 4)), round(yaw, 3))
    pen_xy = _sample_xy(rng, [(np.asarray(xy), radius)], _radius("pen"))
    pen = ObjectSpec(color, "pen", tuple(np.round(pen_xy, 4)), round(float(rng.uniform(-np.pi / 2, np.pi / 2)), 3))
    instruction = f"Put the {color} pen in the green pen case."
    return Task(f"pen_{seed}", "pen", instruction, Layout([pen], [case]), {pen.name: case.name}, seed)


def make_pen_holder_task(seed):
    """'pen_holder': stand a pen upright in a pen holder (a 6 cm square cup with 8 cm walls). The pen lies flat on
    the table, so it has to be grasped, turned from horizontal to vertical and lowered into the cup end first."""
    rng = np.random.default_rng(seed)
    color = str(rng.choice(["red", "blue", "yellow"]))
    radius = np.hypot(*receptacle_half("holder")) + .01
    xy = _sample_xy(rng, [], radius, x_range=(-0.08, 0.12), y_range=(-0.20, 0.20))
    holder = ZoneSpec("magenta", tuple(np.round(xy, 4)))
    pen_xy = _sample_xy(rng, [(np.asarray(xy), radius)], _radius("pen"))
    pen = ObjectSpec(color, "pen", tuple(np.round(pen_xy, 4)), round(float(rng.uniform(-np.pi / 2, np.pi / 2)), 3))
    instruction = f"Stand the {color} pen upright in the magenta pen holder."
    return Task(f"penholder_{seed}", "pen_holder", instruction, Layout([pen], [holder]), {pen.name: holder.name}, seed)


DESK_RULES = {"block": "toy blocks go in the purple toy box", "mug": "mugs go on the cyan tray",
              "book": "books go on the orange tray", "pen": "the pen stands upright in the magenta pen holder"}


def make_desk_task(seed):
    """'desk': a longer tidy-up. A pen (to stand in the holder) plus two of block / mug / book with their receptacles
    (and a second block when blocks are in play); one object starts in the middle of another object's receptacle,
    so it has to be cleared first. The instruction states every rule, so no user memory is needed.
    The table is crowded, so a layout that cannot be sampled is redrawn from the next sub-seed (still reproducible)."""
    for attempt in range(50):
        try:
            return _desk_layout(seed, np.random.default_rng([seed, attempt]))
        except RuntimeError:
            continue
    raise RuntimeError(f"no desk layout for seed {seed}")


def _desk_layout(seed, rng):
    others = [str(k) for k in rng.permutation(["block", "mug", "book"])[:2]]
    if SORT_RULE[others[1]] == "purple":  # the blocker sits on a tray, never inside the walled toy box
        others = others[::-1]
    kinds = others + ["pen"]
    zone_colors = [SORT_RULE[k] for k in others] + ["magenta"]
    zones, placed = [], []
    for color in zone_colors:
        footprint = (max(receptacle_half(RECEPTACLES[color][0])) + .008) * 1.15
        xy = _sample_xy(rng, placed, footprint, x_range=(-0.10, 0.16), y_range=(-0.25, 0.25))
        placed.append((xy, footprint))
        zones.append(ZoneSpec(color, tuple(np.round(xy, 4))))
    zone_xy = {z.name: np.array(z.xy) for z in zones}
    colors = [str(c) for c in rng.permutation(["red", "green", "blue", "yellow"])]
    objects, goal = [], {}
    items = kinds + (["block"] if "block" in kinds else [])
    for i, kind in enumerate(items):
        target = receptacle("magenta" if kind == "pen" else SORT_RULE[kind])
        if i == 0:  # the blocker: sits in the middle of the other object's receptacle
            xy = zone_xy[receptacle(SORT_RULE[others[1]])] + rng.uniform(-.01, .01, 2)
        else:
            xy = _sample_xy(rng, placed, _radius(kind))
        placed.append((np.asarray(xy), _radius(kind)))
        spec = ObjectSpec(colors[i], kind, tuple(np.round(xy, 4)), round(float(rng.uniform(-np.pi / 2, np.pi / 2)), 3))
        objects.append(spec)
        goal[spec.name] = target
    rules = [DESK_RULES[k] for k in others + ["pen"]]
    instruction = "Tidy up the desk: " + ", ".join(rules[:-1]) + " and " + rules[-1] + "."
    return Task(f"desk_{seed}", "desk", instruction, Layout(objects, zones), goal, seed)


# ---------------------------------------------------------------- realistic desk (realdesk)
# Where each kind belongs (receptacle base colour), and the receptacle's footprint for layout spacing.
REALDESK_ZONE = {"pen": "magenta", "mug": "blue", "book": "orange", "paper": "red", "notebook": "pile"}
# nbdesk: the realistic desk with a full-size A5 notebook (21 x 14.8 cm) instead of the pocket one; it belongs on the
# pile of textbooks (moved from the back of the desk, out of reach, to the receptacles' row). Out of place, the notebook
# lies at the right edge of the desk with its long edge 4-6 cm out over the side: the only place a gripper that opens
# 8 cm can take hold of it, pinching that edge from outside the desk (scripts/notebook_probe.py: lifted 7.1-7.5 cm with
# the hand coming in 40 degrees off square). The left edge is taken by the phone.
# The static desk clutter (assets.add_decor) as boxes: the laptop, the lamp and the phone (its rotated footprint's
# bounding box). The realdesk layouts never came near them, so they are added to the layout only for nbdesk.
DECOR_BOXES = [("box", .315, .03, .075, .11), ("box", .32, .31, .045, .045), ("box", -.31, .32, .055, .077)]


def _decor_boxes(phone_side=1):
    return DECOR_BOXES[:2] + [("box", -.31, .32 * phone_side, .055, .077)]
DESK_HALF = .40
NBDESK_INSTRUCTION = ("Tidy up my desk: pens stand upright in the magenta pen holder, the mug goes on the blue coaster, "
                      "the notebook goes on the pile of books, and crumpled paper balls go into the red wastebasket. "
                      "Some things are already where they belong; leave them there.")
# The vague request: where things go must be inferred from where things of the same kind already are.
VAGUE_DESK_INSTRUCTION = "Tidy up my desk, please. Some things are already where they belong; leave them there."
# cdesk: the same desk with every thing in its own colour and no colour or name in the request: which thing is the
# wastebasket, the pen holder, the notebook ... is for the robot to recognise from the camera.
CDESK_INSTRUCTION = ("Tidy up my desk: pens stand upright in the pen holder, the mug goes on the coaster, the notebook goes "
                     "on the pile of books, and crumpled paper balls go into the wastebasket. Some things are already "
                     "where they belong; leave them there.")
# What each thing is, in the words the robot is asked to use (private: used only to score its recognition).
CATEGORY = {"marker": "marker", "can": "drink can", "eraser": "eraser", "sticky": "sticky notes", "tape": "tape roll",
            "pen": "pen", "mug": "mug", "notebook": "notebook", "paper": "crumpled paper ball", "book": "book",
            "block": "toy block", "holder": "pen holder", "coaster": "coaster", "bin": "wastebasket",
            "pile": "pile of books", "tray": "tray", "box": "box", "case": "pen case"}
REALDESK_INSTRUCTION = ("Tidy up my desk: pens stand upright in the magenta pen holder, the mug goes on the blue "
                        "coaster, notebooks go on the orange tray, and crumpled paper balls go into the red "
                        "wastebasket. Some things are already where they belong; leave them there.")
# Clearance around a loose object for the open gripper (8 cm between the fingers): discs (offset along the
# object's long axis, radius). A pen gets three, so it can lie close to other things along its length.
CLEARANCE = {"pen": ((-.05, .022), (0.0, .042), (.05, .022)), "mug": ((0.0, .05),), "book": ((0.0, .052),),
             "paper": ((0.0, .045),), "block": ((0.0, .045),)}
WALL = .008  # receptacle wall thickness plus a little


def _gap(a, b):
    """Clearance between two shapes: ("disc", x, y, r) or ("box", x, y, hx, hy) (axis-aligned)."""
    if a[0] == "box" and b[0] == "box":
        return max(abs(a[1] - b[1]) - a[3] - b[3], abs(a[2] - b[2]) - a[4] - b[4])
    if a[0] == "box":
        a, b = b, a
    if b[0] == "disc":
        return float(np.hypot(a[1] - b[1], a[2] - b[2])) - a[3] - b[3]
    dx, dy = max(abs(a[1] - b[1]) - b[3], 0.0), max(abs(a[2] - b[2]) - b[4], 0.0)
    return float(np.hypot(dx, dy)) - a[3]


def _zone_box(color, xy, half=None):
    style = RECEPTACLES[color][0]
    hx, hy = half or receptacle_half(style)
    return ("box", float(xy[0]), float(xy[1]), hx + WALL, hy + WALL)


def _object_discs(kind, xy, yaw):
    u = np.array([np.cos(yaw), np.sin(yaw)])
    return [("disc", *(np.asarray(xy) + d * u), r) for d, r in CLEARANCE[kind]]


def _place(rng, shapes, make, x_range, y_range, gap, tries=400, reach=MAX_REACH - .02):
    """Sample a position (and yaw) whose shapes keep `gap` from every shape placed so far."""
    for _ in range(tries):
        xy = np.array([rng.uniform(*x_range), rng.uniform(*y_range)])
        if np.linalg.norm(xy - ROBOT_BASE_XY) > reach:
            continue
        yaw = float(rng.uniform(-np.pi / 2, np.pi / 2))
        new = make(xy, yaw)
        if all(_gap(a, b) >= gap for a in new for b in shapes):
            shapes.extend(new)
            return xy, yaw
    raise RuntimeError("no room")


def make_realdesk_task(seed, only=None, notebook=False, vague=False, anchors=False, varied=False, tray=False):
    """'realdesk': a realistic, crowded desk. Four receptacles stand at the back (pen holder, mug coaster, notebook
    tray, round wastebasket) and 3 or 4 things are out of place in front: a pen, a mug, a notebook, a paper ball.
    A kind that is not out of place starts tidy in its receptacle, and the wastebasket may already hold a paper ball. Every object, including the tidy ones, must end in its receptacle; the subtasks are
    the 3 or 4 objects out of place. Static clutter (a laptop, books, a lamp, a phone) sits out of reach.
    varied (nbdesk only): positions vary beyond the standard layout: the notebook lies at either side edge (drawn per
    seed; the phone moves to the other near corner) and sticks out further, 5.5-6.5 cm of its 14.8 cm; the receptacles
    may stand anywhere in reach, not only in a row at the back; and the pen, always out of place, lies on top of the
    pile of books, where the notebook has to go, so it has to be moved first."""
    # The number of subtasks is drawn once per seed, so redrawing a crowded layout does not favour 3 over 4.
    # only: a single kind out of place, the others tidy (validation of a skill learned on this desk).
    n_active = 1 if only else 4 if np.random.default_rng([seed, 99]).random() < .4 else 3
    active = None
    if notebook and not only:  # the notebook plus two others, drawn once per seed (a retry for a layout that fits must
        # not change which objects are out of place); the pile of books and the notebook leave no room for a fourth
        others = np.random.default_rng([seed, 98]).permutation(["pen", "mug", "paper"])[:2]
        if varied:  # the pen (on the pile) and one of the mug and the paper ball
            others = ["pen", str(np.random.default_rng([seed, 98]).choice(["mug", "paper"]))]
        active = ["notebook"] + [str(k) for k in others]
        if tray:  # user-preference desks: the notebook, a pen and a mug are always out of place
            active = ["notebook", "pen", "mug"]
    for attempt in range(1000):  # each attempt is drawn from (seed, attempt): a layout found earlier stays the same
        try:
            return _realdesk_layout(seed, np.random.default_rng([seed, attempt]), n_active, only, notebook, active, vague,
                                    anchors, varied, tray)
        except RuntimeError:
            continue
    raise RuntimeError(f"no realdesk layout for seed {seed}")


def _notebook_at_desk_edge(rng, shapes, color, side=-1, overhang=(.04, .06)):
    """The notebook along the right (side -1) or left (+1) desk edge, its long edge 4-6 cm out over the side. Kept
    clear: the part on the desk plus 4 cm (the wrist and forearm pass over it when the hand comes in from outside)."""
    overhang = float(rng.uniform(*overhang))
    ny = side * (DESK_HALF + overhang - .074)
    for _ in range(400):
        nx = rng.uniform(-0.25, -0.19)  # further out the hand must come in 60 degrees off square, and the fingers hold
        # only a corner of the edge (the notebook slipped out in the gate); nearer the robot (-0.264) the back-off
        # point before the pinch was out of reach
        on_desk = (DESK_HALF - abs(ny) + .074 - overhang) / 2 + .04
        new = [("box", nx, side * (DESK_HALF - on_desk + .04), .105 + .04, on_desk)]
        if all(_gap(a, b) >= .005 for a in new for b in shapes):
            shapes.extend(new)
            return ObjectSpec(color, "notebook", (round(nx, 4), round(ny, 4)), round(float(rng.uniform(-.03, .03)), 3))
    raise RuntimeError("no room")


def _pile_shapes(xy, side=-1):
    """The pile of books plus room for the hand on its right (side -1) or left (+1) side, the side of the desk edge the
    notebook lies at, where a notebook pinched by its edge can be laid on it with that edge out over the side (the
    hand hangs 10.4 cm below the grip point, beside the pile)."""
    hx, hy = RECEPTACLE_HALF["pile"]
    return [("box", float(xy[0]), float(xy[1]), hx + WALL, hy + WALL),
            ("box", float(xy[0]), float(xy[1]) + side * (hy + .07), hx * .8, .07)]


def _pen_on_pile(rng, color, pile):
    """A pen lying on top of the pile of books, wholly on its top surface (1.5 cm from the edges), at any heading."""
    hx, hy = RECEPTACLE_HALF["pile"]
    for _ in range(400):
        yaw = float(rng.uniform(-np.pi / 2, np.pi / 2))
        xy = np.asarray(pile.xy) + rng.uniform(-1, 1, 2) * [hx - .03, hy - .03]
        ends = [xy + s * .07 * np.array([np.cos(yaw), np.sin(yaw)]) for s in (-1, 1)]
        if all(abs(e[0] - pile.xy[0]) < hx - .015 and abs(e[1] - pile.xy[1]) < hy - .015 for e in ends):
            return ObjectSpec(color, "pen", tuple(np.round(xy, 4)), round(yaw, 3), lift=PILE_HEIGHT)
    raise RuntimeError("no room on the pile")


TRAY_HALF = (.085, .115)  # a letter tray, long side across the desk: an A5 notebook (21 x 14.8 cm) lies in it flat,
# so do a pen and a mug. It stands at the back of the desk, in front of the laptop, partly beyond the reach used for the
# other receptacles (the user-preference experiment scores where things are planned to go; the arm still reaches a pen
# or a mug in its front half)


def _realdesk_layout(seed, rng, n_active, only=None, notebook=False, active=None, vague=False, anchors=False,
                     varied=False, tray=False):
    kinds = ["pen", "mug", "notebook" if notebook else "book", "paper"]
    active = active or ([only] if only else [str(k) for k in rng.permutation(kinds)[:n_active]])
    # varied: the edge the notebook lies at, drawn once per seed (not per layout attempt), and the receptacles anywhere
    # in reach (the standard desk keeps them in a row at the back)
    side = (-1 if np.random.default_rng([seed, 97]).random() < .5 else 1) if varied or tray else -1
    back = (-0.12, 0.13) if varied or tray else (-0.03, 0.13)
    shapes, zones = (_decor_boxes(-side) if notebook else []), {}
    order = [str(k) for k in rng.permutation(kinds)]
    if notebook:  # the 25 cm pile of books first, or the smaller receptacles leave it no room
        order.remove("notebook")
        order.insert(0, "notebook")
    if tray:  # tray: a second plausible place (a letter tray), so a user's preference can differ from the desk's
        order.insert(1, "tray")
    for kind in order:  # receptacles at the back of the reachable area
        if kind == "tray":
            xy, _ = _place(rng, shapes, lambda p, _y: [_zone_box("cyan", p, TRAY_HALF)], (.08, .15), (-0.22, 0.22), .01,
                           reach=.80)
            zones["tray"] = ZoneSpec("cyan", tuple(np.round(xy, 4)), half=TRAY_HALF, wall_half=.015)
            continue
        color = REALDESK_ZONE[kind]
        # The pen holder within 0.64 m of the base: a pen held upright high above it is out of reach further out
        # (the act-level reference program failed at 0.665 m in the realdesk gate).
        if color == "pile":
            xy, _ = _place(rng, shapes, lambda p, _y: _pile_shapes(p, side), back,
                           (-0.15, 0.20) if side < 0 else (-0.20, 0.15), .01)
        else:
            xy, _ = _place(rng, shapes, lambda p, _y, c=color: [_zone_box(c, p)], back, (-0.25, 0.25), .01,
                           reach=.64 if kind == "pen" else MAX_REACH - .02)
        zones[kind] = ZoneSpec(color, tuple(np.round(xy, 4)))
    colors = [str(c) for c in rng.permutation(["yellow", "green", "cyan", "purple"])]
    objects, goal, out_of_place = [], {}, []
    for i, kind in enumerate(kinds):
        zone = zones[kind]
        if kind == "notebook" and kind in active:
            spec = _notebook_at_desk_edge(rng, shapes, colors[i], side, (.055, .065) if varied else (.04, .06))
            out_of_place.append(spec.name)
        elif kind == "pen" and varied and kind in active:
            spec = _pen_on_pile(rng, colors[i], zones["notebook"])
            out_of_place.append(spec.name)
        elif kind in active:
            xy, yaw = _place(rng, shapes, lambda p, y, k=kind: _object_discs(k, p, y), (-0.15, 0.14), (-0.26, 0.26), .005)
            spec = ObjectSpec(colors[i], kind, tuple(np.round(xy, 4)), round(yaw, 3))
            out_of_place.append(spec.name)
        else:
            spec = _tidy(rng, colors[i], kind, zone)
        objects.append(spec)
        goal[spec.name] = zone.name
    # An extra tidy paper ball in a wastebasket that still has to take another one (a colour not used above). No
    # extra pen in a holder that has to take another pen: the standing pen's top blocked the gripper in 2 of 2 tasks
    # of the realdesk gate (reference program), which made that subtask a different, much harder problem.
    if "paper" in active and (anchors or len(active) == 4 or rng.random() < .6):
        objects.append(_tidy(rng, "blue", "paper", zones["paper"]))
    # anchors: things of a loose object's kind already kept in their place, so where it goes can be inferred from the
    # desk (scene grouping): a red pen standing in the holder when a pen is out of place. (Destination inference only:
    # with a pen already standing in it, the holder blocked the gripper in the realdesk gate.)
    if anchors and "pen" in active:
        objects.append(_tidy(rng, "red", "pen", zones["pen"]))
    goal.update({o.name: zones[o.kind].name for o in objects})
    for spec in objects:
        if spec.name in out_of_place and spec.kind != "notebook" and not _hand_room(spec, objects, zones):
            raise RuntimeError("no room for the hand")
    tidy = [o.name for o in objects if o.name not in out_of_place]
    family = "nbdesk" if notebook else "realdesk"
    instruction = VAGUE_DESK_INSTRUCTION if vague else NBDESK_INSTRUCTION if notebook else REALDESK_INSTRUCTION
    task_id = "_".join([family] + ([only] if only else []) + (["vague"] if vague else []) +
                       (["anchors"] if anchors else []) + (["varied"] if varied else []) + (["tray"] if tray else [])
                       + [str(seed)])
    return Task(task_id, family, instruction, Layout(objects, list(zones.values()), decor=True, decor_books=not notebook,
                                                     phone_side=-side),
                goal, seed,
                meta={"out_of_place": out_of_place, "already_tidy": tidy, "subtasks": len(out_of_place)})


HAND_DISCS = (-.08, -.04, 0.0, .04, .08)  # the Panda hand (+-0.104 m along the fingers, +-0.032 m across) as discs
HAND_RADIUS = .034


def _hand_room(spec, objects, zones):
    """True if the hand can come down over the object with the fingers in at least one of the directions this kind is
    grasped in, without its body touching anything tall: the pen holder and wastebasket walls, the pile of books and
    the mugs. (The hand body starts 3 cm above the grip point, so trays, coasters, books, a notebook and lying pens are
    below it.) A pen can be taken anywhere along its length: the hand must fit at one of five points along it."""
    tall = [_zone_box(z.color, z.xy) for k, z in zones.items() if k in ("pen", "paper", "notebook")]
    tall += [("disc", *o.xy, .04) for o in objects if o.kind == "mug" and o is not spec]
    if spec.lift >= PILE_HEIGHT:  # on top of the pile of books: only the 20 cm wastebasket reaches above the hand's body
        tall = [_zone_box(z.color, z.xy) for k, z in zones.items() if k == "paper"]
    turns = {"pen": (np.pi / 2,), "book": (0.0, np.pi / 2)}.get(spec.kind, (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4))
    axis = np.array([np.cos(spec.yaw), np.sin(spec.yaw)])
    for turn in turns:
        yaw = (spec.yaw if spec.kind in ("pen", "book") else 0.0) + turn
        u = np.array([np.cos(yaw), np.sin(yaw)])
        for shift in ((-.04, -.02, 0.0, .02, .04) if spec.kind == "pen" else (0.0,)):
            grip = np.asarray(spec.xy) + shift * axis
            hand = [("disc", *(grip + d * u), HAND_RADIUS) for d in HAND_DISCS]
            if all(_gap(h, t) >= .005 for h in hand for t in tall):
                return True
    return False


def _tidy(rng, color, kind, zone):
    """An object that starts in its receptacle (on the 4 mm floor): a pen standing in a corner of the holder, a mug
    near the coaster's centre, a notebook on the tray, a paper ball off the wastebasket's centre."""
    c = np.asarray(zone.xy)
    if kind == "pen":
        xy = c + rng.choice([-1, 1], 2) * .014
        return ObjectSpec(color, kind, tuple(np.round(xy, 4)), 0.0, upright=True, lift=.004)
    if kind == "paper":
        a = rng.uniform(0, 2 * np.pi)
        xy = c + .025 * np.array([np.cos(a), np.sin(a)])
    elif kind == "notebook":  # lying on the pile of books
        xy = c + rng.uniform(-.015, .015, 2)
        return ObjectSpec(color, kind, tuple(np.round(xy, 4)), round(float(rng.uniform(-.05, .05)), 3), lift=PILE_HEIGHT)
    else:
        xy = c + rng.uniform(-.008, .008, 2)
    return ObjectSpec(color, kind, tuple(np.round(xy, 4)), round(float(rng.uniform(-np.pi / 2, np.pi / 2)), 3),
                      lift=.004)


def make_cdesk_task(seed, only=None, vague=False, anchors=False):
    """'cdesk': the nbdesk layout with every object and receptacle in its own palette colour (drawn per seed; the
    receptacles are coloured all over: base, walls, the books of the pile) and a request that names no colour. The
    camera reports things by colour only; recognising them is the robot's job. meta: "perceived_name" maps each
    private name to the colour the camera reports, "true_category" each colour to what it is (both private)."""
    from .palette import PALETTE
    task = make_realdesk_task(seed, only=only, notebook=True, vague=vague, anchors=anchors)
    # No second paper ball already in the wastebasket: nine colours, nine things at most (four receptacles, a pen, a mug,
    # a notebook, a paper ball, and with anchors a pen already in the holder).
    extra = [o for o in task.layout.objects if o.kind == "paper" and o.color == "blue"]
    for o in extra:
        task.layout.objects.remove(o)
        task.goal.pop(o.name, None)
        task.meta["already_tidy"] = [n for n in task.meta["already_tidy"] if n != o.name]
    names = [str(c) for c in np.random.default_rng([seed, 31]).permutation(list(PALETTE))]
    rename, category = {}, {}
    for spec in task.layout.objects:
        old_name, paint = spec.name, names.pop()
        spec.color, spec.rgba = paint, PALETTE[paint]
        spec.variant = "crumpled" if spec.kind == "paper" else None
        rename[old_name] = spec.name
        category[paint] = CATEGORY[spec.kind]
    for zone in task.layout.zones:
        old_name, paint = zone.name, names.pop()
        zone.paint = paint
        rename[old_name] = zone.name
        category[paint] = CATEGORY[RECEPTACLES[zone.color][0]]
    task.goal = {rename[o]: rename[z] for o, z in task.goal.items()}
    task.meta = {**task.meta, "out_of_place": [rename[o] for o in task.meta["out_of_place"]],
                 "already_tidy": [rename[o] for o in task.meta["already_tidy"]],
                 "perceived_name": {new: new.split("_")[0] for new in rename.values()}, "true_category": category}
    task.task_id = task.task_id.replace("nbdesk", "cdesk", 1)
    task.family = "cdesk"
    if not vague:
        task.instruction = CDESK_INSTRUCTION
    return task


def make_sdesk_task(seed, vague=True, anchors=True, look="real", varied=False, tray=False):
    """'sdesk': the nbdesk layout in ordinary desk colours that several things share (white paper balls, a white mug or
    wastebasket, pens and a holder of the same dark blue, ...), drawn per seed from five colours for nine or ten things,
    so the camera cannot tell things apart by colour: what is one thing comes from class-agnostic segmentation (SAM2)
    and depth (SMA_PERCEPTION=sam). With anchors (default) a pen already stands in the holder and a paper ball lies in
    the wastebasket, each the same colour as the loose one of its kind. The request names nothing. meta "category":
    each private name -> what it is (private: used only to score recognition)."""
    from .palette import NATURAL
    task = make_realdesk_task(seed, notebook=True, vague=vague, anchors=anchors, varied=varied, tray=tray)
    rng = np.random.default_rng([seed, 37])
    colours = list(NATURAL)
    by_kind = {"pen": str(rng.choice(["navy", "black", "red"])), "paper": "white"}
    rename, category, used = {}, {}, {}
    for spec in task.layout.objects:
        paint = by_kind.get(spec.kind) or str(rng.choice(colours))
        used[paint] = used.get(paint, 0) + 1
        old_name = spec.name
        spec.color, spec.rgba = f"{paint}{used[paint]}", NATURAL[paint]  # unique names; the colour is shared
        spec.variant = "crumpled" if spec.kind == "paper" else None
        rename[old_name] = spec.name
        category[spec.name] = CATEGORY[spec.kind]
    for zone in task.layout.zones:
        old_name = zone.name
        zone.paint = by_kind["pen"] if RECEPTACLES[zone.color][0] == "holder" and rng.random() < .5 else str(rng.choice(colours))
        rename[old_name] = zone.name
        category[zone.name] = CATEGORY[RECEPTACLES[zone.color][0]]
    task.goal = {rename[o]: rename[z] for o, z in task.goal.items()}
    task.meta = {**task.meta, "out_of_place": [rename[o] for o in task.meta["out_of_place"]],
                 "already_tidy": [rename[o] for o in task.meta["already_tidy"]], "category": category}
    task.task_id = task.task_id.replace("nbdesk", "sdesk" if look == "real" else "sdesk_plain", 1)
    task.family = "sdesk"
    task.layout.look = look
    if not vague:
        task.instruction = CDESK_INSTRUCTION
    return task


# ---------------------------------------------------------------- user-preference sessions (user memory experiment)
# One simulated user, eight sessions on new desks with a letter tray (sdesk, tray=True: a notebook, a pen and a mug out
# of place; a pen already in the holder, a paper ball in the wastebasket). Each session: what the user says, in two
# wordings, and where this user wants each loose thing now ("stay": leave it where it is), with the kind of test it is:
#   stated      the current message says it
#   rule        a lasting preference stated in an earlier session, against what the desk shows (books on the pile)
#   one_off     a request "just this once" in an earlier session, which must not carry over
#   changed     a lasting preference the user later changed
#   untaught    never mentioned: where the desk (things of its kind) or common sense puts it
# Private: the robot gets only the message and its camera.
UPREF_STORY = [
    (("Tidy up my desk, please. From now on, my notebook always goes in the tray, not on the pile of books.",
      "Please tidy my desk. Whenever you tidy up, put my notebook in the tray from now on; the pile is for textbooks."),
     {"notebook": ("tray", "stated"), "pen": ("holder", "untaught"), "mug": ("coaster", "untaught")}),
    (("Tidy up my desk, please.", "Could you tidy up the desk?"),
     {"notebook": ("tray", "rule"), "pen": ("holder", "untaught"), "mug": ("coaster", "untaught")}),
    (("Tidy up my desk, please. Just this once, leave the mug where it is; I'm still drinking from it.",
      "Please tidy up, but leave the mug where it is this time, I haven't finished my coffee."),
     {"notebook": ("tray", "rule"), "pen": ("holder", "untaught"), "mug": ("stay", "stated")}),
    (("Tidy up my desk, please.", "Tidy the desk, thanks."),
     {"notebook": ("tray", "rule"), "pen": ("holder", "untaught"), "mug": ("coaster", "one_off")}),
    (("Please tidy up. I've changed my mind: from now on, the notebook goes back on the pile of books.",
      "Tidy up please. Actually, from now on put the notebook on the pile of books again, not in the tray."),
     {"notebook": ("pile", "stated"), "pen": ("holder", "untaught"), "mug": ("coaster", "one_off")}),
    (("Tidy up my desk, please.", "Could you tidy up the desk?"),
     {"notebook": ("pile", "changed"), "pen": ("holder", "untaught"), "mug": ("coaster", "one_off")}),
    (("Tidy up, thanks.", "Please tidy my desk."),
     {"notebook": ("pile", "changed"), "pen": ("holder", "untaught"), "mug": ("coaster", "one_off")}),
    (("Tidy up my desk, please.", "Tidy the desk, thanks."),
     {"notebook": ("pile", "changed"), "pen": ("holder", "untaught"), "mug": ("coaster", "one_off")}),
]
# Seeds whose tray layout exists (make_sdesk_task(seed, tray=True)); a sequence takes eight in order.
UPREF_SEEDS = [18000, 18002, 18007, 18008, 18009, 18012, 18016, 18023, 18026, 18030, 18032, 18033, 18038, 18040, 18043,
               18046, 18047, 18050, 18051]


def upref_sessions(seq=0):
    """Sequence seq: the eight sessions of UPREF_STORY in wording seq % 2 on the desks UPREF_SEEDS[8 seq: 8 seq + 8].
    task.meta["user_truth"]: loose object -> the zone this user wants now (None: leave it); meta["cases"]: object -> test
    kind; task.goal follows the user (an object to be left is not scored)."""
    seeds = UPREF_SEEDS[8 * seq: 8 * seq + 8]
    tasks = []
    for i, ((words, plan), seed) in enumerate(zip(UPREF_STORY, seeds)):
        task = make_sdesk_task(seed, tray=True)
        task.instruction = words[seq % 2]
        zone_of = {z.name.split("_", 1)[1]: z.name for z in task.layout.zones}
        truth, cases = {}, {}
        for obj in task.meta["out_of_place"]:
            place, case = plan[obj.split("_", 1)[1]]
            truth[obj] = None if place == "stay" else zone_of[place]
            cases[obj] = case
            if truth[obj] is None:
                task.goal.pop(obj, None)
            else:
                task.goal[obj] = truth[obj]
        task.meta = {**task.meta, "user_truth": truth, "cases": cases, "session": i}
        task.task_id = f"upref{seq}_s{i}_{seed}"
        tasks.append(task)
    return tasks


# ---------------------------------------------------------------- xdesk: more kinds of things, varied desks
# Where each kind is kept (receptacle style) and what a desk may have out of place. Writing tools stand in the pen
# holder, trash goes into the wastebasket, the mug on its coaster, a book on the pile of books, small stationery into
# the desk organizer (a walled box).
XDESK_HOME = {"pen": "holder", "marker": "holder", "paper": "bin", "can": "bin", "mug": "coaster", "book": "pile",
              "eraser": "box", "sticky": "box", "tape": "box", "notebook": "pile"}
XDESK_STYLE_COLOR = {"holder": "magenta", "bin": "red", "coaster": "blue", "pile": "pile", "box": "purple"}
XDESK_COLOURS = {  # per kind: (word, rgba) the thing may come in
    "pen": [("navy", [.13, .18, .42, 1]), ("black", [.11, .11, .12, 1]), ("red", [.70, .13, .12, 1])],
    "marker": [("red", [.80, .12, .10, 1]), ("blue", [.10, .25, .75, 1]), ("green", [.10, .55, .25, 1])],
    "mug": [("white", [.86, .86, .83, 1]), ("navy", [.13, .18, .42, 1]), ("red", [.70, .13, .12, 1]),
            ("yellow", [.90, .75, .25, 1])],
    "paper": [("white", [.86, .86, .83, 1])],
    "can": [("red", [.75, .10, .10, 1]), ("blue", [.12, .30, .70, 1]), ("green", [.15, .55, .25, 1]),
            ("silver", [.75, .76, .78, 1])],
    "book": [("teal", [.10, .45, .50, 1]), ("maroon", [.50, .12, .18, 1]), ("ochre", [.75, .55, .15, 1])],
    "eraser": [("blue", [.15, .35, .75, 1]), ("black", [.15, .15, .17, 1]), ("red", [.75, .15, .15, 1])],
    "sticky": [("yellow", [.98, .90, .30, 1]), ("pink", [.98, .55, .70, 1]), ("lime", [.65, .90, .40, 1]),
               ("orange", [.98, .65, .25, 1])],
    "tape": [("beige", [.88, .80, .62, 1])],
    "notebook": [("black", [.11, .11, .12, 1]), ("red", [.70, .13, .12, 1]), ("navy", [.13, .18, .42, 1]),
                 ("grey", [.48, .49, .51, 1])],
}
XDESK_CLEARANCE = {"pen": CLEARANCE["pen"], "marker": CLEARANCE["pen"], "mug": ((0.0, .05),), "paper": ((0.0, .045),),
                   "can": ((0.0, .045),), "book": ((0.0, .052),), "eraser": ((0.0, .04),), "sticky": ((0.0, .048),),
                   "tape": ((0.0, .045),)}
XDESK_TALL = ("holder", "bin", "pile", "box")  # receptacles the hand's body must keep clear of (and mugs, cans)


def _xdesk_hand_room(spec, objects, zones):
    """As _hand_room, for the xdesk kinds: the hand comes down over the object in one of the directions it is taken in
    without its body touching a tall receptacle, a mug or a can."""
    tall = [_zone_box(z.color, z.xy, z.half) for style, z in zones.items() if style in XDESK_TALL]
    tall += [("disc", *o.xy, .04) for o in objects if o.kind in ("mug", "can") and o is not spec]
    elongated = spec.kind in ("pen", "marker", "book", "eraser", "sticky")
    turns = (np.pi / 2,) if spec.kind in ("pen", "marker") else (0.0, np.pi / 2) if elongated else \
        (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
    axis = np.array([np.cos(spec.yaw), np.sin(spec.yaw)])
    for turn in turns:
        yaw = (spec.yaw if elongated else 0.0) + turn
        u = np.array([np.cos(yaw), np.sin(yaw)])
        for shift in ((-.03, 0.0, .03) if spec.kind in ("pen", "marker") else (0.0,)):
            grip = np.asarray(spec.xy) + shift * axis
            hand = [("disc", *(grip + d * u), HAND_RADIUS) for d in HAND_DISCS]
            if all(_gap(h, t) >= .005 for h in hand for t in tall):
                return True
    return False


def _xdesk_decor(rng, shapes):
    """3-5 static things out of the arm's reach, drawn per desk: tall ones (lamp, plant, bottle) at the back on the left
    half, where they do not hide the desk from the demo camera; a phone may also lie at a near corner."""
    from .looks import DECOR_ITEMS
    kinds = [str(k) for k in rng.permutation(["laptop", "lamp", "books", "plant", "keyboard", "bottle", "phone"])]
    items = []
    for kind in kinds[:int(rng.integers(3, 6))]:
        hx, hy, _ = DECOR_ITEMS[kind]
        if kind == "phone" and rng.random() < .5:
            region = ((-.37, -.30), (.27, .35) if rng.random() < .5 else (-.35, -.27))
        else:
            region = ((.25 + hx - .05, .385 - hx), (-.04, .40 - hy) if kind in ("lamp", "plant", "bottle") else
                      (-.40 + hy, .40 - hy))
        for _ in range(200):
            xy = np.array([rng.uniform(*region[0]), rng.uniform(*region[1])])
            box = ("box", float(xy[0]), float(xy[1]), hx + .01, hy + .01)
            if all(_gap(box, b) >= 0 for b in shapes):
                shapes.append(box)
                yaw = float(rng.uniform(-.15, .15)) if kind in ("laptop", "keyboard", "books", "phone") else 0.0
                items.append((kind, round(float(xy[0]), 3), round(float(xy[1]), 3), round(yaw, 3)))
                break
    return items


def make_xdesk_task(seed, only=None, n_loose=None, notebook=False):
    """'xdesk': the realistic desk with more kinds of things and more of them, on a varied desk (SAM2 perception, the
    plain "tidy up" request). Five receptacles anywhere in reach (pen holder, wastebasket, coaster, pile of books, desk
    organizer); 5-6 things out of place, of different kinds drawn from XDESK_HOME (at most one mug: one coaster); some
    already kept in their place (a pen in the holder, a paper ball in the wastebasket, a stationery item in the
    organizer), so where things go can be read off the desk. The desk top (white, oak, walnut, grey) and the static
    clutter at the back (3-5 of a laptop, lamp, books, plant, keyboard, bottle, phone) are drawn per desk.
    only: a single kind out of place (validation of a stored skill).
    notebook ("xdesk_nb"): also the A5 notebook out of place, lying at a side edge of the desk (drawn per seed) with 4-6 cm
    of it out over the edge (a short grip, the only part a gripper can take); it goes on the pile of books, which then has room for
    the hand beside it on that side (as nbdesk); it counts among the 5-6 things out of place."""
    notebook = notebook or only == "notebook"  # a notebook out of place always lies at a desk edge
    for attempt in range(1200 if notebook else 400):  # with the notebook and the hand's room by the pile it is crowded
        try:
            return _xdesk_layout(seed, np.random.default_rng([seed, 500 + attempt]), only, n_loose, notebook)
        except RuntimeError:
            continue
    raise RuntimeError(f"no xdesk layout for seed {seed}")


def _xdesk_layout(seed, rng, only, n_loose, notebook=False):
    from .looks import DESK_SURFACES
    from .palette import NATURAL
    first = np.random.default_rng([seed, 501])  # drawn once per seed: the desk top, how many things, which kinds
    desk = str(first.choice(DESK_SURFACES))
    n = 1 if only else (n_loose or int(first.integers(5, 7)))
    pool = [k for k in XDESK_HOME if k not in ("mug", "notebook")]
    kinds = ([] if only == "notebook" else [only]) if only else \
        [str(k) for k in first.permutation(pool + ["mug"])[:n - (1 if notebook else 0)]]
    shapes = []
    side, nb_spec = -1, None
    if notebook:  # first, at a side edge; the static clutter and everything else keep clear of it
        side = -1 if np.random.default_rng([seed, 503]).random() < .5 else 1
        word, rgba = XDESK_COLOURS["notebook"][int(rng.integers(len(XDESK_COLOURS["notebook"])))]
        nb_spec = _notebook_at_desk_edge(rng, shapes, word, side, (.04, .06))  # a short grip out over the edge
        nb_spec.rgba = rgba
    decor = _xdesk_decor(np.random.default_rng([seed, 502]), shapes)
    zones = {}
    for style in [str(st) for st in rng.permutation(["pile", "bin", "box", "coaster", "holder"])]:
        color = XDESK_STYLE_COLOR[style]
        if style == "pile" and notebook:  # with room for the hand beside it on the notebook's side
            xy, _ = _place(rng, shapes, lambda p, _y: _pile_shapes(p, side), (-0.12, 0.13),
                           (-0.15, 0.20) if side < 0 else (-0.20, 0.15), .01)
        elif style == "pile":
            xy, _ = _place(rng, shapes, lambda p, _y: [_zone_box("pile", p)], (-0.12, 0.13), (-0.25, 0.25), .01)
        else:
            xy, _ = _place(rng, shapes, lambda p, _y, c=color: [_zone_box(c, p)], (-0.12, 0.13), (-0.25, 0.25), .01,
                           reach=.64 if style == "holder" else MAX_REACH - .02)
        zones[style] = ZoneSpec(color, tuple(np.round(xy, 4)))
    used, objects, out_of_place = {}, [], []
    if nb_spec is not None:
        used[nb_spec.color] = 1
        objects.append(nb_spec)
        out_of_place.append(nb_spec.name)

    def spec_of(kind, xy, yaw, **kw):
        word, rgba = XDESK_COLOURS[kind][int(rng.integers(len(XDESK_COLOURS[kind])))]
        used[word] = used.get(word, 0) + 1
        spec = ObjectSpec(f"{word}{used[word]}", kind, tuple(np.round(xy, 4)), round(float(yaw), 3), **kw)
        spec.rgba = rgba
        return spec
    for kind in kinds:
        xy, yaw = _place(rng, shapes, lambda p, y, k=kind: [("disc", *(np.asarray(p) + d * np.array([np.cos(y), np.sin(y)])), r)
                                                          for d, r in XDESK_CLEARANCE[k]], (-0.22, 0.14), (-0.26, 0.26), .005)
        objects.append(spec_of(kind, xy, yaw))
        out_of_place.append(objects[-1].name)
    # things already kept in their place: the evidence for where the loose ones go
    homes = {XDESK_HOME[k] for k in kinds}  # (the pile of books needs no example: it is books)
    if not only:
        if "holder" in homes:
            c = np.asarray(zones["holder"].xy) + rng.choice([-1, 1], 2) * .014
            objects.append(spec_of("pen", c, 0.0, upright=True, lift=.004))
        if "bin" in homes:
            a = rng.uniform(0, 2 * np.pi)
            objects.append(spec_of("paper", np.asarray(zones["bin"].xy) + .025 * np.array([np.cos(a), np.sin(a)]),
                                   rng.uniform(-np.pi, np.pi), lift=.004))
        if "box" in homes:
            k = str(rng.choice(["eraser", "sticky", "tape"]))
            objects.append(spec_of(k, np.asarray(zones["box"].xy) + rng.uniform(-.015, .015, 2), rng.uniform(-.3, .3),
                                   lift=.004))
    for spec in objects:
        if spec.name in out_of_place and spec.kind != "notebook" and not _xdesk_hand_room(spec, objects, zones):
            raise RuntimeError("no room for the hand")
    goal = {o.name: zones[XDESK_HOME[o.kind]].name for o in objects}
    # the receptacles in ordinary colours (as sdesk), named by colour and what they are
    rename, category = {}, {}
    for style, zone in zones.items():
        old = zone.name
        zone.paint = str(rng.choice(list(NATURAL)))
        rename[old] = zone.name
        category[zone.name] = "desk organizer" if style == "box" else CATEGORY[style]
    goal = {o: rename[z] for o, z in goal.items()}
    for spec in objects:
        category[spec.name] = CATEGORY[spec.kind]
    tidy = [o.name for o in objects if o.name not in out_of_place]
    layout = Layout(objects, list(zones.values()), decor=False, look="real", decor_items=decor, desk=desk)
    task_id = "_".join(["xdesk"] + (["nb"] if notebook else []) + ([only] if only else []) + [str(seed)])
    return Task(task_id, "xdesk", VAGUE_DESK_INSTRUCTION, layout, goal, seed,
                meta={"out_of_place": out_of_place, "already_tidy": tidy, "subtasks": len(out_of_place),
                      "category": category, "desk": desk, "decor": [d[0] for d in decor]})


GROUP_KINDS = ("block", "mug", "book", "pen")
GROUP_CONTAINERS = ("orange", "cyan", "purple")  # two trays and a toy box: plain names, no hint of what goes where


def make_groupdesk_task(seed, kinds=None, anchors=True, task_id=None):
    """'groupdesk': three plain containers (an orange tray, a cyan tray, a purple box), each already holding one thing
    of a kind (drawn per seed from toy block, mug, pocket book, pen), and one more thing of each of those kinds loose on
    the desk. Nothing in the names says which container is for what, so under the plain "tidy up" request a loose thing
    belongs with the one of its kind (destination inference from the desk itself, cf. TidyBot). Six objects, one per
    object colour (perception tells objects apart by colour)."""
    fixed = kinds
    for attempt in range(400):
        rng = np.random.default_rng([seed, 23, attempt])
        try:
            drawn = [str(k) for k in rng.permutation(GROUP_KINDS)[:3]]
            kinds = fixed or drawn
            colors = [str(c) for c in rng.permutation(list(OBJECT_COLORS))]
            shapes, zones, objects, goal, loose = [], [], [], {}, []
            for zc, kind in zip(GROUP_CONTAINERS, kinds):
                xy, _ = _place(rng, shapes, lambda p, _y, c=zc: [_zone_box(c, p)], (-0.03, 0.13), (-0.25, 0.25), .01)
                zone = ZoneSpec(zc, tuple(np.round(xy, 4)))
                zones.append(zone)
                if not anchors:  # an empty container: nothing on this desk shows what it is for
                    continue
                # a 14 cm pen lies diagonally (the containers are 11.6-13.6 cm inside)
                yaw = np.pi / 4 * float(rng.choice([-1, 1])) if kind == "pen" else float(rng.uniform(-np.pi / 2, np.pi / 2))
                anchor = ObjectSpec(colors.pop(), kind, tuple(np.round(xy + rng.uniform(-.008, .008, 2), 4)),
                                    round(yaw, 3), lift=.004)
                objects.append(anchor)
                goal[anchor.name] = zone.name
            for zone, kind in zip(zones, kinds):
                xy, yaw = _place(rng, shapes, lambda p, y, k=kind: _object_discs(k, p, y), (-0.15, 0.14), (-0.26, 0.26), .005)
                spec = ObjectSpec(colors.pop(), kind, tuple(np.round(xy, 4)), round(yaw, 3))
                objects.append(spec)
                goal[spec.name] = zone.name
                loose.append(spec.name)
        except RuntimeError:
            continue
        # No pile of books in the decor: it is a bigger group of books than a container's one book (the first check sent
        # every loose book there), and it is out of reach.
        return Task(task_id or f"groupdesk_{seed}", "groupdesk", VAGUE_DESK_INSTRUCTION,
                    Layout(objects, zones, decor=True, decor_books=False), goal, seed,
                    meta={"out_of_place": loose, "already_tidy": [o.name for o in objects if o.name not in loose],
                          "subtasks": len(loose), "container_kinds": dict(zip(GROUP_CONTAINERS, kinds))})
    raise RuntimeError(f"no groupdesk layout for seed {seed}")


def make_single_task(kind, style, seed):
    """One object and one receptacle of the given style (validation of a stored skill on a new layout)."""
    rng = np.random.default_rng([seed, 17])
    color = next(c for c, (st, _) in RECEPTACLES.items() if st == style and (st != "tray" or c == "orange"))
    for _ in range(200):
        try:
            shapes = []
            xy, _ = _place(rng, shapes, lambda p, _y: [_zone_box(color, p)], (-0.03, 0.12), (-0.22, 0.22), .01)
            oxy, yaw = _place(rng, shapes, lambda p, y: _object_discs(kind, p, y), (-0.12, 0.14), (-0.24, 0.24), .005)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError(f"no single-object layout for {kind} / {style} / {seed}")
    zone = ZoneSpec(color, tuple(np.round(xy, 4)))
    obj = ObjectSpec(str(rng.choice(["yellow", "green", "cyan", "purple"])), kind, tuple(np.round(oxy, 4)),
                     round(yaw, 3))
    word = "into" if style in ("bin", "holder", "box", "case") else "on"
    return Task(f"single_{kind}_{style}_{seed}", "single", f"Put the {obj.color} {KIND_WORDS[kind]} {word} the "
                f"{zone.name.replace('_', ' ')}.", Layout([obj], [zone], decor=style in ("coaster", "bin")),
                {obj.name: zone.name}, seed)


PEN_TEACH_INSTRUCTION = ("From now on, whenever you put a pen in my pen case, turn the pen so that it lines up with the "
                         "long side of the case before you lower it. Please put the {color} pen in the green pen case now.")


def make_task(family, seed, *, kinds=None):
    if family == "pen":
        return make_pen_task(seed)
    if family == "pen_holder":
        return make_pen_holder_task(seed)
    if family == "desk":
        return make_desk_task(seed)
    if family == "realdesk":
        return make_realdesk_task(seed)
    if family == "pen_teach":  # same layout family; the user states how pens go into the case, once
        task = make_pen_task(seed)
        task.task_id, task.family = f"pen_teach_{seed}", "pen_teach"
        task.instruction = PEN_TEACH_INSTRUCTION.format(color=task.layout.objects[0].color)
        return task
    if family in ("teach", "tidy"):
        # Same layouts as sort/clear; the instruction no longer states where things go (teach states it once).
        base = make_task("clear" if seed % 2 else "sort", seed, kinds=kinds)
        base.family = family
        base.task_id = f"{family}_{seed}"
        base.instruction = TEACH_INSTRUCTION if family == "teach" else VAGUE_INSTRUCTIONS[seed % len(VAGUE_INSTRUCTIONS)]
        return base
    """family: 'place' (one object, one zone + distractor zone), 'sort' (2-3 objects by kind),
    'clear' (a target zone is initially covered by an object that belongs elsewhere)."""
    rng = np.random.default_rng(seed)
    colors = list(rng.permutation(["red", "green", "blue", "yellow"]))
    if family == "place":
        kinds = kinds or [str(rng.choice(["block", "mug", "book"]))]
        zone_colors = list(rng.permutation(list(SORT_RULE.values())))[:2]
    elif family in ("sort", "clear"):
        kinds = kinds or list(rng.permutation(["block", "mug", "book"]))[:2]
        zone_colors = sorted({SORT_RULE[k] for k in kinds})
        spare = [c for c in SORT_RULE.values() if c not in zone_colors]
        zone_colors += spare[:1]
    else:
        raise ValueError(family)

    zones, placed = [], []
    for color in zone_colors:
        footprint = (max(receptacle_half(RECEPTACLES[color][0])) + .008) * 1.15  # base + walls, as a radius
        xy = _sample_xy(rng, placed, footprint, x_range=(-0.12, 0.16), y_range=(-0.26, 0.26))
        placed.append((xy, footprint))
        zones.append(ZoneSpec(color, tuple(np.round(xy, 4))))
    zone_xy = {z.name: np.array(z.xy) for z in zones}

    objects = []
    goal = {}
    for i, kind in enumerate(kinds):
        spec_color = colors[i]
        target_zone = receptacle(SORT_RULE[kind]) if family != "place" else zones[0].name
        if family == "clear" and i == 0:
            # Object 0 sits on the zone that object 1 needs; it belongs to its own zone.
            blocked = receptacle(SORT_RULE[kinds[1]])
            xy = zone_xy[blocked] + rng.uniform(-.015, .015, 2)
        else:
            xy = _sample_xy(rng, placed, _radius(kind))
        placed.append((np.asarray(xy), _radius(kind)))
        yaw = float(rng.uniform(-np.pi / 2, np.pi / 2))
        spec = ObjectSpec(spec_color, kind, tuple(np.round(xy, 4)), round(yaw, 3))
        objects.append(spec)
        goal[spec.name] = target_zone

    if family == "place":
        target = zones[0].name.replace("_", " ")
        word = "in" if zones[0].name.endswith("box") else "on"
        instruction = f"Put the {objects[0].color} {KIND_WORDS[objects[0].kind]} {word} the {target}."
    else:
        instruction = ("Organize this desk: toy blocks go in the purple toy box, mugs on the cyan tray and "
                       "books on the orange tray.")
    return Task(f"{family}_{seed}", family, instruction, Layout(objects, zones), goal, seed)


def suite(name):
    """Fixed, reproducible task sequences. Seeds >= 1000 are held-out layouts."""
    if name == "train":
        return [make_task("place", 1, kinds=["block"]), make_task("place", 2, kinds=["book"]),
                make_task("place", 3, kinds=["mug"]), make_task("sort", 4)]
    if name == "heldout":
        return [make_task("sort", 1000 + i) for i in range(3)] + [make_task("clear", 1100 + i) for i in range(3)]
    if name == "session":
        # One simulated user: teaches preferences once, then gives vague requests on new layouts.
        return [make_task("teach", 4), make_task("place", 1, kinds=["block"]), make_task("place", 2, kinds=["book"]),
                make_task("place", 3, kinds=["mug"])] + [make_task("tidy", 1000 + i) for i in range(6)]
    if name == "pen":
        # Ten pen-case placements on new layouts (explicit instructions, so user memory plays no role).
        return [make_task("pen", 2000 + i) for i in range(10)]
    if name == "pen_teach":
        # The user teaches how pens go into the case once, then gives the ten plain pen requests.
        return [make_task("pen_teach", 1999)] + [make_task("pen", 2000 + i) for i in range(10)]
    if name.startswith(("pen@", "pen_teach@")):
        # Replication on other layout sets: "pen@3000" = pen_3000..pen_3009; "pen_teach@3000" adds the
        # teaching task on layout 2999 first.
        family, base = name.split("@")
        tasks = [make_task("pen", int(base) + i) for i in range(10)]
        return ([make_task("pen_teach", int(base) - 1)] if family == "pen_teach" else []) + tasks
    if name == "pen_holder":
        # Pen-holder tasks: stand a pen upright in a cup (needs the gripper tilted, not only turned).
        return [make_task("pen_holder", 6000 + i) for i in range(SUITE_SIZE)]
    if name == "desk" or name.startswith("desk@"):  # long-horizon tidy-up: "desk" = desk_9000, 9001, ...
        base = 9000 if name == "desk" else int(name.split("@")[1])
        return [make_task("desk", base + i) for i in range(SUITE_SIZE)]
    if name == "realdesk" or name.startswith("realdesk@"):  # realistic desk: "realdesk" = realdesk_11000, ...
        base = 11000 if name == "realdesk" else int(name.split("@")[1])
        return [make_task("realdesk", base + i) for i in range(SUITE_SIZE)]
    if name.startswith("cdesk"):  # the colour-coded desk: "cdesk@16000", "cdesk_notebook@16000", "cdesk_vague@16000"
        head, _, base = name.partition("@")
        words = head.split("_")[1:]
        only = next((w for w in words if w not in ("vague", "anchors")), None)
        return [make_cdesk_task(int(base or 16000) + i, only=only, vague="vague" in words, anchors="anchors" in words)
                for i in range(SUITE_SIZE)]
    if name.startswith("xdesk"):  # "xdesk@19000": more kinds and more things on varied desks (make_xdesk_task);
        # "xdesk_nb@19100": also the A5 notebook out over a side edge
        base, tasks = int(name.partition("@")[2] or 19000), []
        for seed in range(base, base + 10 * SUITE_SIZE):  # the seeds that have a layout, in order
            try:
                tasks.append(make_xdesk_task(seed, notebook=name.startswith("xdesk_nb")))
            except RuntimeError:
                continue
            if len(tasks) == SUITE_SIZE:
                break
        return tasks
    if name.startswith("upref"):  # "upref@0": the user-preference sessions, sequence 0 (upref_sessions)
        return upref_sessions(int(name.partition("@")[2] or 0))
    if name.startswith("sdesk"):  # ordinary colours shared by several things: "sdesk@17000", "sdesk_named@17000" (the
        # request lists where things go, still naming no colour), "sdesk_noanchors@17000"; things look real (textures,
        # caps, handles, page edges; looks.py) unless "plain" ("sdesk_plain@17000": the flat-coloured shapes);
        # "sdesk_varied@17300": positions vary beyond the standard layout (make_realdesk_task)
        head, _, base = name.partition("@")
        words = head.split("_")[1:]
        return [make_sdesk_task(int(base or 17000) + i, vague="named" not in words, anchors="noanchors" not in words,
                                look="plain" if "plain" in words else "real", varied="varied" in words)
                for i in range(SUITE_SIZE)]
    if name.startswith("groupseq"):  # one user's desk seen again and again: "groupseq@15100" = plain containers kept for
        # the same kinds on every desk (drawn once for the sequence); the first three desks show one thing of each kind
        # already in its container, the rest show only empty containers, under the plain "tidy up" request
        # "groupseq_warm@..." = only the three desks with things in place, "groupseq_empty@..." = only the rest
        head, _, base = name.partition("@")
        base = int(base or 15100)
        kinds = [str(k) for k in np.random.default_rng([base, 41]).permutation(GROUP_KINDS)[:3]]
        part = {"groupseq_warm": range(3), "groupseq_empty": range(3, max(SUITE_SIZE, 4))}.get(head, range(SUITE_SIZE))
        return [make_groupdesk_task(base + i, kinds=kinds, anchors=i < 3, task_id=f"groupseq_{base + i}") for i in part]
    if name.startswith("groupdesk"):  # "groupdesk@15000": plain containers, each holding one thing of a kind
        base = int(name.partition("@")[2] or 15000)
        return [make_groupdesk_task(base + i) for i in range(SUITE_SIZE)]
    if name.startswith("nbdesk"):  # the desk with an A5 notebook at the right desk edge, which goes on the pile of books:
        # "nbdesk@14000" = the notebook and two others out of place, "nbdesk_notebook@14000" = only the notebook;
        # "_vague" = the plain "tidy up" request (where things go must be inferred from the desk); "_anchors" = a pen
        # already in the holder and a paper ball already in the wastebasket whenever those kinds are out of place
        head, _, base = name.partition("@")
        words = head.split("_")[1:]
        only = next((w for w in words if w not in ("vague", "anchors")), None)
        return [make_realdesk_task(int(base or 14000) + i, only=only, notebook=True, vague="vague" in words,
                                   anchors="anchors" in words) for i in range(SUITE_SIZE)]
    if name.startswith("pen_holder@"):  # other layout sets: "pen_holder@7000" = penholder_7000, 7001, ...
        return [make_task("pen_holder", int(name.split("@")[1]) + i) for i in range(SUITE_SIZE)]
    if name == "smoke":
        return [make_task("place", 7, kinds=["block"])]
    raise ValueError(name)

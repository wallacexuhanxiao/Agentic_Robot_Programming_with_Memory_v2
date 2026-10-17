"""Task families and randomized layouts for tabletop organization.

A task is an instruction plus a layout. The goal mapping (object -> zone) is used only by
the private evaluator; the agent must assign destinations from the instruction itself.
"""
from dataclasses import dataclass

import numpy as np

from .scene import KIND_RADIUS, Layout, ObjectSpec, RECEPTACLE_HALF, RECEPTACLES, ZoneSpec

# Reachable top-down workspace for the Panda on this table (world frame, metres).
OBJECT_X, OBJECT_Y = (-0.12, 0.20), (-0.26, 0.26)
SORT_RULE = {"block": "purple", "mug": "cyan", "book": "orange"}  # the simulated user's preference (colour of base)
KIND_WORDS = {"block": "toy block", "mug": "mug", "book": "book"}


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


def make_task(family, seed, *, kinds=None):
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
        footprint = (RECEPTACLE_HALF[RECEPTACLES[color][0]] + .008) * 1.15  # base + walls, as a radius
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
    if name == "smoke":
        return [make_task("place", 7, kinds=["block"])]
    raise ValueError(name)

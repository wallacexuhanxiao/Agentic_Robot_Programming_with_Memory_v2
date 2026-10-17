"""Task families and randomized layouts for tabletop organization.

A task is an instruction plus a layout. The goal mapping (object -> zone) is used only by
the private evaluator; the agent must assign destinations from the instruction itself.
"""
from dataclasses import dataclass

import numpy as np

from .scene import KINDS, Layout, ObjectSpec, ZONE_HALF, ZoneSpec

# Reachable top-down workspace for the Panda on this table (world frame, metres).
OBJECT_X, OBJECT_Y = (-0.12, 0.20), (-0.26, 0.26)
SORT_RULE = {"cube": "purple", "cup": "cyan", "book": "orange"}


@dataclass
class Task:
    task_id: str
    family: str
    instruction: str
    layout: Layout
    goal: dict  # object name -> zone name (private)
    seed: int


def _radius(kind):
    shape, size = KINDS[kind]
    return float(np.hypot(size[0], size[1])) if shape == "box" else size[0]


ROBOT_BASE_XY = np.array([-0.56, 0.0])
MAX_REACH = 0.68  # horizontal distance from the base that leaves margin for approach heights and wrist yaw


def _sample_xy(rng, placed, radius, x_range=OBJECT_X, y_range=OBJECT_Y, tries=2000):
    for _ in range(tries):
        xy = np.array([rng.uniform(*x_range), rng.uniform(*y_range)])
        if np.linalg.norm(xy - ROBOT_BASE_XY) > MAX_REACH:
            continue
        if all(np.linalg.norm(xy - p) > radius + r + .03 for p, r in placed):
            return xy
    raise RuntimeError("Could not sample a collision-free layout")


TEACH_INSTRUCTION = ("From now on, I always want cubes on the purple zone, cups on the cyan zone and books on the "
                     "orange zone. Please tidy the table that way.")
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
        kinds = kinds or [str(rng.choice(["cube", "cup", "book"]))]
        zone_colors = list(rng.permutation(list(SORT_RULE.values())))[:2]
    elif family in ("sort", "clear"):
        kinds = kinds or list(rng.permutation(["cube", "cup", "book"]))[:2 if family == "sort" else 2]
        zone_colors = sorted({SORT_RULE[k] for k in kinds})
        spare = [c for c in SORT_RULE.values() if c not in zone_colors]
        zone_colors += spare[:1]
    else:
        raise ValueError(family)

    zones, placed = [], []
    for color in zone_colors:
        xy = _sample_xy(rng, placed, ZONE_HALF * 1.2, x_range=(-0.12, 0.18), y_range=(-0.24, 0.24))
        placed.append((xy, ZONE_HALF * 1.2))
        zones.append(ZoneSpec(color, tuple(np.round(xy, 4))))
    zone_xy = {z.name: np.array(z.xy) for z in zones}

    objects = []
    goal = {}
    for i, kind in enumerate(kinds):
        spec_color = colors[i]
        target_zone = f"{SORT_RULE[kind]}_zone" if family != "place" else zones[0].name
        if family == "clear" and i == 0:
            # Object 0 sits on the zone that object 1 needs; it belongs to its own zone.
            blocked = f"{SORT_RULE[kinds[1]]}_zone"
            xy = zone_xy[blocked] + rng.uniform(-.015, .015, 2)
        else:
            xy = _sample_xy(rng, placed, _radius(kind))
        placed.append((np.asarray(xy), _radius(kind)))
        yaw = float(rng.uniform(-np.pi / 2, np.pi / 2))
        spec = ObjectSpec(spec_color, kind, tuple(np.round(xy, 4)), round(yaw, 3))
        objects.append(spec)
        goal[spec.name] = target_zone

    names = [o.name.replace("_", " ") for o in objects]
    if family == "place":
        instruction = f"Put the {names[0]} on the {zones[0].name.replace('_', ' ')}."
    else:
        instruction = ("Organize this table: every cube belongs on the purple zone, every cup on the "
                       "cyan zone and every book on the orange zone. Leave each object resting on its zone.")
    return Task(f"{family}_{seed}", family, instruction, Layout(objects, zones), goal, seed)


def suite(name):
    """Fixed, reproducible task sequences. Seeds >= 1000 are held-out layouts."""
    if name == "train":
        return [make_task("place", 1, kinds=["cube"]), make_task("place", 2, kinds=["book"]),
                make_task("place", 3, kinds=["cup"]), make_task("sort", 4)]
    if name == "heldout":
        return [make_task("sort", 1000 + i) for i in range(3)] + [make_task("clear", 1100 + i) for i in range(3)]
    if name == "session":
        # One simulated user: teaches preferences once, then gives vague requests on new layouts.
        return [make_task("teach", 4), make_task("place", 1, kinds=["cube"]), make_task("place", 2, kinds=["book"]),
                make_task("place", 3, kinds=["cup"])] + [make_task("tidy", 1000 + i) for i in range(6)]
    if name == "smoke":
        return [make_task("place", 7, kinds=["cube"])]
    raise ValueError(name)

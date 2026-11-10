"""Robosuite tabletop with varied layouts: toy blocks, mugs, books and pens, plus trays, a toy box and a pen case.

The layout (which objects, where, which zones) is supplied by a Layout; the scene itself
only builds MJCF. Object state is read back only by the private evaluator.
"""
from dataclasses import dataclass, field

import numpy as np
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import new_geom
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

TABLE_Z = 0.8
# Object colors are segmented above the table and zone colors at table height, so the two sets may share a hue.
OBJECT_RGBA = {"red": [.85, .08, .06, 1], "green": [.08, .62, .12, 1], "blue": [.06, .18, .85, 1],
               "yellow": [.92, .78, .05, 1],
               "cyan": [.05, .72, .78, 1], "purple": [.55, .12, .70, 1]}  # same RGBA as the cyan / purple zone bases
                                                                        # (calibrated hues 90 / 144); realdesk only
# Pen case: same green as the objects (rendered hue 62); pink (150) and lime (32) rendered too close to the purple
# box / orange tray. Coaster and wastebasket (realdesk): the object blue and red.
ZONE_RGBA = {"purple": [.55, .12, .70, 1], "cyan": [.05, .72, .78, 1], "orange": [.95, .45, .05, 1],
             "green": [.08, .62, .12, 1], "magenta": [.85, .08, .45, 1],
             "blue": [.06, .18, .85, 1], "red": [.85, .08, .06, 1]}
# Receptacle catalogue: base colour -> (style, name). Trays are shallow; the toy box has walls.
RECEPTACLES = {"orange": ("tray", "orange_tray"), "cyan": ("tray", "cyan_tray"), "purple": ("box", "purple_box"),
               "green": ("case", "green_case"),  # the pen case is long and narrow: a pen fits only when aligned
               "magenta": ("holder", "magenta_holder"),  # a pen holder (cup): a pen goes in standing on its end
               "blue": ("coaster", "blue_coaster"),  # realdesk: a small mug coaster with a low rim
               "red": ("bin", "red_bin"),  # realdesk: a round wastebasket (floor disc, 20 cm walls)
               "pile": ("pile", "book_pile")}  # nbdesk: a pile of four textbooks the notebook goes on (no colour:
                                               # perception finds it from depth, a flat grey top 11 cm up)
# Object half heights (m), used by the evaluator.
KIND_HALF_HEIGHT = {"block": .02, "book": .0125, "mug": .035, "pen": .006, "paper": .02, "notebook": .005}
KIND_RADIUS = {"block": .028, "book": .047, "mug": .04, "pen": .072, "paper": .025, "notebook": .128}
ZONE_HALF = .055                               # default inner half width (toy box)
# Trays are larger so a rotated book lies flat; the bin's value is its radius at the mouth (it is round, 14 cm
# across at the floor: assets.BIN_BOTTOM_RADIUS).
RECEPTACLE_HALF = {"tray": .068, "box": .058, "case": (.0775, .0225), "holder": .03, "coaster": .045, "bin": .085,
                   "pile": (.125, .09)}  # B5 textbooks
# 8 mm tray rims, 5 cm toy-box walls, 1.5 cm pen-case walls, 8 cm holder, 6 mm coaster rim, 20 cm wastebasket.
WALL_HALF_HEIGHT = {"tray": .004, "box": .025, "case": .0075, "holder": .04, "coaster": .003, "bin": .10}
PILE_HEIGHT = .112  # four 2.8 cm textbooks: the hand, 0.21 m tall when its fingers close vertically, hangs beside it


def receptacle_half(style, zone=None):
    """Inner half extents (along the receptacle's own x, y axes); a zone's own size if it has one."""
    if zone is not None and getattr(zone, "half", None):
        return tuple(zone.half)
    half = RECEPTACLE_HALF[style]
    return tuple(half) if isinstance(half, tuple) else (half, half)


def look_at(eye, target):
    """MuJoCo camera quaternion (wxyz) looking from eye to target with world z up."""
    from scipy.spatial.transform import Rotation
    z = np.asarray(eye, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    q = Rotation.from_matrix(np.column_stack([x, np.cross(z, x), z])).as_quat()
    return [q[3], q[0], q[1], q[2]]


@dataclass
class ObjectSpec:
    color: str
    kind: str
    xy: tuple
    yaw: float = 0.0
    upright: bool = False   # a pen standing on its end (already in the holder at the start)
    lift: float = 0.0       # extra start height, e.g. the floor of the receptacle it starts in
    rgba: list = None       # an explicit colour (cdesk: a palette colour), else OBJECT_RGBA[color]
    variant: str = None     # a different model of the same kind (cdesk: a crumpled-looking paper ball)

    @property
    def name(self):
        return f"{self.color}_{self.kind}"


@dataclass
class ZoneSpec:
    color: str
    xy: tuple
    yaw: float = 0.0
    half: tuple = None  # inner half extents, if not the style's default (a letter tray for an A5 notebook)
    paint: str = None   # cdesk: a palette colour for the whole receptacle (base, walls, books); `color` keeps the style

    @property
    def name(self):
        return f"{self.paint}_{RECEPTACLES[self.color][0]}" if self.paint else RECEPTACLES[self.color][1]


@dataclass
class Layout:
    objects: list = field(default_factory=list)
    zones: list = field(default_factory=list)
    decor: bool = False  # static clutter at the back of the desk (realdesk)
    decor_books: bool = True  # nbdesk: the pile of books is a receptacle in reach instead
    stacks: list = field(default_factory=list)  # textbook piles: (x, y, height, yaw); a notebook may lie on one
    look: str = "plain"  # "real" (sdesk): things shaped and textured like real ones, same physics (looks.py)
    phone_side: int = 1  # the decor phone at the left (+1) or right (-1) near corner of the desk


class TabletopOrganize(Lift):
    """Reuses Panda/Lift physics; replaces the scene with a supplied layout."""

    def __init__(self, *args, layout=None, **kwargs):
        if layout is None:
            raise ValueError("TabletopOrganize requires a layout")
        self.layout = layout
        super().__init__(*args, **kwargs)

    def _load_model(self):
        ManipulationEnv._load_model(self)
        self.robots[0].robot_model.set_base_xpos(
            self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0]))
        arena = TableArena(table_full_size=self.table_full_size,
                           table_friction=self.table_friction, table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        # Perception camera straight above the table (image right = world +x, up = +y).
        arena.set_camera(camera_name="topdown", pos=[0.0, 0.0, 1.95], quat=[1, 0, 0, 0])
        arena.set_camera(camera_name="demo", pos=[1.0, -0.8, 1.5], quat=look_at([1.0, -0.8, 1.5], [0, 0, .82]))
        from .assets import PEN_HALF, add_bin, add_decor, add_receptacle, add_stack, make_object
        from .palette import NATURAL, PALETTE, shade
        for zone in self.layout.zones:
            style = RECEPTACLES[zone.color][0]
            # cdesk / sdesk: the whole thing coloured
            paint = {**NATURAL, **PALETTE}[zone.paint] if getattr(zone, "paint", None) else None
            look = getattr(self.layout, "look", "plain")
            base = paint or ZONE_RGBA.get(zone.color)
            wall = shade(paint, .8) if paint else None
            if style == "bin":
                add_bin(arena.worldbody, zone.name, base, zone.xy, TABLE_Z,
                        radius=RECEPTACLE_HALF["bin"], half_height=WALL_HALF_HEIGHT["bin"], wall_rgba=wall,
                        look=look, asset=arena.asset)
                continue
            if style == "pile":
                add_stack(arena.worldbody, zone.name, zone.xy, TABLE_Z, PILE_HEIGHT, half=receptacle_half(style, zone),
                          yaw=zone.yaw, covers=[shade(paint, f) for f in (.75, 1.0, .85, 1.1)] if paint else None,
                          look=look, asset=arena.asset, paint=paint)
                continue
            add_receptacle(arena.worldbody, zone.name, style, base,
                           zone.xy, TABLE_Z, half=receptacle_half(style, zone), yaw=zone.yaw,
                           wall_half_height=WALL_HALF_HEIGHT[style], wall_rgba=wall, look=look, asset=arena.asset)
        if getattr(self.layout, "decor", False):
            add_decor(arena.worldbody, TABLE_Z, books=getattr(self.layout, "decor_books", True),
                      phone_side=getattr(self.layout, "phone_side", 1),
                      look=getattr(self.layout, "look", "plain"), asset=arena.asset)
        for i, (x, y, height, yaw) in enumerate(getattr(self.layout, "stacks", [])):
            add_stack(arena.worldbody, f"stack_{i}", (x, y), TABLE_Z, height, yaw=yaw)
        self.items = {}
        self.placement_initializer = SequentialCompositeSampler("LayoutSampler")
        for spec in self.layout.objects:
            obj = make_object(spec.kind, spec.name, getattr(spec, "rgba", None) or OBJECT_RGBA[spec.color],
                              getattr(spec, "variant", None), look=getattr(self.layout, "look", "plain"))
            self.items[spec.name] = obj
            upright = getattr(spec, "upright", False)
            # A standing pen: turned about y by 90 degrees, so its long axis is vertical; the sampler places the
            # body origin at z_offset + half height, so add the rest of the half length.
            z = .002 + getattr(spec, "lift", 0.0) + (PEN_HALF[0] - PEN_HALF[2] if upright else 0.0)
            self.placement_initializer.append_sampler(UniformRandomSampler(
                name=spec.name + "_sampler", mujoco_objects=obj,
                x_range=[spec.xy[0], spec.xy[0]], y_range=[spec.xy[1], spec.xy[1]],
                rotation=np.pi / 2 if upright else spec.yaw, rotation_axis="y" if upright else "z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False, reference_pos=self.table_offset,
                z_offset=z, rng=self.rng))
        self.cube = next(iter(self.items.values()))  # Inherited Lift helpers use this name.
        self.model = ManipulationTask(mujoco_arena=arena,
                                      mujoco_robots=[robot.robot_model for robot in self.robots],
                                      mujoco_objects=list(self.items.values()))

    def _setup_references(self):
        super()._setup_references()
        self.item_body_ids = {name: self.sim.model.body_name2id(obj.root_body)
                              for name, obj in self.items.items()}

    def _setup_observables(self):
        # Robot proprioception only; Lift's object-state observables are skipped.
        return ManipulationEnv._setup_observables(self)

    def reward(self, action=None):
        return 0.0

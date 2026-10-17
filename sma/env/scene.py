"""Robosuite tabletop with varied layouts: toy blocks, mugs and books, plus trays and a toy box.

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
# Object colors (segmented above the table) and zone colors (segmented at table height) are disjoint.
OBJECT_RGBA = {"red": [.85, .08, .06, 1], "green": [.08, .62, .12, 1], "blue": [.06, .18, .85, 1],
               "yellow": [.92, .78, .05, 1]}
ZONE_RGBA = {"purple": [.55, .12, .70, 1], "cyan": [.05, .72, .78, 1], "orange": [.95, .45, .05, 1]}
# Receptacle catalogue: base colour -> (style, name). Trays are shallow; the toy box has walls.
RECEPTACLES = {"orange": ("tray", "orange_tray"), "cyan": ("tray", "cyan_tray"), "purple": ("box", "purple_box")}
# Object half heights (m), used by the evaluator.
KIND_HALF_HEIGHT = {"block": .02, "book": .0125, "mug": .035}
KIND_RADIUS = {"block": .028, "book": .054, "mug": .04}
ZONE_HALF = .055                               # default inner half width (toy box)
RECEPTACLE_HALF = {"tray": .068, "box": .058}  # trays are larger so a rotated book still lies flat


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

    @property
    def name(self):
        return f"{self.color}_{self.kind}"


@dataclass
class ZoneSpec:
    color: str
    xy: tuple

    @property
    def name(self):
        return RECEPTACLES[self.color][1]


@dataclass
class Layout:
    objects: list = field(default_factory=list)
    zones: list = field(default_factory=list)


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
        from .assets import add_receptacle, make_object
        for zone in self.layout.zones:
            style = RECEPTACLES[zone.color][0]
            add_receptacle(arena.worldbody, zone.name, style, ZONE_RGBA[zone.color],
                           zone.xy, TABLE_Z, half=RECEPTACLE_HALF[style])
        self.items = {}
        self.placement_initializer = SequentialCompositeSampler("LayoutSampler")
        for spec in self.layout.objects:
            obj = make_object(spec.kind, spec.name, OBJECT_RGBA[spec.color])
            self.items[spec.name] = obj
            self.placement_initializer.append_sampler(UniformRandomSampler(
                name=spec.name + "_sampler", mujoco_objects=obj,
                x_range=[spec.xy[0], spec.xy[0]], y_range=[spec.xy[1], spec.xy[1]],
                rotation=spec.yaw, ensure_object_boundary_in_range=False,
                ensure_valid_placement=False, reference_pos=self.table_offset,
                z_offset=.002, rng=self.rng))
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

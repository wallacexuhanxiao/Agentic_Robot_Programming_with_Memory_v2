"""Pose-level robot API over the robosuite tabletop, plus a private evaluator.

Agent-visible: top-down RGB-D perception, gripper/end-effector proprioception, and the
results of its own commands. Private (evaluator only): simulator object poses.
"""
import os

import numpy as np
from scipy.spatial.transform import Rotation

from .perception import (OBJECT_HUES, _footprint, _palette_label, inside_zone, object_on_zone,
                         object_pixels, perceive)
from .scene import KIND_HALF_HEIGHT, RECEPTACLES, TABLE_Z, TabletopOrganize, receptacle_half

HOME = np.array([-0.20, 0.0, 1.08])
# x, y reach past the desk edges (at +-0.40 m): a notebook sticking out over an edge is pinched from outside the desk.
WORKSPACE = np.array([[-0.50, 0.30], [-0.52, 0.52], [TABLE_Z + 0.003, 1.30]])  # a 1 cm notebook's mid-plane: 5 mm


def _wrap_half_pi(angle):
    """Direction of a symmetric axis (finger line, long axis): the angle modulo pi, in [-pi/2, pi/2)."""
    return float((angle + np.pi / 2) % np.pi - np.pi / 2)


def _closing_yaw(rotation):
    """Yaw of the finger closing line, or None when the fingers close (nearly) vertically (a side pinch), where a
    yaw means nothing and motions are not checked against one."""
    closing = rotation.as_matrix()[:, 0]
    return float(np.arctan2(closing[1], closing[0])) if np.hypot(closing[0], closing[1]) > .3 else None


def _rot2(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


class RobotError(RuntimeError):
    """A command that could not be executed; carries structured details for feedback. A program may raise one itself,
    also as RobotError("message", reason="...")."""

    def __init__(self, *args, **details):
        reason = str(args[0]) if args else str(details.pop("reason", "error"))
        if args and "reason" in details:  # RobotError("curobo_plan_failed", reason=note): the note is a detail
            details["reason_detail"] = details.pop("reason")
        super().__init__(f"{reason}: {details}" if details else reason)
        self.reason = reason
        self.details = details


class BudgetExceeded(RobotError):
    pass


class TabletopEnv:
    def __init__(self, task, *, image_size=None, video=True, video_every=3, step_budget=2400,
                 render_gpu=None):
        self.task = task
        # Perception backend: "color" (hue segmentation), "gsam" (Grounding DINO + SAM2 service) or "sam" (SAM2 masks with
        # no text or categories, separated and measured by depth: things are not told apart by colour).
        self.perception = os.environ.get("SMA_PERCEPTION", "color")
        self.image_size = image_size or (512 if self.perception in ("gsam", "sam") else 256)
        self.video = video
        self.video_every = video_every
        self.step_budget = step_budget
        self.render_gpu = int(os.environ.get("MUJOCO_EGL_DEVICE_ID", -1)) if render_gpu is None else render_gpu
        self.frames = []
        self.env = None
        # Colour-coded desk: the camera names things by colour only; what each is comes from set_identity().
        if getattr(task, "family", None) == "sdesk":  # shared colours: only class-agnostic segmentation tells things apart
            self.perception = "sam"
            self.image_size = image_size or 512
        self.neutral = getattr(task, "family", None) in ("cdesk", "sdesk") or os.environ.get("SMA_NEUTRAL") == "1" or \
            self.perception == "sam"
        self.identity = {}
        self._robot_geoms = None

    # ---------------------------------------------------------------- lifecycle
    def reset(self):
        import robosuite as suite
        from robosuite import load_composite_controller_config
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
        if self.env is not None:
            self.close()
        if os.environ.get("SMA_CPU_RENDER") == "1":  # no GPU in this job: Mesa software rendering
            from .cpu_gl import enable_surfaceless
            enable_surfaceless()
        from robosuite.controllers import load_part_controller_config
        config = load_composite_controller_config(controller="BASIC")
        # Joint-position control: every motion is solved by IK, then tracked in joint space.
        arm = load_part_controller_config(default_controller="JOINT_POSITION")
        arm["kp"] = 400
        config["body_parts"]["right"] = {**arm, "gripper": config["body_parts"]["right"]["gripper"]}
        self.env = suite.make("TabletopOrganize", robots="Panda", layout=self.task.layout,
                              controller_configs=config, has_renderer=False, has_offscreen_renderer=True,
                              use_camera_obs=False, use_object_obs=False, control_freq=20,
                              horizon=100000, ignore_done=True, hard_reset=False, seed=self.task.seed,
                              render_gpu_device_id=self.render_gpu)
        self.obs = self.env.reset()
        self.grip = -1.0
        self.steps = 0
        self.frames = []
        sim = self.env.sim
        self.K = get_camera_intrinsic_matrix(sim, "topdown", self.image_size, self.image_size)
        self.world_from_camera = get_camera_extrinsic_matrix(sim, "topdown")
        self.down = Rotation.from_quat(self.obs["robot0_eef_quat_site"])
        # Finger closing axis at the reset orientation defines yaw = 0 offset.
        closing = self.down.as_matrix()[:, 0]  # Panda fingers slide along the grip-site x axis.
        self.yaw0 = float(np.arctan2(closing[1], closing[0]))
        self._setup_ik()
        # sam perception: names of things across views, the thing in the hand, the one just let go of, and each
        # receptacle's best view
        self._tracks, self._next_thing, self._held_track, self._released, self._zone_best = {}, 1, None, None, {}
        self._own_size = {}  # each object's size as first seen (_note_sizes)
        self.q_rest = self._q()
        self.wait(10)
        self.go_home()
        self.steps = 0
        return self.observe()

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    # ---------------------------------------------------------------- low level
    def _ee(self):
        return np.asarray(self.obs["robot0_eef_pos"], float).copy()

    def _yaw(self):
        closing = Rotation.from_quat(self.obs["robot0_eef_quat_site"]).as_matrix()[:, 0]
        return float(np.arctan2(closing[1], closing[0]))

    def _target_rotation(self, yaw):
        return Rotation.from_euler("z", yaw - self.yaw0) * self.down

    # Joint-space execution -------------------------------------------------
    def _setup_ik(self):
        import mujoco
        self._mj = mujoco
        model = self.env.sim.model._model
        self._model = model
        self._ik_data = mujoco.MjData(model)  # private copy: only the robot's own joints are ever written
        joints = [model.joint(i).name for i in range(model.njnt) if model.joint(i).name.startswith("robot0_joint")]
        ids = [model.joint(n).id for n in joints]
        self.qadr = np.array([model.jnt_qposadr[i] for i in ids])
        self.dadr = np.array([model.jnt_dofadr[i] for i in ids])
        self.q_low, self.q_high = model.jnt_range[ids, 0] + .02, model.jnt_range[ids, 1] - .02
        self.site_id = model.site("gripper0_right_grip_site").id

    def _q(self):
        return self.env.sim.data._data.qpos[self.qadr].copy()

    def _ik(self, position, rotation, q0=None, iterations=150):
        """Damped least squares on the robot model, within joint limits, biased toward the rest posture."""
        mj, model, d = self._mj, self._model, self._ik_data
        d.qpos[:] = model.qpos0  # default configuration; object poses are never copied into the IK model
        q = self._q() if q0 is None else np.array(q0, float)
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        for _ in range(iterations):
            d.qpos[self.qadr] = q
            mj.mj_fwdPosition(model, d)
            ep = position - d.site_xpos[self.site_id]
            er = (rotation * Rotation.from_matrix(d.site_xmat[self.site_id].reshape(3, 3)).inv()).as_rotvec()
            if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(er) < 5e-3:
                return q, 0.0
            mj.mj_jacSite(model, d, jacp, jacr, self.site_id)
            J = np.vstack([jacp[:, self.dadr], jacr[:, self.dadr]])
            e = np.concatenate([ep, .5 * er])
            JJt = J @ J.T + 1e-4 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            null = np.eye(len(q)) - J.T @ np.linalg.solve(JJt, J)
            dq += null @ (.05 * (self.q_rest - q))
            q = np.clip(q + np.clip(dq, -.15, .15), self.q_low, self.q_high)
        return q, float(np.linalg.norm(ep) + .1 * np.linalg.norm(er))

    def _plan(self, target, goal, residual_tol=.004):
        """Straight Cartesian line (with slerped orientation) as warm-started IK waypoints ~2 cm apart."""
        from scipy.spatial.transform import Slerp
        p0 = self._ee()
        r0 = Rotation.from_quat(self.obs["robot0_eef_quat_site"])
        angle = np.linalg.norm((goal * r0.inv()).as_rotvec())
        n = max(1, int(np.ceil(np.linalg.norm(target - p0) / .02)), int(np.ceil(angle / .08)))
        slerp = Slerp([0, 1], Rotation.concatenate([r0, goal]))
        q, path = self._q(), []
        for i in range(1, n + 1):
            q, residual = self._ik(p0 + (target - p0) * i / n, slerp(i / n), q0=q)
            if residual > residual_tol:
                return None, {"failed_at_fraction": round(i / n, 2), "ik_residual": round(residual, 4),
                              "waypoint": (p0 + (target - p0) * i / n).round(4).tolist()}
            path.append(q)
        return path, None

    def _step(self, q_target):
        if self.steps >= self.step_budget:
            raise BudgetExceeded("control_step_budget_exhausted", budget=self.step_budget)
        action = np.zeros(self.env.action_dim)
        action[:7] = np.clip((np.asarray(q_target) - self._q()) / .05, -1, 1)
        action[-1] = self.grip
        self.obs, _, _, _ = self.env.step(action)
        self.steps += 1
        if self.video and self.steps % self.video_every == 0:
            frame = self.render("demo", 480, 360)
            hook = getattr(self, "frame_hook", None)
            self.frames.append(hook(frame) if hook else frame)

    def render(self, camera, width, height, depth=False):
        out = self.env.sim.render(camera_name=camera, width=width, height=height, depth=depth)
        if depth:
            rgb, d = out
            return rgb[::-1].copy(), d[::-1].copy()
        return out[::-1].copy()

    # ---------------------------------------------------------------- agent API
    def observe(self):
        rgb, depth = self.render("topdown", self.image_size, self.image_size, depth=True)
        raw = np.asarray(depth, float)
        self.depth_raw_range = [float(np.nanmin(raw)), float(np.nanmax(raw))]
        if not (np.isfinite(raw).mean() > .5 and -1e-3 <= self.depth_raw_range[0] and self.depth_raw_range[1] <= 1.001):
            rgb, depth = self.render("topdown", self.image_size, self.image_size, depth=True)  # re-render once
            self.depth_rerendered = getattr(self, "depth_rerendered", 0) + 1
        depth_m = self._metric_depth(depth)
        if self.perception == "gsam":
            from .perception_gsam import perceive_gsam
            scene = perceive_gsam(rgb, depth_m, self.K, self.world_from_camera, os.environ["SMA_SEG_URL"])
        elif self.neutral:
            scene = self._perceive_neutral(rgb, depth_m)
        else:
            self._masks = {}
            scene = perceive(rgb, depth_m, self.K, self.world_from_camera, masks_out=self._masks)
        self._table_z = scene["table_z"]
        scene["gripper"] = self.gripper_state()
        scene["end_effector"] = {"position": [round(float(v), 4) for v in self._ee()],
                                 "yaw": round(self._yaw(), 3)}
        # Objects keep the name they were first seen with (one object per colour): a pen knocked onto a rim can be
        # classified as another kind, which would otherwise rename it mid-task.
        names = getattr(self, "_names_by_color", {})
        renamed = {}
        for name, o in scene["objects"].items():
            first = names.setdefault(o.get("color"), name)
            if first != name:
                o = {**o, "name": first, "perceived_kind": o["kind"], "kind": first.split("_", 1)[1]}
            renamed[first] = o
        scene["objects"], self._names_by_color = renamed, names
        self._zones_seen = {**getattr(self, "_zones_seen", {}), **scene["zones"]}
        # Receptacles do not move: one hidden by the arm right now is reported from the last view that showed it.
        for name, zone in self._zones_seen.items():
            if name not in scene["zones"]:
                scene["zones"][name] = {**zone, "visible_now": False}
        self.last_rgb = rgb
        self._note_sizes(scene)
        return scene

    def _note_sizes(self, scene):
        """Each object's own size, measured the first time the camera sees it at rest (nothing in the hand): its height
        above what it rests on (the desk, the top of a flat receptacle or pile, the floor of a hollow one) and its length
        (the longer of its footprint and that height: a pen standing in a holder shows its length as height). The camera
        checks compare later views with these, so no table of nominal object sizes is needed."""
        sizes = getattr(self, "_own_size", None)
        if sizes is None or scene["gripper"]["holding_something"]:
            return
        for name, obj in scene["objects"].items():
            if name in sizes:
                continue
            base = 0.0
            for zone in scene["zones"].values():
                if zone.get("visible_now", True) and inside_zone(obj["center"], zone, .005):
                    base = zone.get("top_height") or (zone.get("inner_floor_height") if zone.get("hollow") else 0.0) or 0.0
                    break
            own = max(float(obj["height"]) - float(base), .003)
            sizes[name] = {"height": round(own, 4), "length": round(max(float(obj["footprint_long"]), own), 4)}

    # ---------------------------------------------------------------- colour-coded desk (neutral perception)
    def _hues(self):
        """The colour palette perception uses here, with its tolerance."""
        if self.neutral:
            from .palette import PALETTE_TOLERANCE, RENDERED_HUE
            return RENDERED_HUE, PALETTE_TOLERANCE
        return OBJECT_HUES, 9

    def set_identity(self, identity):
        """identity: {colour: {"category": <VOCABULARY word or "other: ...">, "role": "object"|"receptacle"|"other"}},
        the model's own reading of the scene. From then on observe() reports the things it calls objects under
        "objects" and the ones it calls receptacles under "zones", with its category words."""
        self.identity = {k: dict(v) for k, v in (identity or {}).items()}

    def _perceive_neutral(self, rgb, depth_m):
        from .naming import KIND_OF
        from .palette import PALETTE_TOLERANCE, RENDERED_HUE
        from .perception import perceive_things
        self._masks = {}
        if self.perception == "sam":
            from .perception_sam import perceive_sam
            released = self._tracks.get(self._released)  # let go of last (it stays where it was put until grasped)
            expect = None if not released or released["turned_over"] else \
                {"center": released["center"], "pixels": released["visible_pixels"], "seen_z": released["seen_z"]}
            raw = perceive_sam(rgb, depth_m, self.K, self.world_from_camera, os.environ["SMA_SEG_URL"],
                               masks_out=self._masks, exclude=self._robot_pixels(), expect=expect)
            raw["things"] = self._track(raw["things"])
        else:
            raw = perceive_things(rgb, depth_m, self.K, self.world_from_camera, RENDERED_HUE, PALETTE_TOLERANCE,
                                  masks_out=self._masks)
        objects, zones, others = {}, {}, {}
        for name, t in raw["things"].items():
            ident = self.identity.get(name) or {}
            category, role = ident.get("category"), ident.get("role")
            if role == "receptacle":
                if self.perception == "sam":  # receptacles do not move: one partly covered now (a notebook laid on
                    # the pile) is measured from the view that showed most of it
                    best = self._zone_best.setdefault(name, t)
                    if t["visible_pixels"] >= best["visible_pixels"]:
                        self._zone_best[name] = t
                    elif t["visible_pixels"] < .85 * best["visible_pixels"]:
                        t = best
                zones[name] = self._as_zone(t, category, raw["table_z"])
            elif role == "object" or not self.identity:  # before identification every thing is a candidate object
                objects[name] = {**t, "category": category, "kind": KIND_OF.get(category, category)}
            else:
                others[name] = {**t, "category": category}
        return {"table_z": raw["table_z"], "objects": objects, "zones": zones, "others": others,
                "things": raw["things"]}

    def _robot_pixels(self):
        """Where the robot itself is in the top-down image: its own model drawn at its current joint angles (a real
        robot has the same from its URDF and joint encoders), so that no part of the arm is taken for a thing."""
        import mujoco
        model = self.env.sim.model._model
        if self._robot_geoms is None:
            self._robot_geoms = np.array([model.body(int(model.geom_bodyid[i])).name.startswith(
                ("robot0_", "gripper0_", "mount0_")) for i in range(model.ngeom)] + [False])
        seg = self.env.sim.render(camera_name="topdown", width=self.image_size, height=self.image_size,
                                  segmentation=True)[::-1]
        # The first ordinary render after a segmentation render comes out garbled (tiles of the wrong place, textures
        # on the wrong things): render once and throw it away, so no camera image, video frame or close-up gets it.
        self.env.sim.render(camera_name="topdown", width=self.image_size, height=self.image_size)
        ids = np.where(seg[..., 1] >= 0, seg[..., 1], model.ngeom)
        return (seg[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)) & self._robot_geoms[np.clip(ids, 0, model.ngeom)]

    def _note_held(self):
        """The fingers just closed on something: the tracked thing nearest the grip point is taken to move with the hand
        until the gripper opens."""
        ee = self._ee()[:2]
        near = sorted(self._tracks, key=lambda n: np.hypot(*(np.asarray(self._tracks[n]["center"][:2]) - ee)))
        if near and np.hypot(*(np.asarray(self._tracks[near[0]]["center"][:2]) - ee)) < .12:
            self._held_track, self._released = near[0], None
            track = self._tracks[near[0]]
            # held rigidly: its centre keeps this offset in the hand's frame (it turns with the hand)
            self._held_rotation = self._hand_rotation()
            self._held_offset = self._held_rotation.T @ (np.asarray(track["center"], float) - self._ee())
            # how it looked on the desk: in the hand the fingers hide part of it and it is closer to the camera
            self._held_look = {"lab": track["lab"], "visible_pixels": track["visible_pixels"], "seen_z": track["center"][2]}

    def _hand_rotation(self):
        return Rotation.from_quat(self.obs["robot0_eef_quat_site"]).as_matrix()

    def _carried_center(self):
        """Where the centre of the thing in the hand is now (see _note_held)."""
        return self._ee() + self._hand_rotation() @ self._held_offset

    def _track(self, things):
        """Names for the things SAM2 separated ("thing1", "thing2", ...), the same across views: each thing seen now is
        matched (Hungarian) to one seen before by its mean colour (CIELAB, lightness weighted half: shading changes it),
        how far it is from where it was expected (where it was last seen; a held thing goes with the hand) and its size
        in the image. Only the thing in the hand or one put down and not seen since may have moved far: any other must be
        within 10 cm of where it was (things move only when the robot moves them). The thing seen where the thing let go
        of last lies (until the next grasp), at about its size, takes its name whatever its colour now (it looks lighter
        or darker on a coaster, in a basket); one turned over in the hand, by where it was let go of alone. A thing that
        matches none gets a new name; one not seen now (hidden by the arm) keeps its name."""
        from scipy.optimize import linear_sum_assignment
        tracks, held = self._tracks, getattr(self, "_held_track", None)
        if held is not None and not self.gripper_state()["holding_something"]:
            held = self._held_track = None
        names = list(tracks)
        carried = self._carried_center()[:2] if held is not None else None
        cost = np.full((len(things), len(names)), 1e3)
        for i, t in enumerate(things):
            for j, n in enumerate(names):
                tr = tracks[n]
                d = np.asarray(t["lab"]) - tr["lab"]
                de = float(np.sqrt((d[0] / 2) ** 2 + d[1] ** 2 + d[2] ** 2))
                dist = float(np.hypot(*(np.asarray(t["center"][:2]) - (carried if n == held else np.asarray(tr["center"][:2])))))
                size = abs(np.log(max(t["visible_pixels"], 1) / max(tr["visible_pixels"], 1)))
                if n == self._released and tr.get("turned_over"):
                    # let go of after the hand turned it over (a pen stood up in a holder): from above it looks nothing
                    # like it did on the desk (its top, not its length; neither colour nor size compare), so only where
                    # it was let go of counts, within 6 cm (a pen leaning 30-40 degrees in a holder has its top 4-5 cm
                    # from where its middle was); the things already there keep their own names (one to one)
                    if dist < .06:
                        cost[i, j] = dist / .05
                elif (de < 15 or (dist < .03 and de < 30)) and (dist < .10 or n == held or tr.get("put_down")):
                    cost[i, j] = de / 8 + dist / .05 + .3 * size
                if n == self._released and t.get("released_here"):
                    cost[i, j] = 0.0
        pairs = {}
        if things and names:
            rows, cols = linear_sum_assignment(cost)
            pairs = {int(i): names[j] for i, j in zip(rows, cols) if cost[i, j] < 1e3}
        out = {}
        for i in sorted(range(len(things)), key=lambda i: (-things[i]["center"][1], things[i]["center"][0])):
            t = things[i]
            name = pairs.get(i)
            if name is None:
                name = f"thing{self._next_thing}"
                self._next_thing += 1
            mask = t.pop("mask")
            lab = t.pop("lab")
            t.pop("released_here", None)
            t["name"] = t["color"] = name  # "color": the key of its pixel mask, as on the colour-coded desk
            self._masks[name] = mask
            tracks[name] = {"lab": np.asarray(lab, float), "center": t["center"], "visible_pixels": t["visible_pixels"],
                            "seen_z": t["center"][2], "turned_over": False}
            out[name] = t
        return out

    @staticmethod
    def _as_zone(t, category, table_z):
        """A thing read as a receptacle: its footprint (inside the walls of a hollow one, about 8 mm thick) as the place
        things go, and the height things rest at: the inner floor of a hollow one, the top of a flat one."""
        wall = .008 if t["hollow"] else 0.0
        top = t.get("top_footprint") or {}  # the top surface or rim only: its sides seen in perspective widen the rest
        cx, cy = top.get("center", t["center"][:2])
        zone = {"name": t["name"], "color": t["color"], "category": category, "kind": category,
                "center": [cx, cy, round(table_z, 4)],
                "half_size": [round(max(top.get("long", t["footprint_long"]) / 2 - wall, .005), 4),
                              round(max(top.get("short", t["footprint_short"]) / 2 - wall, .005), 4)],
                "long_axis_yaw": top.get("long_axis_yaw", t["long_axis_yaw"]), "shape": t["shape"], "hollow": t["hollow"],
                "height": t["height"]}
        if t["hollow"]:
            zone.update({"rim_height": t["rim_height"], "inner_floor_height": t["inner_floor_height"]})
        else:
            zone.update({"top_height": t["height"], "rim_height": t["height"]})
        return zone

    def _project_points(self, camera, points, h, w):
        """World points -> (row, col) in render(camera, w, h) images."""
        from robosuite.utils.camera_utils import (get_camera_intrinsic_matrix, get_camera_transform_matrix,
                                                  project_points_from_world_to_camera)
        points = np.asarray(points, float)
        if camera == "topdown":  # the inverse of perception's back-projection (the convention it is checked in)
            K = get_camera_intrinsic_matrix(self.env.sim, camera, h, w)
            c = (points - self.world_from_camera[:3, 3]) @ self.world_from_camera[:3, :3]
            return np.column_stack([K[1, 1] * c[:, 1] / c[:, 2] + K[1, 2], K[0, 0] * c[:, 0] / c[:, 2] + K[0, 2]])
        # robosuite's projection already gives (row, col) in the upright image render() returns (flipping it put the
        # demo-view tags on the wrong things)
        return np.asarray(project_points_from_world_to_camera(points, get_camera_transform_matrix(self.env.sim, camera, h, w),
                                                              h, w), float)

    def labelled_views(self, labels=None, size=512, marks=None):
        """The top-down camera and the demo camera (from the side) with each thing marked by a small tag placed beside it
        and a thin line to it, so that no tag covers a thing: images for a model to recognise the things by (PNG bytes).
        labels: {thing name: tag text} (default: the names). The things are those of the latest observe().
        marks "som" (default SMA_MARKS, else "tags"): also each thing's outline (its camera mask from above, the hull of its
        depth points projected into the side view), and a third image with every thing close up (_closeups), as in
        Set-of-Mark prompting."""
        import cv2
        marks = marks or os.environ.get("SMA_MARKS", "tags")
        scene = self.observe()
        things = scene.get("things") or {**scene["objects"], **scene["zones"]}
        labels = labels or {n: n for n in things}
        names = [n for n in things if n in labels]
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, .5
        views = []
        for camera, (w, h) in (("topdown", (size, size)), ("demo", (640, 480))):
            image = np.ascontiguousarray(self.render(camera, w, h))
            if marks == "som":
                for n in names:
                    outline = self._thing_outline(n, camera, h, w)
                    if outline is not None:
                        for colour, width in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                            cv2.drawContours(image, outline, -1, colour, width, cv2.LINE_AA)
            if names:
                top = [things[n].get("top_z", things[n]["center"][2]) for n in names]
                centre = np.array([[things[n]["center"][0], things[n]["center"][1], z] for n, z in zip(names, top)])
                side = centre + np.array([[things[n]["footprint_long"] / 2, 0, 0] for n in names])
                px = self._project_points(camera, np.vstack([centre, side]), h, w)
                at, radius = px[:len(names)], np.linalg.norm(px[len(names):] - px[:len(names)], axis=1)
                occupied = np.zeros((h, w), bool)  # where things are: tags go elsewhere
                if camera == "topdown" and getattr(self, "_masks", None):
                    for n in names:
                        if n in self._masks:
                            occupied |= cv2.resize(self._masks[n].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                    occupied = cv2.dilate(occupied.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                else:
                    rows, cols = np.mgrid[:h, :w]
                    for (r, c), rad in zip(at, radius):
                        occupied |= (rows - r) ** 2 + (cols - c) ** 2 < (rad + 6) ** 2
                taken = np.zeros((h, w), bool)
                for i in np.argsort(radius):
                    text = str(labels[names[i]])
                    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
                    bw, bh = tw + 6, th + 8
                    r0, c0 = at[i]
                    best = None
                    for dist in (radius[i] + 10, radius[i] + 22, radius[i] + 36, radius[i] + 54):
                        for angle in np.radians([-45, -135, 45, 135, -90, 90, 0, 180]):
                            cy, cx = r0 + dist * np.sin(angle), c0 + dist * np.cos(angle)
                            y0, x0 = int(cy - bh / 2), int(cx - bw / 2)
                            if y0 < 0 or x0 < 0 or y0 + bh >= h or x0 + bw >= w:
                                continue
                            cost = occupied[y0:y0 + bh, x0:x0 + bw].sum() + 50 * taken[y0:y0 + bh, x0:x0 + bw].sum()
                            if best is None or cost < best[0]:
                                best = (cost, y0, x0, cy, cx)
                        if best is not None and best[0] == 0:
                            break
                    if best is None:
                        continue
                    _, y0, x0, cy, cx = best
                    taken[max(0, y0 - 3):y0 + bh + 3, max(0, x0 - 3):x0 + bw + 3] = True
                    d = np.array([cy - r0, cx - c0])
                    rim = np.array([r0, c0]) + d / max(np.linalg.norm(d), 1e-6) * radius[i] * .6
                    for colour, width in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                        cv2.line(image, (int(rim[1]), int(rim[0])), (int(cx), int(cy)), colour, width, cv2.LINE_AA)
                    cv2.rectangle(image, (x0, y0), (x0 + bw, y0 + bh), (255, 255, 255), -1)
                    cv2.rectangle(image, (x0, y0), (x0 + bw, y0 + bh), (0, 0, 0), 1)
                    cv2.putText(image, text, (x0 + 3, y0 + bh - 5), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
            ok, png = cv2.imencode(".png", image[..., ::-1])
            views.append({"camera": camera, "png": png.tobytes(), "image": image})
        if marks == "som" and names:
            views.append(self._closeups(names, labels))
        return views

    def _thing_outline(self, name, camera, h, w):
        """A thing's outline in render(camera, w, h) as OpenCV contours: its top-down mask, scaled (top-down camera), or the
        convex hull of those of its depth points the other camera sees (not hidden behind something nearer, by that
        camera's own depth image). None when there is no mask of it, or another camera sees less than a sixth of it."""
        import cv2
        mask = getattr(self, "_masks", {}).get(name)
        if mask is None or not mask.any():
            return None
        if camera == "topdown":
            m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            return cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] or None
        pts = self._masks["_xyz"][mask]
        pts = pts[np.isfinite(pts).all(1)]
        if len(pts) < 3:
            return None
        pts = pts[np.linspace(0, len(pts) - 1, min(len(pts), 400)).astype(int)]
        px = self._project_points(camera, pts, h, w)  # (row, col)
        seen = self._seen_from(camera, pts, px, h, w)
        if seen.sum() < max(3, len(pts) / 6):
            return None
        return [cv2.convexHull(np.round(px[seen][:, ::-1]).astype(np.int32))]

    def _seen_from(self, camera, pts, px, h, w, tolerance=.015):
        """Which world points (projected to px, (row, col)) camera sees: their distance along its axis within tolerance of
        its depth image there (its depth render is kept for the current control step)."""
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
        key = (camera, w, h, self.steps)
        if getattr(self, "_depth_cache", (None,))[0] != key:
            _, depth = self.render(camera, w, h, depth=True)
            self._depth_cache = (key, self._metric_depth(depth), get_camera_extrinsic_matrix(self.env.sim, camera))
        _, depth_m, pose = self._depth_cache
        z = ((pts - pose[:3, 3]) @ pose[:3, :3])[:, 2]
        r = np.clip(np.round(px[:, 0]).astype(int), 0, h - 1)
        c = np.clip(np.round(px[:, 1]).astype(int), 0, w - 1)
        inside = (px[:, 0] >= 0) & (px[:, 0] < h) & (px[:, 1] >= 0) & (px[:, 1] < w)
        return inside & (np.abs(depth_m[r, c] - z) < tolerance)

    def _closeups(self, names, labels, tile=176, per_row=3):
        """One image with every thing close up: its tag text, then the top-down camera (1024 px render) and the demo
        camera (1280 x 960) cropped square around it, each with its outline."""
        import cv2
        renders = {"topdown": (1024, 1024), "demo": (1280, 960)}
        images = {c: np.ascontiguousarray(self.render(c, w, h)) for c, (w, h) in renders.items()}
        cells = []
        for n in names:
            pair = []
            for camera, (w, h) in renders.items():
                outline = self._thing_outline(n, camera, h, w)
                if outline is None:  # hidden from this camera
                    blank = np.full((tile, tile, 3), 235, np.uint8)
                    cv2.putText(blank, "not visible", (30, tile // 2), cv2.FONT_HERSHEY_SIMPLEX, .55, (90, 90, 90), 1,
                                cv2.LINE_AA)
                    pair.append(blank)
                    continue
                pts = np.vstack([c.reshape(-1, 2) for c in outline])
                (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
                side = int(min(max(x1 - x0, y1 - y0) * 1.4 + 24, w, h))
                a = int(np.clip((x0 + x1) / 2 - side / 2, 0, w - side))
                b = int(np.clip((y0 + y1) / 2 - side / 2, 0, h - side))
                crop = cv2.resize(images[camera][b:b + side, a:a + side], (tile, tile),
                                  interpolation=cv2.INTER_AREA if side > tile else cv2.INTER_CUBIC)
                shifted = [np.round((c - [a, b]) * tile / side).astype(np.int32) for c in outline]
                for colour, width in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                    cv2.drawContours(crop, shifted, -1, colour, width, cv2.LINE_AA)
                pair.append(crop)
            cell = np.full((tile + 26, 2 * tile + 6, 3), 255, np.uint8)
            cell[26:, :tile], cell[26:, tile + 6:] = pair
            cv2.putText(cell, str(labels[n]), (4, 19), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2, cv2.LINE_AA)
            cells.append(cell)
        rows = -(-len(cells) // per_row)
        ch, cw = cells[0].shape[:2]
        sheet = np.full((rows * (ch + 10) + 10, per_row * (cw + 10) + 10, 3), 225, np.uint8)
        for i, cell in enumerate(cells):
            y, x = 10 + (i // per_row) * (ch + 10), 10 + (i % per_row) * (cw + 10)
            sheet[y:y + ch, x:x + cw] = cell
        ok, png = cv2.imencode(".png", sheet[..., ::-1])
        return {"camera": "closeups", "png": png.tobytes(), "image": sheet}

    def _metric_depth(self, depth):
        """MuJoCo's normalised depth buffer -> metres (same formula as robosuite), tolerant of values a
        float epsilon outside [0, 1]; non-finite pixels become NaN instead of failing an assertion."""
        d = np.asarray(depth, float)
        if d.ndim == 3:
            d = d[..., 0]
        model = self.env.sim.model
        extent = model.stat.extent
        near, far = model.vis.map.znear * extent, model.vis.map.zfar * extent
        bad = ~np.isfinite(d)
        d = np.clip(np.where(bad, 1.0, d), 0.0, 1.0)
        metres = near / (1.0 - d * (1.0 - near / far))
        metres[bad] = np.nan
        self.depth_out_of_range = int(bad.sum())
        return metres

    def gripper_state(self):
        q = np.asarray(self.obs["robot0_gripper_qpos"], float)
        width = float(abs(q[0] - q[1]))
        return {"width": round(width, 4), "closed_command": self.grip > 0,
                "holding_something": bool(self.grip > 0 and width > .006)}

    def move_to(self, x, y, z, yaw=None, tolerance=.006, max_steps=60):
        """Move the grip point (between the fingertips) to (x, y, z), gripper pointing down."""
        target = np.array([x, y, z], float)
        if not np.all(np.isfinite(target)):
            raise RobotError("non_finite_target", target=target.tolist())
        low, high = WORKSPACE[:, 0], WORKSPACE[:, 1]
        if np.any(target < low) or np.any(target > high):
            raise RobotError("target_outside_workspace", target=target.round(4).tolist(),
                             workspace=WORKSPACE.tolist())
        yaw = self._yaw() if yaw is None else float(yaw)
        # The two fingers are symmetric: yaw and yaw + pi are the same grasp. Use the
        # representative closest to the rest-pose yaw so the wrist stays mid-range.
        yaw = self.yaw0 + ((yaw - self.yaw0 + np.pi / 2) % np.pi - np.pi / 2)
        path, failure = self._plan(target, self._target_rotation(yaw))
        if path is None:
            raise RobotError("unreachable_pose", target=target.round(4).tolist(), yaw=round(yaw, 3), **failure)
        return self._follow(path, target, yaw, tolerance, max_steps)

    def _follow(self, path, target, yaw, tolerance=.006, max_steps=60):
        """Track joint waypoints, then wait until the grip point is at target and the wrist at yaw."""
        start = self.steps
        for q in path:  # waypoints ~2 cm / 0.08 rad apart, 4 control steps each (~0.1 m/s): small tracking lag
            for _ in range(4):
                self._step(q)
        q_goal = path[-1]

        def yaw_error():  # finger line is symmetric: compare modulo pi (None: fingers close vertically, no yaw)
            return 0.0 if yaw is None else abs((self._yaw() - yaw + np.pi / 2) % np.pi - np.pi / 2)

        # Done when the grip point is in place AND the wrist has finished turning (large turns lag behind the
        # waypoints; returning on position alone left the gripper up to ~50 degrees short).
        for _ in range(max_steps):
            if np.linalg.norm(target - self._ee()) < tolerance and yaw_error() < .05:
                return {"ok": True, "position": self._ee().round(4).tolist(), "steps": self.steps - start}
            self._step(q_goal)
            if self.steps - start > len(path) + 40 and np.linalg.norm(self.obs.get("robot0_joint_vel", np.zeros(7))) < 1e-3:
                break
        if np.linalg.norm(target - self._ee()) < tolerance:  # in place, wrist still slightly off: report, not fail
            return {"ok": True, "position": self._ee().round(4).tolist(), "steps": self.steps - start,
                    "yaw_error_deg": round(float(np.degrees(yaw_error())), 1)}
        raise RobotError("motion_blocked", target=target.round(4).tolist(),
                         reached=self._ee().round(4).tolist(),
                         remaining_error_m=round(float(np.linalg.norm(target - self._ee())), 4))

    def _gripper(self, command, steps=25):
        self.grip = command
        hold = self._q()
        for _ in range(steps):
            self._step(hold)
        return self.gripper_state()

    def open_gripper(self):
        held = getattr(self, "_held_track", None)
        if held in getattr(self, "_tracks", {}):  # let go of here: look for it where the hand leaves it, as it looked
            # on the desk; turned over since the grasp (a pen stood up), its look from above is not known
            track = self._tracks[held]
            turned = self._hand_rotation() @ self._held_rotation.T
            track.update(self._held_look, center=self._carried_center().round(4).tolist(),
                         turned_over=bool(turned[2, 2] < np.cos(np.radians(20))), put_down=True)
            self._released = held
        self._held_track = None
        return {**self._gripper(-1.0), "gripper_yaw": round(self._yaw(), 3)}

    def close_gripper(self):
        out = self._gripper(1.0)
        if self.perception == "sam" and out.get("holding_something"):
            self._note_held()
        return out

    def go_home(self):
        """Raise the arm and park it at the table edge (out of the top camera's view). Best effort. While something
        is held the gripper keeps its yaw, so parking (e.g. inside free_spots or check) never turns the object."""
        yaw = None if self.gripper_state()["holding_something"] else self.yaw0
        try:
            if self._ee()[2] < HOME[2] - .03:
                here = self._ee()
                self.move_to(here[0], here[1], HOME[2], tolerance=.03)
            return self.move_to(*HOME, yaw=yaw, tolerance=.03)
        except RobotError as exc:
            if exc.reason == "motion_blocked" and exc.details["remaining_error_m"] < .06:
                return {"ok": True, "position": self._ee().round(4).tolist(), "approximate": True}
            tilted = abs(Rotation.from_quat(self.obs["robot0_eef_quat_site"]).as_matrix()[2, 2] + 1) > .1
            if exc.reason != "unreachable_pose" or not tilted:
                raise
        # From a tilted gripper (after place_upright) the straight-line path home can be out of reach: go back to
        # the rest posture in joint space first, then home.
        q0 = self._q()
        n = max(1, int(np.ceil(np.max(np.abs(self.q_rest - q0)) / .06)))
        for i in range(1, n + 1):
            for _ in range(4):
                self._step(q0 + (self.q_rest - q0) * i / n)
        self.wait(20)
        return self.move_to(*HOME, yaw=yaw, tolerance=.03)

    # ---------------------------------------------------------------- task-level API
    # Parameterized primitives built on the pose-level calls. The caller must choose the grasp
    # height and finger direction; nothing object-specific is decided here.
    def pick(self, object_name, grasp_height, yaw):
        """Top-down grasp of object_name with the grip point at table_z + grasp_height, fingers
        closing along yaw. Approaches from above, closes, lifts. Returns the gripper state."""
        grasp_height, yaw = float(grasp_height), float(yaw)
        if not .004 <= grasp_height <= .15:
            hint = ("grasp_height is the grip point's height ABOVE the table surface (table_z), not an absolute z"
                    if grasp_height > .5 else "grasp_height must be between 0.004 and 0.15 m above the table")
            raise RobotError("grasp_height_out_of_range", grasp_height=grasp_height, allowed=[.004, .15], hint=hint)
        if self.gripper_state()["holding_something"]:
            raise RobotError("already_holding_something")
        self.go_home()
        scene = self.observe()
        obj = scene["objects"].get(object_name)
        if obj is None:
            raise RobotError("object_not_visible", object=object_name, visible=sorted(scene["objects"]))
        x, y = obj["center"][:2]
        table = scene["table_z"]
        self.open_gripper()
        self.move_to(x, y, max(obj["top_z"], table + grasp_height) + .10, yaw)
        try:
            self.move_to(x, y, table + grasp_height, yaw)
        except RobotError as exc:  # touching down slightly early is fine; holding is checked after closing
            if exc.reason != "motion_blocked" or exc.details["remaining_error_m"] > .012:
                raise
        state = self.close_gripper()
        self._held_grasp_height = grasp_height
        self.move_to(x, y, table + .18, yaw)
        return {**self.gripper_state(), "grasp_point_z": round(table + grasp_height, 4),
                "object_top_z_before": obj["top_z"]}

    def place(self, zone_name, dx=0.0, dy=0.0):
        """Carry the held object above zone_name's centre (+dx, dy), lower it until it is just
        above the table, release, lift and park the arm."""
        if not self.gripper_state()["holding_something"]:
            raise RobotError("not_holding_anything")
        scene_zone = self._zone_from_last_view(zone_name)
        x, y = scene_zone["center"][0] + float(dx), scene_zone["center"][1] + float(dy)
        table = scene_zone["center"][2]
        here = self._ee()
        self.move_to(here[0], here[1], max(here[2], table + .18))
        self.move_to(x, y, table + .18)
        touched_early = None
        try:
            self.move_to(x, y, table + self._held_grasp_height + .006)
        except RobotError as exc:  # compliant placing: resting on something slightly early is acceptable
            if exc.reason != "motion_blocked" or exc.details["remaining_error_m"] > .015:
                raise
            touched_early = exc.details["remaining_error_m"]
        self.open_gripper()
        self.move_to(x, y, table + .18)
        self.go_home()
        return {"released_at": [round(x, 4), round(y, 4)], "touched_down_early_m": touched_early}

    def release(self):
        """Open the fingers where the arm is now, lift 10 cm and park the arm (recovery when stuck holding)."""
        self.open_gripper()
        here = self._ee()
        self.move_to(here[0], here[1], min(here[2] + .10, 1.2))
        self.go_home()
        return self.gripper_state()

    def _zone_from_last_view(self, zone_name):
        # Zones are flat and static; the view from before the grasp is used if the arm hides one now.
        scene = self.observe()
        zone = scene["zones"].get(zone_name) or getattr(self, "_zones_seen", {}).get(zone_name)
        if zone is None:
            raise RobotError("zone_not_visible", zone=zone_name, visible=sorted(scene["zones"]))
        return zone

    # ---------------------------------------------------------------- mid-level API
    # Object-centric primitives between the task level (pick/place) and the ASPIRE level (point clouds, poses):
    # perception, grasp proposals and motion execution happen here; the caller chooses the grasp, where and in
    # which orientation the object goes, how far to lower it, when to release, the order and the recovery.
    SAFE_HEIGHT = .20
    # The Panda finger meshes reach 9.2 mm past the grip point (robosuite panda_gripper.xml). With the grip point lower
    # than this above the table the arm pushes the fingertips onto it, and fingers dragging on the table closed too
    # slowly to catch a 1.2 cm pen before the lift (grip point 9.7 mm up, 3.8 cm apart after closing; sdesk 17000/17004).
    LOWEST_GRIP = .011

    def _table(self):
        if getattr(self, "_table_z", None) is None:
            self.observe()
        return self._table_z

    def grasp_candidates(self, object_name, max_candidates=8):
        """Top-down grasp proposals for a visible object: Contact-GraspNet (if the service is running) and
        footprint geometry. Each candidate: {"id", "object", "source", "x", "y", "height", "yaw",
        "squeeze_width", "fits_open_fingers", "score", ...}."""
        self.go_home()
        scene = self.observe()
        obj = scene["objects"].get(object_name)
        if obj is None:
            raise RobotError("object_not_visible", object=object_name, visible=sorted(scene["objects"]))
        table, h = scene["table_z"], obj["height"]
        masks = getattr(self, "_masks", {})
        points = masks["_xyz"][masks[obj["color"]]] if obj.get("color") in masks else None

        def squeeze(x, y, yaw):  # object extent along the finger-closing direction, from its points
            if points is None:
                return None
            d = (points[:, :2] - [x, y]) @ np.array([np.cos(yaw), np.sin(yaw)])
            return round(float(np.quantile(d, .98) - np.quantile(d, .02)), 4)

        out, note = [], None
        try:
            proposals, note = self._graspnet_proposals(obj, table)
        except Exception as exc:  # the service is optional; its failure is reported, not hidden
            proposals, note = [], f"contact-graspnet failed: {type(exc).__name__}: {str(exc)[:160]}"
        out += proposals[:4]
        low = self.LOWEST_GRIP
        heights = sorted({round(max(low, .5 * h), 3), round(max(low, .5 * h, h - .012), 3)})
        if obj["footprint_long"] - obj["footprint_short"] > .01:
            axes = [("short", obj["short_axis_yaw"]), ("long", obj["long_axis_yaw"])]
        else:  # round footprint (paper ball, mug body): any finger direction works; offer four, at one height
            axes = [("short", _wrap_half_pi(obj["short_axis_yaw"] + k * np.pi / 4)) for k in range(4)]
            heights = heights[:1]
        for axis, yaw in axes:
            for height in heights:
                out.append({"source": f"geometry (fingers across the {axis} side)", "x": obj["center"][0],
                            "y": obj["center"][1], "height": height, "yaw": yaw,
                            "score": .5 if axis == "short" else .3})  # numeric, so candidates can be sorted
        # Room for the hand: depth points of anything other than this object where the hand body or the open
        # fingers would go. The Panda hand (robosuite hand.stl) spans +-0.104 m along the finger direction and
        # +-0.032 m across, and its body starts 3.1 cm above the grip point; the open fingers are 2.1 cm wide.
        # Points higher than 0.25 m above the table (the parked arm, the lamp) are ignored.
        xyz = masks.get("_xyz")
        other = None
        if xyz is not None:
            keep = np.isfinite(xyz[..., 2])
            if obj.get("color") in masks:
                keep &= ~masks[obj["color"]]
            other = xyz[keep]
            other = other[(other[:, 2] - table > .006) & (other[:, 2] - table < .25)]
            # Also drop points on or right next to the object itself: thin parts of it (a mug's 3 mm rim) can fall
            # out of its colour mask and would otherwise count as an obstacle under the hand.
            near = np.hypot(other[:, 0] - obj["center"][0], other[:, 1] - obj["center"][1])
            other = other[near > obj["footprint_long"] / 2 + .012]

        def clearance(x, y, yaw, height):
            if other is None:
                return {}
            d = other[:, :2] - [x, y]
            along = d @ np.array([np.cos(yaw), np.sin(yaw)])
            across = d @ np.array([-np.sin(yaw), np.cos(yaw)])
            z = other[:, 2] - table
            body = (np.abs(along) <= .105) & (np.abs(across) <= .033) & (z > height + .03)
            fingers = (np.abs(along) >= .03) & (np.abs(along) <= .055) & (np.abs(across) <= .012) & (z > height - .005)
            return {"hand_clear": bool(body.sum() < 3), "fingers_clear": bool(fingers.sum() < 3),
                    "highest_obstacle_under_hand_m": round(float(z[body].max()), 3) if body.sum() >= 3 else None}

        if obj["footprint_short"] > .075 and xyz is not None:  # too wide for the open fingers either way (a notebook)
            out = self._edge_pinches(obj, table, masks, size=getattr(self, "_own_size", {}).get(object_name)) + out
        for i, c in enumerate(out[:max_candidates]):
            if c.get("pinch"):  # its own fields (the top-down squeeze and clearance do not apply)
                c.update({"id": i, "object": object_name, "object_height": h,
                          "object_center": [round(float(v), 4) for v in obj["center"][:2]],
                          "object_long_axis_yaw": obj["long_axis_yaw"],
                          "object_footprint": [obj["footprint_long"], obj["footprint_short"]]})
                continue
            width = squeeze(c["x"], c["y"], c["yaw"])
            if width is None:
                width = obj["footprint_short"] if "short" in c["source"] else obj["footprint_long"] \
                    if "long" in c["source"] else None
            c.update({"id": i, "object": object_name, "x": round(float(c["x"]), 4), "y": round(float(c["y"]), 4),
                      "height": round(float(c["height"]), 4), "yaw": round(float(c["yaw"]), 3),
                      "squeeze_width": width, "fits_open_fingers": None if width is None else bool(width < .075),
                      "object_height": h, "object_center": [round(float(v), 4) for v in obj["center"][:2]],
                      "object_long_axis_yaw": obj["long_axis_yaw"],
                      "object_footprint": [obj["footprint_long"], obj["footprint_short"]],
                      **clearance(c["x"], c["y"], c["yaw"], c["height"])})
        self.last_grasp_note = note
        for c in out[:max_candidates]:  # what the learned detector returned (or why not), for the program and the log
            c["graspnet_note"] = note
        return out[:max_candidates]

    def _edge_pinches(self, obj, table, masks, depth_in=.02, size=None):
        """Side grasps for a flat object wider than the open fingers (an A5 notebook): the gripper comes in level
        with the object, fingers pointing at one of its edges and closing vertically, one finger on the top face and
        one under the bottom face, depth_in inside the edge. That needs free space under the edge: the object must
        stick out beyond whatever it lies on (the desk edge, the edge of a pile of books). Per side of the footprint,
        from depth only: "drop_below_edge_m", how far the surface just beyond the edge lies below the object's bottom
        (about 0.8 m past the desk edge: the floor); "overhang_m", how far the object sticks out past the edge of its
        support, where that support is seen beside the object (None if it is hidden); "hand_clear", nothing within
        the hand's reach behind the grip point (the hand is 0.21 m tall when its fingers close vertically). The two
        measurements are left out when the camera cannot see them (past the desk edge it cannot see the floor below);
        "edge_free_below" is always given: whether the edge sticks out over a drop."""
        xyz, own = masks["_xyz"], masks[obj["color"]]
        valid = np.isfinite(xyz[..., 2])
        thick = size["height"] if size else obj["height"]  # its own thickness as first seen (_note_sizes)
        bottom = obj["top_z"] - thick
        c = np.asarray(obj["center"][:2], float)
        yaw = obj["long_axis_yaw"]
        u, v = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
        L, S = obj["footprint_long"], obj["footprint_short"]
        d = xyz[..., :2] - c
        out = []
        for n, half_n, t, half_t in ((u, L / 2, v, S / 2), (-u, L / 2, v, S / 2), (v, S / 2, u, L / 2), (-v, S / 2, u, L / 2)):
            dn, dt = d @ n, d @ t
            beyond = valid & ~own & (dn > half_n + .01) & (dn < half_n + .05) & (np.abs(dt) < half_t - .01)
            # beyond the desk edge the depth camera sees the floor (or nothing): count missing depth as a long drop
            below = float(bottom - np.median(xyz[..., 2][beyond])) if beyond.sum() >= 10 else None
            support = valid & ~own & (np.abs(xyz[..., 2] - bottom) < .006) & (np.abs(dt) > half_t + .01) & \
                (np.abs(dt) < half_t + .12) & (dn > 0) & (dn < half_n + .05)
            overhang = float(half_n - np.quantile(dn[support], .98)) if support.sum() >= 10 else None
            grip = c + n * (half_n - depth_in)
            z = max((obj["top_z"] + bottom) / 2, table + .005)
            # The hand body: 3-9 cm behind the grip point along the approach, +-0.104 m vertically, +-0.033 m sideways.
            body = valid & ~own & (dn > half_n - depth_in + .03) & (dn < half_n - depth_in + .10) & (np.abs(d @ t - (grip - c) @ t) < .033) & \
                (np.abs(xyz[..., 2] - z) < .104)
            # The camera above the desk cannot see the floor right next to the desk edge (the desk hides it): with no
            # depth beyond the edge, an object seen to stick out past its support counts as having free space below.
            free = (below is not None and below > .04) or (below is None and (overhang or 0) > 0)
            score = .6 if free and (overhang is None or overhang > depth_in + .01) and not body.sum() >= 3 else .1
            measured = {k: round(v, 3) for k, v in (("drop_below_edge_m", below), ("overhang_m", overhang)) if v is not None}
            out.append({"source": "edge pinch (fingers close on the top and bottom faces, from the side)", "pinch": True,
                        "x": round(float(grip[0]), 4), "y": round(float(grip[1]), 4), "height": round(float(z - table), 4),
                        "yaw": round(float(np.arctan2(t[1], t[0])), 3),
                        "approach": [round(float(-n[0]), 3), round(float(-n[1]), 3), 0.0], "finger_axis": [0.0, 0.0, 1.0],
                        "tilt_deg": 90.0, "squeeze_width": round(float(thick), 4), "fits_open_fingers": bool(thick < .075),
                        "edge_length_m": round(float(2 * half_t), 4), "grip_depth_inside_edge_m": depth_in,
                        "edge_free_below": bool(free), **measured, "hand_clear": bool(body.sum() < 3), "score": score})
        out.sort(key=lambda c: -c["score"])
        for c in out[:1] if out and out[0]["score"] > .5 else []:
            c.update(self._reachable_pinch(c, table))
        return out

    def _reachable_pinch(self, c, table, back=.05, lift=.08):
        """For a pinch past a desk edge the approach square to the edge is often out of reach. Turn it about the vertical
        (towards pointing away from the robot base first) and try both finger directions (+z / -z: the same grasp with
        the wrist turned half a turn) until the grasp pose, the point `back` m back along the approach and the point
        `lift` m above the grasp all have an inverse-kinematics solution. Nothing moves."""
        grip = np.array([c["x"], c["y"], table + c["height"]])
        a0, up = np.array(c["approach"], float), np.array([0.0, 0.0, 1.0])
        base = self._frame("body", "robot0_base")[:2, 3]
        away = np.append(grip[:2] - base, 0.0)
        side = 1.0 if np.cross(a0, away)[2] > 0 else -1.0  # the turn that brings the approach towards `away`
        for turn in (0, 20, 40, 60, -20, -40):
            t = np.radians(side * turn)
            a = np.array([a0[0] * np.cos(t) - a0[1] * np.sin(t), a0[0] * np.sin(t) + a0[1] * np.cos(t), 0.0])
            for f in (up, -up):
                rot = self._rotation_from(a, f)
                if all(self._ik_multi(p, rot)[0] is not None for p in (grip, grip - back * a, grip + lift * up)):
                    return {"approach": [round(float(v), 3) for v in a], "finger_axis": [0.0, 0.0, float(f[2])],
                            "approach_turned_deg": int(side * turn), "reachable_checked": True}
        return {"reachable_checked": False}

    def _dense_cloud(self, obj, table, size=512):
        """The object's points and the scene's points from a 512 px top-down depth render (the 256 px perception
        image gives an object only 80-300 points, too sparse for Contact-GraspNet: it returned 0-3 grasps)."""
        from robosuite.utils.camera_utils import get_camera_intrinsic_matrix
        from .perception import back_project
        if self.perception == "sam":  # the latest view is 512 px already, with the thing's own mask
            xyz = self._masks["_xyz"]
            valid = np.isfinite(xyz[..., 2])
            mask = self._masks.get(obj["color"], np.zeros(valid.shape, bool))
            near = np.hypot(xyz[..., 0] - obj["center"][0], xyz[..., 1] - obj["center"][1]) < obj["footprint_long"] / 2 + .01
            scene = xyz[valid]
            return xyz[mask & near], scene[::max(1, len(scene) // 20000)]
        rgb, depth = self.render("topdown", size, size, depth=True)
        K = get_camera_intrinsic_matrix(self.env.sim, "topdown", size, size)
        xyz = back_project(self._metric_depth(depth), K, self.world_from_camera)
        valid = np.isfinite(xyz[..., 2])
        labels, names = _palette_label(rgb, *self._hues())
        mask = object_pixels(labels == names.index(obj["color"]), valid, xyz[..., 2] - table) if obj.get("color") in names \
            else np.zeros(valid.shape, bool)
        near = np.hypot(xyz[..., 0] - obj["center"][0], xyz[..., 1] - obj["center"][1]) < obj["footprint_long"] / 2 + .01
        scene = xyz[valid]
        return xyz[mask & near], scene[::max(1, len(scene) // 20000)]

    def _graspnet_raw(self, points, view=None, max_candidates=40, scene_pts=None):
        """Contact-GraspNet grasps on an object's points as world-frame Panda grip-site poses [(4x4, score)].
        view "camera" (default) gives the network the clouds in the top-down camera's frame (z = viewing direction),
        the frame it was trained in. view "world" is the earlier call, kept for comparison: with world points the
        network's implicit camera sits at the world origin below the table, so its grasps come from below."""
        view = view or os.environ.get("SMA_GRASPNET_VIEW", "camera")
        url = os.environ.get("CAP_HARNESS_GRASPNET_URL")
        if not url or points is None or len(points) < 30:
            return None, "contact-graspnet not available (no service or no object points)"
        from cap_harness.contracts import PointCloud
        from cap_harness.providers.graspnet.client import ContactGraspNetProvider
        from ..aspire_level.grasp_calibration import GripSiteGraspProvider
        if scene_pts is None:
            xyz = self._masks["_xyz"]
            scene_pts = xyz[np.isfinite(xyz[..., 2])]
            scene_pts = scene_pts[::max(1, len(scene_pts) // 20000)]
        if view == "camera":
            to_world = self.world_from_camera
            R, t = to_world[:3, :3], to_world[:3, 3]
            obj_cloud, scene_cloud, frame = (points - t) @ R, (scene_pts - t) @ R, "topdown_camera"
        else:
            to_world, obj_cloud, scene_cloud, frame = np.eye(4), points, scene_pts, "world"
        # Three forward passes: more sampled grasps clear the network's score threshold (one pass returned none for
        # about half of the realdesk objects).
        provider = GripSiteGraspProvider(ContactGraspNetProvider(base_url=url, max_retries=0,
                                                                 forward_passes=int(os.environ.get("SMA_GRASPNET_PASSES", "3"))))
        result = provider.generate_grasps(PointCloud(obj_cloud, frame), scene_point_cloud=PointCloud(scene_cloud, frame),
                                          max_candidates=max_candidates)
        if not result.ok:
            return None, f"contact-graspnet: {str(result.error)[:160]}"
        return [(to_world @ g.pose.as_matrix(), float(g.score)) for g in result.grasps], f"contact-graspnet ({view} frame)"

    def _graspnet_proposals(self, obj, table, limit=4, max_tilt_deg=65, view=None):
        """Contact-GraspNet grasps on a dense cloud of the object. On objects lying on a table the network mostly
        proposes approaches 35-65 degrees from vertical (realdesk probe, 2026-11-05), so these are kept, the most
        vertical first. Each becomes a candidate whose x, y, height and yaw are its grip point and finger direction
        (what grasp() executes, descending vertically), plus the network's own 6-DoF approach for a program that
        wants to reproduce the tilted grasp with move_gripper / move_by."""
        points, scene_pts = self._dense_cloud(obj, table)
        grasps, note = self._graspnet_raw(points, view, scene_pts=scene_pts)
        self.last_graspnet_tilts = []
        if grasps is None:
            return [], note
        down = self.down.as_matrix()[:, 2]
        rows = []
        for m, score in grasps:
            tilt = float(np.degrees(np.arccos(np.clip(m[:3, 2] @ down, -1, 1))))
            self.last_graspnet_tilts.append(round(tilt, 1))
            if tilt <= max_tilt_deg:
                rows.append((tilt, -score, m))
        rows.sort(key=lambda r: (round(r[0] / 15), r[1]))  # nearly vertical first (15-degree bands), then by score
        kept = []
        for tilt, neg_score, m in rows[:limit]:
            p = m[:3, 3]
            kept.append({"source": "contact-graspnet", "x": float(p[0]), "y": float(p[1]),
                         "height": max(.004, float(p[2] - table)), "yaw": float(np.arctan2(m[1, 0], m[0, 0])),
                         "score": round(-neg_score, 3), "tilt_deg": round(tilt, 1),
                         "approach": [round(float(v), 3) for v in m[:3, 2]],
                         "finger_axis": [round(float(v), 3) for v in m[:3, 0]]})
        return kept, f"{note}: {len(grasps)} grasps on {len(points)} points, {len(kept)} kept (tilt <= {max_tilt_deg} deg)"

    def grasp(self, candidate):
        """Execute one candidate (its x, y, height, yaw may be adjusted): open, move above at a safe height,
        descend vertically, close, lift. Returns holding_something and the finger gap."""
        c = dict(candidate)
        try:
            x, y, height, yaw = (float(c[k]) for k in ("x", "y", "height", "yaw"))
        except (KeyError, TypeError, ValueError):
            raise RobotError("invalid_candidate", hint="pass one dict from grasp_candidates (x, y, height, yaw may be changed)")
        if not .004 <= height <= .15:
            raise RobotError("grasp_height_out_of_range", height=height, allowed=[.004, .15],
                             hint="height is the grip point's height ABOVE the table surface, not an absolute z")
        if self.gripper_state()["holding_something"]:
            raise RobotError("already_holding_something")
        raised = height < self.LOWEST_GRIP  # the fingertips stay off the table
        height = max(height, self.LOWEST_GRIP)
        table = self._table()
        self.open_gripper()
        here = self._ee()
        if here[2] < table + self.SAFE_HEIGHT - .005:
            self.move_to(here[0], here[1], table + self.SAFE_HEIGHT)
        self.move_to(x, y, table + self.SAFE_HEIGHT, yaw)
        stopped = None
        try:  # to within 2 mm: with the usual 6 mm the arm could stop 5 mm high, above most of a thin object
            self.move_to(x, y, table + height, yaw, tolerance=.002)
        except RobotError as exc:  # touching down slightly early is fine; holding is checked after closing
            if exc.reason != "motion_blocked" or exc.details["remaining_error_m"] > .012:
                raise
            stopped = exc.details["remaining_error_m"]
        closed = self.close_gripper()
        self._held_grasp_height = height
        self._held = {"object": c.get("object"), "grasp_yaw": yaw, "object_long_axis_yaw": c.get("object_long_axis_yaw"),
                      "footprint": c.get("object_footprint")}
        self.move_to(x, y, table + .18, yaw)
        after = self.gripper_state()
        out = {"holding_something": after["holding_something"], "finger_gap": after["width"],
               "finger_gap_when_closed": closed["width"], "expected_squeeze_width": c.get("squeeze_width"),
               "stopped_above_target_m": stopped}
        if raised:
            out["grip_height_raised_to"] = height
        if after["holding_something"] and c.get("object"):
            # The gripper only moved vertically since closing, so the object's pose before the grasp, relative to
            # the gripper now, is the fallback estimate of how it sits in the hand.
            fp = c.get("object_footprint") or [0.0, 0.0]
            estimate = None if c.get("object_center") is None or c.get("object_long_axis_yaw") is None else \
                {"center": c["object_center"], "yaw": c["object_long_axis_yaw"], "elongated": fp[0] - fp[1] > .01}
            self._held["in_hand"] = self._measure_in_hand(c["object"], estimate)
            out["object_in_hand"] = self._in_hand_summary(self._held["in_hand"])
        return out

    def _measure_in_hand(self, object_name, estimate=None):
        """How the held object sits in the hand: its centre offset in the gripper frame (along the finger line,
        across it) and its long-axis yaw minus the gripper yaw. Taken from the object's pose seen before the grasp
        (estimate); the camera view of the lifted object is used only when there is no estimate, because the hand
        hides most of a held pen (in the pose gate the camera was off by 2-6 cm and up to 27 degrees, the
        pre-grasp estimate by at most 3 mm and 0.1 degrees). The camera reading is kept for diagnosis."""
        g, p = self._yaw(), self._ee()[:2]
        seen = None
        if self.perception == "color":
            self.observe()
            xyz, color = self._masks["_xyz"], object_name.split("_")[0]
            labels, names = _palette_label(self.last_rgb, *self._hues())
            if color in names:
                lifted = (labels == names.index(color)) & np.isfinite(xyz[..., 2]) & (xyz[..., 2] > self._table_z + .04)
                if lifted.sum() >= 12:
                    center, long_side, short_side, yaw = _footprint(xyz[lifted][:, :2])
                    seen = {"center": center, "yaw": yaw, "elongated": long_side - short_side > .01,
                            "visible_pixels": int(lifted.sum())}
        source, pose = ("grasp estimate", estimate) if estimate else ("camera", seen)
        if pose is None:
            return None
        offset = _rot2(-g) @ (np.asarray(pose["center"][:2], float) - p)
        return {"source": source, "offset": [round(float(v), 4) for v in offset], "elongated": bool(pose["elongated"]),
                "yaw_offset": round(_wrap_half_pi(pose["yaw"] - g), 3) if pose["elongated"] else 0.0,
                "estimate": estimate, "camera": seen}

    @staticmethod
    def _in_hand_summary(rel):
        if rel is None:
            return None
        return {"source": rel["source"], "centre_offset_from_grip_point_m": rel["offset"],
                "object_long_axis_minus_gripper_yaw_deg": round(float(np.degrees(rel["yaw_offset"])), 1)
                if rel["elongated"] else None}

    def free_spots(self, zone_name, object_name=None, max_spots=5):
        """Placement spots inside a zone for an object's footprint (object_name, or the held object), avoiding
        other objects. Returns {"zone", "center", "half_size", "long_axis_yaw", "occupants", "spots": [{"x", "y",
        "object_long_axis_yaw", "clearance"}]}; spots is empty if nothing fits. object_long_axis_yaw is the direction
        the object's long axis must have at that spot; turning the held object there is left to the caller."""
        self.go_home()
        scene = self.observe()
        zone = self._zone_from_last_view(zone_name)
        held = getattr(self, "_held", None) if scene["gripper"]["holding_something"] else None
        if object_name is not None and object_name in scene["objects"]:
            o = scene["objects"][object_name]
            long_side, short_side = o["footprint_long"], o["footprint_short"]
        elif held and held.get("footprint"):
            long_side, short_side = held["footprint"]
            object_name = held["object"]
        else:
            raise RobotError("footprint_unknown", hint="pass a visible object_name, or call while holding an object")
        colour = object_name.split("_")[0]  # the held object may be seen (and misnamed) hanging at the park pose
        others = [o for n, o in scene["objects"].items() if n.split("_")[0] != colour]
        cx, cy = zone["center"][:2]
        hx, hy = zone["half_size"]
        zyaw = zone.get("long_axis_yaw", 0.0)
        ax, ay = np.array([np.cos(zyaw), np.sin(zyaw)]), np.array([-np.sin(zyaw), np.cos(zyaw)])
        occupants = [o["name"] for o in others if inside_zone(o["center"], zone)]
        spots = []
        for turn, ex, ey in ((0.0, long_side / 2, short_side / 2), (np.pi / 2, short_side / 2, long_side / 2)):
            # Camera footprints are accurate to a few millimetres; allow a near-exact fit (pen in its case).
            ux, uy = hx - ex - .003, hy - ey - .003
            if ux < -.004 or uy < -.004:
                continue
            ux, uy = max(ux, 0.0), max(uy, 0.0)
            for a in np.linspace(-ux, ux, 5):
                for b in np.linspace(-uy, uy, 5):
                    x, y = np.array([cx, cy]) + a * ax + b * ay
                    gaps = [np.hypot(x - o["center"][0], y - o["center"][1]) - o["footprint_long"] / 2
                            - np.hypot(ex, ey) for o in others]
                    clearance = min(gaps) if gaps else .1
                    if clearance < .005:
                        continue
                    yaw = (zyaw + turn + np.pi / 2) % np.pi - np.pi / 2
                    spots.append({"x": round(float(x), 4), "y": round(float(y), 4),
                                  "object_long_axis_yaw": round(float(yaw), 3),
                                  "clearance": round(float(min(clearance, .1)), 4),
                                  "_rank": (-min(clearance, .04), float(np.hypot(a, b)))})
        spots.sort(key=lambda s: s["_rank"])
        unique = []
        for sp in spots:
            sp.pop("_rank")
            if sp not in unique:
                unique.append(sp)
        return {"zone": zone_name, "center": zone["center"], "half_size": zone["half_size"],
                "long_axis_yaw": zyaw, "occupants": occupants, "spots": unique[:max_spots]}

    def _tilted(self):
        """Whether the fingers point more than 45 degrees away from straight down (a side grasp, a pen turned upright
        in the hand). carry_to, rotate_by and lower_until_contact then keep the hand's orientation: turning the fingers
        down, as they do otherwise, would turn whatever is held by as much (in DeepSeek runs a notebook taken by an
        edge ended up standing on that edge)."""
        return Rotation.from_quat(self.obs["robot0_eef_quat_site"]).as_matrix()[2, 2] > -np.cos(np.radians(45))

    def carry_to(self, x, y, height=.18, yaw=None):
        """Rise vertically to table_z + height (if lower), then move horizontally to (x, y) at that height,
        fingers along yaw (None keeps the current yaw). Whatever is held comes along. A tilted hand (_tilted) keeps
        its orientation; yaw is then refused (turn it with rotate_by)."""
        height = float(height)
        if not .05 <= height <= .30:
            raise RobotError("carry_height_out_of_range", height=height, allowed=[.05, .30],
                             hint="height is measured above the table surface")
        z = self._table() + height
        here = self._ee()
        if self._tilted():
            if yaw is not None:
                raise RobotError("yaw_needs_fingers_down", hint="the hand is tilted (e.g. a side grasp) and keeps its "
                                 "orientation: carry with yaw=None, turn it with rotate_by")
            rotation = Rotation.from_quat(self.obs["robot0_eef_quat_site"])
            if here[2] < z - .005:
                self._line_move([here[0], here[1], z], rotation)
            planner = "straight line"
            if os.environ.get("SMA_MOTION") == "curobo":
                note = self._curobo_move(float(x), float(y), z, rotation=rotation)
                planner = "curobo" if note is None else f"straight line (cuRobo: {note})"
            if planner != "curobo":
                self._line_move([float(x), float(y), z], rotation)
            return {"position": self._ee().round(4).tolist(), "holding_something": self.gripper_state()["holding_something"],
                    "planner": planner, "orientation_kept": True}
        if here[2] < z - .005:
            self.move_to(here[0], here[1], z)
        planner = "straight line"
        if os.environ.get("SMA_MOTION") == "curobo":  # collision-aware transit; straight line if no plan is found
            note = self._curobo_move(float(x), float(y), z, yaw)
            planner = "curobo" if note is None else f"straight line (cuRobo: {note})"
        if planner != "curobo":
            self.move_to(float(x), float(y), z, yaw)
        return {"position": self._ee().round(4).tolist(), "holding_something": self.gripper_state()["holding_something"],
                "planner": planner}

    def lower_until_contact(self):
        """Lower straight down until the held object's bottom reaches table level, or until something stops it
        earlier (a wall, another object). Does not open the fingers. A tilted hand (_tilted) keeps its orientation
        and goes down until something stops it (how far the held object reaches below the fingers is not known)."""
        table = self._table()
        holding = self.gripper_state()["holding_something"]
        here = self._ee()
        if self._tilted():
            # In 1 cm steps, stopping at the first one something blocks or that turns the hand. Sent on to the table in
            # one move, the arm kept pushing after the hand (hanging 10 cm below the grip point) met the pile, the
            # wrist twisted by 40-60 degrees and the notebook slid off (side-carry gate, 2026-11-06).
            rotation = Rotation.from_quat(self.obs["robot0_eef_quat_site"])
            floor, stopped = table + self.LOWEST_GRIP, False
            while not stopped and self._ee()[2] > floor + .002:
                z = max(float(self._ee()[2]) - .01, floor)
                stopped = self._line_move([here[0], here[1], z], rotation, tolerance=.003, stop_on_contact=True)
                turned = (Rotation.from_quat(self.obs["robot0_eef_quat_site"]) * rotation.inv()).magnitude()
                stopped = stopped or turned > np.radians(5)
            return {"stopped_by_contact": stopped, "grip_point_height": round(float(self._ee()[2]) - table, 4),
                    "held_object_bottom_above_table": None, "holding_something": self.gripper_state()["holding_something"],
                    "orientation_kept": True}
        grip = getattr(self, "_held_grasp_height", .01) if holding else .01
        stopped = None
        try:
            self.move_to(here[0], here[1], table + grip + .004, tolerance=.004)
        except RobotError as exc:
            if exc.reason != "motion_blocked":
                raise
            stopped = exc.details["remaining_error_m"]
        z = float(self._ee()[2]) - table
        return {"stopped_by_contact": bool(stopped is not None and stopped > .004), "grip_point_height": round(z, 4),
                "held_object_bottom_above_table": round(z - grip, 4) if holding else None,
                "holding_something": self.gripper_state()["holding_something"]}

    def rotate_by(self, delta):
        """Turn the gripper, and whatever it holds, by delta radians about the vertical axis without moving it
        (positive = counter-clockwise seen from above). The finger line is symmetric, so the shorter of delta and
        delta -/+ pi is executed; for a symmetric object (pen, book) both give the same orientation. A tilted hand
        (_tilted) keeps its tilt and turns by delta itself (within -/+ pi); the direction it points is reported."""
        delta = float(delta)
        if not np.isfinite(delta):
            raise RobotError("non_finite_rotation", delta=delta)
        if self._tilted():
            def heading():  # of the approach, which is not vertical here
                a = Rotation.from_quat(self.obs["robot0_eef_quat_site"]).as_matrix()[:, 2]
                return float(np.arctan2(a[1], a[0]))
            before, here = heading(), self._ee()
            delta = float((delta + np.pi) % (2 * np.pi) - np.pi)
            goal = Rotation.from_euler("z", delta) * Rotation.from_quat(self.obs["robot0_eef_quat_site"])
            self._move_pose(here, goal, tolerance=.01, residual_tol=.008)
            turned = (heading() - before + np.pi) % (2 * np.pi) - np.pi
            return {"approach_heading_before": round(before, 3), "approach_heading": round(heading(), 3),
                    "turned_by": round(float(turned), 3), "holding_something": self.gripper_state()["holding_something"],
                    "orientation_kept": True}
        before, here = self._yaw(), self._ee()
        self.move_to(here[0], here[1], here[2], before + delta, tolerance=.01)
        after = self._yaw()
        turned = (after - before + np.pi / 2) % np.pi - np.pi / 2
        return {"gripper_yaw_before": round(before, 3), "gripper_yaw": round(after, 3), "turned_by": round(float(turned), 3),
                "holding_something": self.gripper_state()["holding_something"]}

    def place_object(self, object_name, x, y, object_yaw=None):
        """Put the held object down with its centre at (x, y) and, for an object with a long axis, that axis at
        object_yaw (None keeps the current orientation). Uses how the object sits in the hand (measured at the
        grasp) to compute the gripper pose, carries at 0.18 m, lowers until contact, releases, retreats, then
        measures where the object ended up with the camera."""
        if not self.gripper_state()["holding_something"]:
            raise RobotError("not_holding_anything")
        held = getattr(self, "_held", None) or {}
        if held.get("object") not in (None, object_name):
            raise RobotError("holding_a_different_object", holding=held.get("object"), requested=object_name)
        rel = held.get("in_hand") or self._measure_in_hand(object_name)
        if rel is None:
            raise RobotError("in_hand_pose_unknown", hint="grasp the object with grasp() first")
        x, y = float(x), float(y)
        turn = object_yaw is not None and rel["elongated"]
        g = float(object_yaw) - rel["yaw_offset"] if turn else self._yaw()
        g = self.yaw0 + _wrap_half_pi(g - self.yaw0)  # the representative move_to will execute
        gx, gy = np.array([x, y]) - _rot2(g) @ np.asarray(rel["offset"])
        carried = self.carry_to(gx, gy, .18, g)
        lowered = self.lower_until_contact()
        self.open_gripper()
        self.retreat()
        self._held = None
        self.go_home()
        scene = self.observe()
        color = object_name.split("_")[0]
        o = scene["objects"].get(object_name) or next((o for o in scene["objects"].values() if o["color"] == color), None)
        out = {"target": {"x": round(x, 4), "y": round(y, 4), "object_yaw": None if object_yaw is None else
                          round(float(object_yaw), 3)},
               "gripper_target": {"x": round(float(gx), 4), "y": round(float(gy), 4), "yaw": round(g, 3)},
               "in_hand_from": rel["source"], "planner": carried["planner"], "stopped_by_contact": lowered["stopped_by_contact"],
               "holding_something": scene["gripper"]["holding_something"], "measured": None}
        if object_yaw is not None and not rel["elongated"]:
            out["object_yaw_ignored"] = "the object has no long axis"
        if o is not None:
            out["measured"] = {"x": o["center"][0], "y": o["center"][1], "object_long_axis_yaw": o["long_axis_yaw"],
                               "height_above_table": o["height"]}
            out["position_error_m"] = round(float(np.hypot(o["center"][0] - x, o["center"][1] - y)), 4)
            if turn:
                out["yaw_error_deg"] = round(abs(np.degrees(_wrap_half_pi(o["long_axis_yaw"] - float(object_yaw)))), 1)
        return out

    def _move_pose(self, position, rotation, tolerance=.006, blocked_ok=False, residual_tol=.004):
        """Move the grip point to position with an arbitrary gripper orientation (scipy Rotation), along the same
        straight-line IK path as move_to. blocked_ok: stopping early on contact is reported, not raised."""
        target = np.asarray(position, float)
        if np.any(target < WORKSPACE[:, 0]) or np.any(target > WORKSPACE[:, 1]):
            raise RobotError("target_outside_workspace", target=target.round(4).tolist(), workspace=WORKSPACE.tolist())
        path, failure = self._plan(target, rotation, residual_tol)
        if path is None:
            raise RobotError("unreachable_pose", target=target.round(4).tolist(), **failure)
        try:
            return self._follow(path, target, _closing_yaw(rotation), tolerance)
        except RobotError as exc:
            if blocked_ok and exc.reason == "motion_blocked":
                return {"ok": False, "stopped_m": exc.details["remaining_error_m"]}
            raise

    def _ik_multi(self, position, rotation, tol=.004):
        """IK for one pose from several starting postures (current, rest, and the rest posture with the wrist turned
        by +-90 degrees); returns (joints, residual) of the first solution within tol, else (None, best residual)."""
        q_now = self._q()
        seeds = [q_now, self.q_rest] + [self.q_rest + np.eye(7)[6] * d for d in (np.pi / 2, -np.pi / 2)] + \
                [q_now + np.eye(7)[6] * d for d in (np.pi / 2, -np.pi / 2)]
        best = np.inf
        for seed in seeds:
            q, residual = self._ik(np.asarray(position, float), rotation, q0=np.clip(seed, self.q_low, self.q_high),
                                   iterations=400)
            if residual <= tol:
                return q, residual
            best = min(best, residual)
        return None, float(best)

    def _descent(self, q_top, top_pos, low_pos, rotation, step=.015):
        """Straight vertical path from top_pos to low_pos at a fixed orientation. Each waypoint's IK starts from the
        joint interpolation between the top and bottom solutions, which keeps the whole path on one IK branch.
        Returns (joint path, info) or (None, info)."""
        q_low, residual = self._ik_multi(low_pos, rotation)
        if q_low is None:
            return None, {"bottom_ik_residual": round(residual, 4)}
        n = max(2, int(np.ceil(np.linalg.norm(low_pos - top_pos) / step)))
        path = []
        for i in range(1, n + 1):
            seed = q_top + (q_low - q_top) * i / n
            q, residual = self._ik(top_pos + (low_pos - top_pos) * i / n, rotation, q0=seed, iterations=300)
            if residual > .006:
                return None, {"failed_at_fraction": round(i / n, 2), "ik_residual": round(residual, 4)}
            path.append(q)
        return path, {"waypoints": n}

    def _joint_move(self, q_goal, position, rotation):
        """Interpolate in joint space to q_goal (steps of at most 0.06 rad), then settle at the pose."""
        q0 = self._q()
        n = max(1, int(np.ceil(np.max(np.abs(q_goal - q0)) / .06)))
        path = [q0 + (q_goal - q0) * i / n for i in range(1, n + 1)]
        return self._follow(path, np.asarray(position, float), _closing_yaw(rotation), .01)

    def place_upright(self, object_name, x, y):
        """Stand the held long object (a pen) on its end with its lower end at (x, y), e.g. in a pen holder. The
        object is carried flat to a point high above (x, y); there the gripper turns in place to point horizontally
        away from the robot base (its most natural reach) with the object hanging vertically below the fingers, the
        longer part (beyond the grip point) down; then it is lowered until it touches down, released, and the
        gripper retreats upward; the arm parks and the camera result is reported. Grasp the object across its width
        and 3-4 cm from one end, so that most of it hangs below the fingers and they stay above the holder walls."""
        if not self.gripper_state()["holding_something"]:
            raise RobotError("not_holding_anything")
        held = getattr(self, "_held", None) or {}
        if held.get("object") not in (None, object_name):
            raise RobotError("holding_a_different_object", holding=held.get("object"), requested=object_name)
        rel = held.get("in_hand") or self._measure_in_hand(object_name)
        length = (held.get("footprint") or [None])[0]
        if rel is None or not rel["elongated"] or not length:
            raise RobotError("needs_a_long_object_held_across_its_width")
        table, g = self._table(), self._yaw()
        axis = np.array([np.cos(g + rel["yaw_offset"]), np.sin(g + rel["yaw_offset"]), 0.0])
        centre = np.append(_rot2(g) @ np.asarray(rel["offset"]), 0.0)  # grip point -> object centre (world)
        if centre @ axis < 0:  # point the axis at the longer part (the side the centre lies on)
            axis = -axis
        hang = length / 2 + float(centre @ axis)  # grip point to the lower end once upright
        lower_end = centre + axis * (length / 2)  # grip point -> that end, now (world)
        # Target orientation: long part straight down, approach axis (gripper z) horizontal. Preferred heading: away
        # from the base; if the arm cannot reach that pose, headings up to 90 degrees to either side are tried.
        now = Rotation.from_quat(self.obs["robot0_eef_quat_site"])
        e1 = now.inv().apply(axis)
        e1 /= np.linalg.norm(e1)
        e2 = np.array([0.0, 0.0, 1.0])
        away = np.array([x, y]) - self._frame("body", "robot0_base")[:2, 3]
        heading0 = float(np.arctan2(away[1], away[0]))
        down = np.array([0.0, 0.0, -1.0])
        here = self._ee()
        high = max(here[2], table + .12 + hang + .08)  # the swinging object stays well above the 8 cm walls
        tried = []
        for turn in np.radians([0, 30, -30, 60, -60, 90, -90]):
            u = np.array([np.cos(heading0 + turn), np.sin(heading0 + turn), 0.0])
            upright = Rotation.from_matrix(np.column_stack([down, u, np.cross(down, u)]) @
                                           np.column_stack([e1, e2, np.cross(e1, e2)]).T)
            end_offset = upright.apply(now.inv().apply(lower_end))  # grip point -> lower end after the turn
            gx, gy = x - end_offset[0], y - end_offset[1]
            top_pos = np.array([gx, gy, high])
            low_pos = np.array([gx, gy, table + .004 + .008 - end_offset[2]])  # lower end 8 mm above the floor
            q_up, residual = self._ik_multi(top_pos, upright)
            entry = {"heading_offset_deg": round(float(np.degrees(turn))), "top_ik_residual": round(residual, 4)}
            tried.append(entry)
            if q_up is None:
                continue
            path_down, entry["descent"] = self._descent(q_up, top_pos, low_pos, upright)
            if path_down is not None:
                break
        else:
            raise RobotError("unreachable_pose", stage="turning the object upright and lowering it", tried=tried)
        self.move_to(here[0], here[1], high)
        self.move_to(gx, gy, high)  # carried flat, fingers still down
        self._joint_move(q_up, top_pos, upright)  # turn in joint space: no straight-line constraint up here
        yaw_up = _closing_yaw(upright)
        touched = False
        try:
            self._follow(path_down, low_pos, yaw_up, .004)
        except RobotError as exc:  # touching down early (the floor, another pen) is fine
            if exc.reason != "motion_blocked":
                raise
            touched = True
        self.open_gripper()
        self._follow(path_down[::-1] + [q_up], top_pos, yaw_up, .02)  # back up the same way, fingers open
        self._held = None
        self.go_home()
        scene = self.observe()
        return {"target_lower_end": [round(float(x), 4), round(float(y), 4)], "grip_point_to_lower_end_m": round(hang, 4),
                "heading_offset_deg": tried[-1]["heading_offset_deg"], "stopped_by_contact": touched,
                "holding_something": scene["gripper"]["holding_something"]}

    # ---------------------------------------------------------------- action-level API (SMA_API_LEVEL=act)
    # Free gripper orientation for the model: where the fingers point (approach) and along which line they close
    # (finger_axis). Nothing here decides how to manipulate a particular object.
    def _rotation_from(self, approach, finger_axis=None):
        z = np.asarray(approach, float)
        if z.shape != (3,) or not np.all(np.isfinite(z)) or np.linalg.norm(z) < 1e-6:
            raise RobotError("invalid_approach", hint="approach is a 3-vector, e.g. [0, 0, -1] for fingers pointing down")
        z = z / np.linalg.norm(z)
        current = Rotation.from_quat(self.obs["robot0_eef_quat_site"]).as_matrix()[:, 0]
        x = current if finger_axis is None else np.asarray(finger_axis, float)
        if x.shape != (3,) or not np.all(np.isfinite(x)):
            raise RobotError("invalid_finger_axis", hint="finger_axis is a 3-vector perpendicular to approach")
        x = x - (x @ z) * z
        if np.linalg.norm(x) < .3:
            if finger_axis is not None:
                raise RobotError("finger_axis_not_perpendicular_to_approach",
                                 hint="give a finger_axis (nearly) perpendicular to the approach direction")
            x = np.cross(z, [0.0, 0.0, 1.0]) if abs(z[2]) < .9 else current - (current @ z) * z
        x = x / np.linalg.norm(x)
        return Rotation.from_matrix(np.column_stack([x, np.cross(z, x), z]))

    def gripper_pose(self):
        """The grip point (between the fingertips) and the gripper's axes as unit vectors in the world frame:
        approach (the direction the fingers point), finger_axis (the line along which they close) and
        side = approach x finger_axis. Something held across its width lies along side."""
        R = Rotation.from_quat(self.obs["robot0_eef_quat_site"]).as_matrix()
        return {"position": self._ee().round(4).tolist(), "approach": R[:, 2].round(4).tolist(),
                "finger_axis": R[:, 0].round(4).tolist(), "side": np.cross(R[:, 2], R[:, 0]).round(4).tolist(),
                "yaw": round(self._yaw(), 3), "holding_something": self.gripper_state()["holding_something"]}

    def _line_move(self, target, rotation=None, tolerance=.006, stop_on_contact=False):
        """Straight-line move of the grip point at a fixed orientation (the current one by default). The IK of each
        waypoint starts from the joint interpolation between the start and the goal solutions, which keeps the path
        on one IK branch (tilted grippers otherwise drift off it)."""
        rotation = rotation or Rotation.from_quat(self.obs["robot0_eef_quat_site"])
        target = np.asarray(target, float)
        if np.any(target < WORKSPACE[:, 0]) or np.any(target > WORKSPACE[:, 1]):
            raise RobotError("target_outside_workspace", target=target.round(4).tolist(), workspace=WORKSPACE.tolist())
        path, info = self._descent(self._q(), self._ee(), target, rotation)
        if path is None:
            raise RobotError("unreachable_pose", target=target.round(4).tolist(), **info)
        try:
            self._follow(path, target, _closing_yaw(rotation), tolerance)
        except RobotError as exc:
            if stop_on_contact and exc.reason == "motion_blocked":
                return True
            raise
        return False

    def move_gripper(self, x, y, z, approach=(0.0, 0.0, -1.0), finger_axis=None, path="line"):
        """Move the grip point to (x, y, z) with the fingers pointing along approach and closing along finger_axis
        (None keeps the current finger line where possible). path: "line" (straight line, orientation turned on
        the way), "joint" (joint-space motion to an IK solution: no straight-line constraint, so the hand and a held
        object swing on the way; use it well above obstacles), "curobo" (collision-aware plan against the camera's
        depth points; a held object is not modelled). Returns gripper_pose()."""
        rotation = self._rotation_from(approach, finger_axis)
        target = np.array([x, y, z], float)
        if path == "line":
            now = Rotation.from_quat(self.obs["robot0_eef_quat_site"])
            if (rotation * now.inv()).magnitude() < np.radians(2):
                self._line_move(target, rotation)
            else:
                self._move_pose(target, rotation, residual_tol=.008)
        elif path == "joint":
            q, residual = self._ik_multi(target, rotation)
            if q is None:
                raise RobotError("unreachable_pose", target=target.round(4).tolist(), ik_residual=round(residual, 4),
                                 hint="try another approach direction, finger axis or position (solve_ik checks without moving)")
            self._joint_move(q, target, rotation)
        elif path == "curobo":
            if not os.environ.get("CAP_HARNESS_CUROBO_URL"):
                raise RobotError("curobo_not_available", hint="use path='line' or 'joint'")
            note = self._curobo_move(x, y, z, rotation=rotation)
            if note:
                raise RobotError("curobo_plan_failed", reason=note)
        else:
            raise RobotError("unknown_path", allowed=["line", "joint", "curobo"])
        return self.gripper_pose()

    def move_by(self, dx=0.0, dy=0.0, dz=0.0):
        """Move the grip point by (dx, dy, dz) in a straight line, keeping the gripper's orientation. Stopping early
        because something blocks the motion is reported (stopped_by_contact), not raised."""
        stopped = self._line_move(self._ee() + np.array([dx, dy, dz], float), tolerance=.004, stop_on_contact=True)
        return {"stopped_by_contact": stopped, **self.gripper_pose()}

    def solve_ik(self, x, y, z, approach=(0.0, 0.0, -1.0), finger_axis=None, collision_check=False):
        """Can the arm reach this grip pose? Nothing moves. With collision_check=True cuRobo also checks the arm
        against the camera's depth points (None if cuRobo is not running)."""
        rotation = self._rotation_from(approach, finger_axis)
        target = np.array([x, y, z], float)
        q, residual = self._ik_multi(target, rotation)
        out = {"reachable": q is not None, "ik_residual_m": round(residual, 4)}
        if collision_check:
            if not os.environ.get("CAP_HARNESS_CUROBO_URL"):
                out["collision_free"] = None
            else:
                goal = np.eye(4)
                goal[:3, :3], goal[:3, 3] = rotation.as_matrix(), target
                held = (getattr(self, "_held", None) or {}).get("object") if self.gripper_state()["holding_something"] else None
                self.observe()
                reply, error = self._curobo("/plan", goal, self._curobo_scene(held=held))  # /ik: service bug
                out["collision_free"] = reply is not None
        return out

    # ---------------------------------------------------------------- cuRobo (collision-aware IK and planning)
    # The ASPIRE provider service. Obstacles are the top-down depth cloud (the service carves out the robot's own
    # points); a grip-site target is converted to robosuite's right-hand body, as in the ASPIRE adapter.
    def _frame(self, kind, name):
        data = self.env.sim.data
        get_mat, get_pos = (data.get_body_xmat, data.get_body_xpos) if kind == "body" else \
            (data.get_site_xmat, data.get_site_xpos)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = get_mat(name), get_pos(name)
        return T

    def _grip_pose(self, x, y, z, yaw):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self._target_rotation(yaw).as_matrix(), [x, y, z]
        return T

    def _curobo_scene(self, exclude=(), held=None):
        """World points of the latest depth view in and around the work area (the table included), without the
        named objects' points (e.g. one about to be grasped) and without every lifted pixel of the held object's
        colour (the hand can split it in two), on a 1 cm voxel grid."""
        xyz = self._masks["_xyz"].reshape(-1, 3)
        keep = np.isfinite(xyz[:, 2])
        if held and self.perception == "sam":
            mask = self._masks.get(held)
            if mask is not None:
                keep &= ~(mask.reshape(-1) & (xyz[:, 2] > self._table_z + .03))
        elif held:
            labels, names = _palette_label(self.last_rgb, *self._hues())
            color = held.split("_")[0]
            if color in names:
                keep &= ~((labels.reshape(-1) == names.index(color)) & (xyz[:, 2] > self._table_z + .03))
        lo, hi = WORKSPACE[:, 0] - [.10, .10, .05], WORKSPACE[:, 1] + [.10, .10, 0.0]
        keep &= np.all((xyz >= lo) & (xyz <= hi), axis=1)
        for name in exclude:
            mask = self._masks.get(name.split("_")[0])
            if mask is None:
                continue
            pts = self._masks["_xyz"][mask]
            lower, upper = pts.min(0) - [.015, .015, .01], pts.max(0) + [.015, .015, .02]
            keep &= ~np.all((xyz >= lower) & (xyz <= upper), axis=1)
        pts = xyz[keep]
        _, first = np.unique(np.floor(pts / .01).astype(np.int64), axis=0, return_index=True)
        return pts[np.sort(first)]

    def _curobo(self, path, grip_target, scene_points, timeout=120):
        """POST one IK or planning request; returns (response dict, None) or (None, reason)."""
        import requests
        from cap_harness.providers.wire import encode_numpy
        url = os.environ.get("CAP_HARNESS_CUROBO_URL")
        if not url:
            return None, "cuRobo service not configured"
        base = self._frame("body", "robot0_base")
        grip_in_hand = np.linalg.inv(self._frame("body", "robot0_right_hand")) @ self._frame("site", "gripper0_right_grip_site")
        hand = np.linalg.inv(base) @ grip_target @ np.linalg.inv(grip_in_hand)
        qx, qy, qz, qw = Rotation.from_matrix(hand[:3, :3]).as_quat()
        points = (scene_points - base[:3, 3]) @ base[:3, :3]
        payload = {"base_frame": "robot0_base", "model": "panda", "joint_positions": {"primary": self._q().tolist()},
                   "joint_names": {"primary": [f"panda_joint{i}" for i in range(1, 8)]},
                   "base_transforms": {"primary": np.eye(4).tolist()}, "end_effector_links": {"primary": "panda_hand"},
                   "arm": "primary", "target": {"type": "pose", "position": hand[:3, 3].tolist(),
                                                "quaternion_wxyz": [qw, qx, qy, qz], "frame": "robot0_base"},
                   "scene_points_base64": encode_numpy(points.astype(np.float32)), "voxel_size_m": .01}
        try:
            reply = requests.post(url.rstrip("/") + path, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            return None, f"request failed: {type(exc).__name__}"
        if reply.status_code != 200:
            return None, f"HTTP {reply.status_code}: {reply.text[:200]}"
        return reply.json(), None

    def physical_feasibility(self, object_name, zone_name):
        """Physical level of the skill applicability check: collision-free arm poses (cuRobo plans against the observed
        depth cloud, gripper pointing down) above the object, above the zone's first free spot and at that spot with
        the object's bottom 2 cm above the floor. None if the cuRobo service is not running.
        For a tall receptacle the hand does not go in: a paper ball is dropped from above a wastebasket's rim and a pen
        is lowered into a holder by a tilted hand high above it (the 21 cm hand never fits inside the 20 cm basket or
        the 6 cm cup, so checking a pose down at the floor rejected every such skill). There the poses are above the
        object and above the receptacle's centre, at rim height plus 8 cm (bin) or 25 cm (holder)."""
        if not os.environ.get("CAP_HARNESS_CUROBO_URL"):
            return None
        self.go_home()
        scene = self.observe()
        o = scene["objects"].get(object_name)
        zone = scene["zones"].get(zone_name) or getattr(self, "_zones_seen", {}).get(zone_name)
        if o is None or zone is None:
            return {"ok": False, "failed": ["visibility"], "poses": {}}
        style = zone_name.split("_")[-1]
        if self.neutral and zone.get("hollow") and (zone.get("rim_height") or 0) > .06:  # no names: from geometry
            style = "holder" if max(zone["half_size"]) < .045 else "bin"
        if style in ("bin", "holder"):
            table, rim = scene["table_z"], zone.get("rim_height") or .1
            yaw = o["short_axis_yaw"]
            poses = {"above_object": (o["center"][0], o["center"][1], o["top_z"] + .06, yaw),
                     "above_zone": (zone["center"][0], zone["center"][1], table + rim + (.08 if style == "bin" else .25), yaw)}
            points = self._curobo_scene(exclude=[object_name])
            result = {}
            for name, (x, y, z, yaw) in poses.items():
                reply, error = self._curobo("/plan", self._grip_pose(x, y, z, self.yaw0 + _wrap_half_pi(yaw - self.yaw0)), points)
                result[name] = {"ok": reply is not None, **({"error": error} if error else {})}
            failed = [k for k, v in result.items() if not v["ok"]]
            return {"ok": not failed, "failed": failed, "poses": result, "scene_points": int(len(points))}
        try:
            spots = self.free_spots(zone_name, object_name)["spots"]
        except RobotError as exc:
            return {"ok": False, "failed": [exc.reason], "poses": {}}
        if not spots:
            return {"ok": False, "failed": ["no free spot"], "poses": {}}
        table, s = scene["table_z"], spots[0]
        floor = zone.get("top_height") or .004  # a pile of books: the object goes on its top
        elongated = o["footprint_long"] - o["footprint_short"] > .01
        grasp_yaw = o["short_axis_yaw"]  # fingers across the short side: the long axis is 90 degrees from them
        place_yaw = s["object_long_axis_yaw"] + np.pi / 2 if elongated else grasp_yaw
        poses = {"above_object": (o["center"][0], o["center"][1], o["top_z"] + .06, grasp_yaw),
                 "above_zone": (s["x"], s["y"], table + .18, place_yaw),
                 "into_zone": (s["x"], s["y"], table + floor + .02 + o["height"] / 2, place_yaw)}
        points = self._curobo_scene(exclude=[object_name])
        result = {}
        for name, (x, y, z, yaw) in poses.items():
            # A collision-free plan to the pose (the service's /ik endpoint fails to load its collision model).
            reply, error = self._curobo("/plan", self._grip_pose(x, y, z, self.yaw0 + _wrap_half_pi(yaw - self.yaw0)), points)
            result[name] = {"ok": reply is not None, **({"error": error} if error else {})}
        failed = [k for k, v in result.items() if not v["ok"]]
        return {"ok": not failed, "failed": failed, "poses": result, "scene_points": int(len(points))}

    def _curobo_move(self, x, y, z, yaw=None, rotation=None):
        """Move the grip point to (x, y, z) along a cuRobo collision-aware plan (held object excluded from the
        obstacles), fingers down at yaw or with the given orientation. Returns None when done, or the reason no
        plan was used (nothing moved then)."""
        target = np.array([x, y, z], float)
        if np.any(target < WORKSPACE[:, 0]) or np.any(target > WORKSPACE[:, 1]):
            return "target outside the workspace"
        if rotation is None:
            yaw = self._yaw() if yaw is None else float(yaw)
            yaw = self.yaw0 + _wrap_half_pi(yaw - self.yaw0)
            goal = self._grip_pose(x, y, z, yaw)
        else:
            goal = np.eye(4)
            goal[:3, :3], goal[:3, 3] = rotation.as_matrix(), target
            yaw = _closing_yaw(rotation)
        self.observe()
        held = (getattr(self, "_held", None) or {}).get("object") if self.gripper_state()["holding_something"] else None
        reply, error = self._curobo("/plan", goal, self._curobo_scene(held=held))
        if reply is None:
            self.last_curobo_error = error
            return error[:120]
        from cap_harness.providers.wire import decode_numpy
        plan = np.asarray(decode_numpy(reply["joint_positions_base64"]), float)
        path, last = [], self._q()
        for q in plan:  # keep waypoints at most ~0.06 rad apart (the straight-line planner's spacing)
            if np.max(np.abs(q - last)) >= .06:
                path.append(q)
                last = q
        path.append(plan[-1])
        self._follow(path, target, yaw)
        return None

    def retreat(self, dz=.12):
        """Move straight up by dz, keeping the gripper's orientation (fingers unchanged)."""
        here = self._ee()
        self._line_move([here[0], here[1], min(here[2] + float(dz), WORKSPACE[2, 1] - .01)])
        return {"position": self._ee().round(4).tolist(), "holding_something": self.gripper_state()["holding_something"]}

    def check(self, object_name, zone_name):
        """Camera check (arm parked first): is the object inside the zone and resting, with the gripper empty?"""
        return self.check_on_zone(object_name, zone_name)

    def wait(self, steps=20):
        hold = self._q()
        for _ in range(int(steps)):
            self._step(hold)

    def _park_for_check(self):
        """Park before a camera check. A program can leave the arm stuck (e.g. a tilted hand caught on a cup); the
        verifier then goes back through joint space, and if even that fails it looks from where the arm is."""
        try:
            self.go_home()
            return True
        except RobotError:
            pass
        try:
            q0 = self._q()
            n = max(1, int(np.ceil(np.max(np.abs(self.q_rest - q0)) / .06)))
            for i in range(1, n + 1):
                for _ in range(4):
                    self._step(q0 + (self.q_rest - q0) * i / n)
            self.wait(20)
            self.go_home()
            return True
        except RobotError:
            return False

    def check_on_zone(self, object_name, zone_name):
        """Perception-only subgoal check (robot moved home first so it does not occlude)."""
        parked = self._park_for_check()
        scene = self.observe()
        if not parked:
            scene["arm_not_parked"] = True
        return self._evaluate_goal(scene, object_name, zone_name, save_view=True)

    def check_goals(self, goals):
        """Monitoring between subgoals: park once, look once, and check every (object, zone) goal. Returns
        ({object: check}, scene)."""
        self._park_for_check()
        scene = self.observe()
        return {obj: self._evaluate_goal(scene, obj, zone, save_view=False) for obj, zone in goals}, scene

    def _evaluate_goal(self, scene, object_name, zone_name, save_view=True):
        obj, zone = scene["objects"].get(object_name), scene["zones"].get(zone_name)
        # A pen holder: by name, or (colour-coded desk, no names) a narrow hollow thing more than 6 cm tall.
        narrow_tall = zone is not None and zone.get("hollow") and (zone.get("rim_height") or 0) > .06 and \
            max(zone["half_size"]) < .045
        if zone is not None and (zone_name.endswith("holder") or narrow_tall) and self.perception in ("color", "sam"):
            return self._check_in_holder(object_name, zone, zone_name, scene, save_view)
        if obj is None or zone is None:
            return {"satisfied": False, "reason": "not_visible", "object_visible": obj is not None,
                    "zone_visible": zone is not None}
        result = object_on_zone(obj, zone, scene["table_z"], own=getattr(self, "_own_size", {}).get(object_name))
        result["gripper_holding"] = scene["gripper"]["holding_something"]
        result["satisfied"] = (result["inside_zone"] and result["resting"] and result.get("upright", True)
                               and not result["gripper_holding"])
        if self.perception in ("color", "sam") and save_view:  # a zoomed copy of the camera view, for a VLM judge
            self.check_views = getattr(self, "check_views", [])
            self.check_views.append({"object": object_name, "zone": zone_name, "satisfied": result["satisfied"],
                                     "image": self._zone_view(zone)})
        return result

    def _check_in_holder(self, object_name, zone, zone_name, scene, save_view=True):
        """Camera check for a pen holder: from above, a pen standing in the cup shows only its top and a little of
        its side. Satisfied when pixels of the pen's colour are seen inside the cup's footprint (1 cm margin) well
        above its floor, the highest of them is more than 2 cm above the cup's rim (a pen lying across the rim tops
        out just above it), and the pixels of that colour within 15 cm of the cup span less than 11 cm (a pen lying flat spans
        about 14 cm; one leaning 37 degrees, about 8 cm), and none of them lie outside the cup below its rim (a pen
        leaning on the outside of the cup)."""
        color, table = object_name.split("_")[0], scene["table_z"]
        xyz = self._masks["_xyz"]
        valid = np.isfinite(xyz[..., 2])
        if self.perception == "sam":  # the thing's own mask
            mask = self._masks.get(object_name, np.zeros(valid.shape, bool))
        else:
            labels, names = _palette_label(self.last_rgb, *self._hues())
            mask = (object_pixels(labels == names.index(color), valid, xyz[..., 2] - table) if color in names
                    else np.zeros(valid.shape, bool))
        near = (np.abs(xyz[..., 0] - zone["center"][0]) < .15) & (np.abs(xyz[..., 1] - zone["center"][1]) < .15)
        pts = xyz[mask & near]  # the pen's colour above the table within 15 cm of the holder
        inside = np.array([inside_zone(p[:2], zone, -.01) for p in pts], bool) if len(pts) else np.zeros(0, bool)
        high = pts[inside & (pts[:, 2] > table + .06)] if len(pts) else pts
        top = float(high[:, 2].max() - table) if len(high) else None
        rim = zone.get("rim_height") or .085
        # A pen leaning on the outside of the cup also shows high pixels over the cup, but its lower part lies outside
        # the cup below the rim; a pen standing inside shows outside the cup only parts above the rim.
        low_outside = int((~inside & (pts[:, 2] < table + rim)).sum()) if len(pts) else 0
        span = float(_footprint(pts[:, :2])[1]) if len(pts) >= 3 else None
        # Standing: the top more than 2 cm above the rim and at least as high as the object, at the length the camera
        # measured when it first saw it, tilted 40 degrees from vertical (the ground-truth limit for a pen); a pen
        # knocked over to ~50 degrees still rose above an 8 cm rim. Compact: its pixels near the cup span less than
        # 0.78 of that length (a pen lying flat spans its length; one leaning 37 degrees, about 0.6 of it).
        length = (getattr(self, "_own_size", {}).get(object_name) or {}).get("length")
        need = max(rim + .02, .004 + length * np.cos(np.radians(40))) if length else rim + .02
        result = {"inside_zone": bool(len(high) >= 3 and low_outside < 3),
                  "standing": bool(top is not None and top > need),
                  "compact": bool(span is not None and (length is None or span < .78 * length)),
                  "own_length_first_seen": length, "pen_pixels_low_outside": low_outside,
                  "pen_pixels_high_in_holder": int(len(high)), "top_height_above_table": None if top is None else round(top, 4),
                  "rim_height": rim, "horizontal_span_m": None if span is None else round(span, 4),
                  "zone_half_size": zone["half_size"], "gripper_holding": scene["gripper"]["holding_something"]}
        result["satisfied"] = (result["inside_zone"] and result["standing"] and result["compact"]
                               and not result["gripper_holding"])
        if self.neutral:  # no names on the colour-coded desk, not even in the check's field names
            result = {k.replace("pen_pixels", "object_pixels").replace("in_holder", "inside"): v for k, v in result.items()}
        if save_view:
            self.check_views = getattr(self, "check_views", [])
            self.check_views.append({"object": object_name, "zone": zone_name, "satisfied": result["satisfied"],
                                     "image": self._zone_view(zone)})
        return result

    def _zone_view(self, zone, half=.15, size=512):
        """The top-down camera at 512 px, cropped to the area within 15 cm of the zone centre (the crop is found from
        the latest depth view's back-projected points, so no simulator state is involved)."""
        rgb = self.render("topdown", size, size)
        xyz, (cx, cy) = self._masks["_xyz"], zone["center"][:2]
        near = np.isfinite(xyz[..., 0]) & (np.abs(xyz[..., 0] - cx) < half) & (np.abs(xyz[..., 1] - cy) < half)
        if not near.any():
            return rgb
        rows, cols = np.nonzero(near)
        k = size / xyz.shape[0]
        return rgb[int(rows.min() * k):int((rows.max() + 1) * k), int(cols.min() * k):int((cols.max() + 1) * k)].copy()

    # ---------------------------------------------------------------- private evaluator
    def private_thing_truth(self, scene=None):
        """Private, for scoring only (SAM perception): which of the task's items each thing the camera reports is,
        {thing name: private name, or None for a thing that is none of them (the laptop, a fragment)}. One to one
        (Hungarian) by the distance between centres (horizontal, plus the difference in the heights of the centres),
        at most 7 cm."""
        from scipy.optimize import linear_sum_assignment
        from .scene import PILE_HEIGHT, WALL_HALF_HEIGHT
        things = (scene or self.observe())["things"]
        items = {n: np.asarray(s["position"], float) for n, s in self.private_state().items()}
        for z in self.task.layout.zones:
            style = RECEPTACLES[z.color][0]
            items[z.name] = np.array([*z.xy, TABLE_Z + (PILE_HEIGHT / 2 if style == "pile" else WALL_HALF_HEIGHT[style])])
        names, keys = list(things), list(items)
        cost = np.array([[np.hypot(*(np.asarray(things[n]["center"][:2]) - items[k][:2])) +
                          abs(things[n]["center"][2] - items[k][2]) for k in keys] for n in names])
        out = {n: None for n in names}
        if names and keys:
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < .07:
                    out[names[i]] = keys[j]
        return out

    def private_state(self):
        sim = self.env.sim
        out = {}
        for name, body in self.env.item_body_ids.items():
            out[name] = {"position": sim.data.body_xpos[body].round(4).tolist(),
                         "x_axis": sim.data.body_xmat[body].reshape(3, 3)[:, 0].round(3).tolist(),
                         "z_axis": sim.data.body_xmat[body].reshape(3, 3)[:, 2].round(3).tolist()}
        return out

    def private_evaluate(self, goal):
        """Ground-truth success of object -> zone goals. Never shown to the agent."""
        state = self.private_state()
        zones = {z.name: {"center": z.xy, "long_axis_yaw": z.yaw, "style": RECEPTACLES[z.color][0],
                          "half_size": receptacle_half(RECEPTACLES[z.color][0], z)} for z in self.task.layout.zones}
        held = self.gripper_state()["holding_something"]
        details, ok_all = {}, True
        for name, zone in goal.items():
            pos = np.array(state[name]["position"])
            kind = name.split("_", 1)[1]
            half_h = KIND_HALF_HEIGHT[kind]
            if kind == "pen" and zones[zone]["style"] == "holder":
                # Standing in the cup: lower end inside the footprint and on the floor, centre within 1 cm of the
                # footprint, long axis within 40 degrees of vertical (a pen leaning on the rim of a 6 cm cup
                # with 8 cm walls tilts by up to about 37 degrees).
                from .assets import PEN_HALF
                axis = np.array(state[name]["x_axis"])
                bottom = min((pos + sgn * axis * PEN_HALF[0] for sgn in (-1, 1)), key=lambda e: e[2])
                inside = bool(inside_zone(bottom, zones[zone]) and inside_zone(pos, zones[zone], -.01))
                resting = bool(bottom[2] <= TABLE_Z + .004 + .02)
                upright = bool(abs(axis[2]) > np.cos(np.radians(40)))
                details[name] = {"zone": zone, "inside": inside, "resting": resting, "upright": upright,
                                 "tilt_from_vertical_deg": round(float(np.degrees(np.arccos(min(1.0, abs(axis[2]))))), 1),
                                 "position": state[name]["position"]}
                ok_all &= inside and resting and upright
                continue
            style = zones[zone]["style"]
            if style == "bin":  # round wastebasket: centre within the radius (the walls keep it below the rim)
                r = float(np.hypot(*(pos[:2] - np.asarray(zones[zone]["center"]))))
                inside = r <= zones[zone]["half_size"][0] - .005
                details_extra = {"distance_from_bin_centre": round(r, 4)}
            else:
                # Coaster: the mug's centre within 3 cm of the coaster's centre, so it stands on the coaster rather
                # than on its 6 mm rim.
                inside = inside_zone(pos, zones[zone], .015 if style == "coaster" else .005)
                details_extra = {}
            if style == "pile":  # on the pile of books: centre over its top (2 cm in), lying flat on it
                from .scene import PILE_HEIGHT
                inside = inside_zone(pos, zones[zone], .02)
                on_top = abs(pos[2] - (TABLE_Z + PILE_HEIGHT + half_h)) < .006
                flat = abs(state[name]["z_axis"][2]) > .99
                details[name] = {"zone": zone, "inside": bool(inside), "resting": bool(on_top), "upright": bool(flat),
                                 "position": state[name]["position"]}
                ok_all &= bool(inside and on_top and flat)
                continue
            if kind == "notebook":  # all four corners inside the tray (its centre inside is not enough for 21 cm)
                from .assets import NOTEBOOK_HALF
                R = np.array([state[name]["x_axis"], np.cross(state[name]["z_axis"], state[name]["x_axis"])])[:, :2]
                corners = [pos[:2] + sx * NOTEBOOK_HALF[0] * R[0] + sy * NOTEBOOK_HALF[1] * R[1]
                           for sx in (-1, 1) for sy in (-1, 1)]
                inside = inside and all(inside_zone(c, zones[zone], -.003) for c in corners)
            if kind == "pen":  # both ends inside the case (a pen across a wall has its centre inside too)
                from .assets import PEN_HALF
                tip = np.array(state[name]["x_axis"][:2]) * PEN_HALF[0]
                inside = inside and all(inside_zone(pos[:2] + sgn * tip, zones[zone]) for sgn in (-1, 1))
            inside = bool(inside)
            # base is 4 mm thick; a notebook lying over the 8 mm tray rim is 4 mm higher than one on the floor
            resting = bool(pos[2] <= TABLE_Z + .004 + half_h + {"pen": .006, "notebook": .003}.get(kind, .01))
            upright = bool(kind == "paper" or abs(state[name]["z_axis"][2]) > (.99 if kind == "notebook" else .9))
            ok = inside and resting and upright
            details[name] = {"zone": zone, "inside": inside, "resting": resting, "upright": upright,
                             "position": state[name]["position"], **details_extra}
            ok_all &= ok
        return {"success": bool(ok_all and not held), "objects": details, "gripper_holding": held}


def register():
    """Robosuite discovers environments by import; keep the import explicit."""
    return TabletopOrganize


register()

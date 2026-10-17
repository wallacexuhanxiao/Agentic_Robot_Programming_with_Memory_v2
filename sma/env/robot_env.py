"""Pose-level robot API over the robosuite tabletop, plus a private evaluator.

Agent-visible: top-down RGB-D perception, gripper/end-effector proprioception, and the
results of its own commands. Private (evaluator only): simulator object poses.
"""
import os

import numpy as np
from scipy.spatial.transform import Rotation

from .perception import object_on_zone, perceive
from .scene import KIND_HALF_HEIGHT, RECEPTACLE_HALF, RECEPTACLES, TABLE_Z, TabletopOrganize

HOME = np.array([-0.20, 0.0, 1.08])
WORKSPACE = np.array([[-0.32, 0.30], [-0.36, 0.36], [TABLE_Z + 0.004, 1.30]])


class RobotError(RuntimeError):
    """A command that could not be executed; carries structured details for feedback."""

    def __init__(self, reason, **details):
        super().__init__(f"{reason}: {details}" if details else reason)
        self.reason = reason
        self.details = details


class BudgetExceeded(RobotError):
    pass


class TabletopEnv:
    def __init__(self, task, *, image_size=256, video=True, video_every=3, step_budget=2400,
                 render_gpu=None):
        self.task = task
        self.image_size = image_size
        self.video = video
        self.video_every = video_every
        self.step_budget = step_budget
        self.render_gpu = int(os.environ.get("MUJOCO_EGL_DEVICE_ID", -1)) if render_gpu is None else render_gpu
        self.frames = []
        self.env = None

    # ---------------------------------------------------------------- lifecycle
    def reset(self):
        import robosuite as suite
        from robosuite import load_composite_controller_config
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
        if self.env is not None:
            self.close()
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
        self._ik_data = mujoco.MjData(model)
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
        d.qpos[:] = self.env.sim.data._data.qpos
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

    def _plan(self, target, goal):
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
            if residual > .004:
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
        scene = perceive(rgb, depth_m, self.K, self.world_from_camera)
        scene["gripper"] = self.gripper_state()
        scene["end_effector"] = {"position": [round(float(v), 4) for v in self._ee()],
                                 "yaw": round(self._yaw(), 3)}
        self._zones_seen = {**getattr(self, "_zones_seen", {}), **scene["zones"]}
        self.last_rgb = rgb
        return scene

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
        start = self.steps
        for q in path:  # waypoints ~2 cm / 0.08 rad apart, 4 control steps each (~0.1 m/s): small tracking lag
            for _ in range(4):
                self._step(q)
        q_goal = path[-1]
        for _ in range(max_steps):
            if np.linalg.norm(target - self._ee()) < tolerance:
                return {"ok": True, "position": self._ee().round(4).tolist(), "steps": self.steps - start}
            self._step(q_goal)
            if self.steps - start > len(path) + 40 and np.linalg.norm(self.obs.get("robot0_joint_vel", np.zeros(7))) < 1e-3:
                break
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
        return self._gripper(-1.0)

    def close_gripper(self):
        return self._gripper(1.0)

    def go_home(self):
        """Raise the arm and park it at the table edge (out of the top camera's view). Best effort."""
        try:
            if self._ee()[2] < HOME[2] - .03:
                here = self._ee()
                self.move_to(here[0], here[1], HOME[2], tolerance=.03)
            return self.move_to(*HOME, yaw=self.yaw0, tolerance=.03)
        except RobotError as exc:
            if exc.reason == "motion_blocked" and exc.details["remaining_error_m"] < .06:
                return {"ok": True, "position": self._ee().round(4).tolist(), "approximate": True}
            raise

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

    def wait(self, steps=20):
        hold = self._q()
        for _ in range(int(steps)):
            self._step(hold)

    def check_on_zone(self, object_name, zone_name):
        """Perception-only subgoal check (robot moved home first so it does not occlude)."""
        self.go_home()
        scene = self.observe()
        obj, zone = scene["objects"].get(object_name), scene["zones"].get(zone_name)
        if obj is None or zone is None:
            return {"satisfied": False, "reason": "not_visible", "object_visible": obj is not None,
                    "zone_visible": zone is not None}
        result = object_on_zone(obj, zone, scene["table_z"])
        result["gripper_holding"] = scene["gripper"]["holding_something"]
        result["satisfied"] = result["inside_zone"] and result["resting"] and not result["gripper_holding"]
        return result

    # ---------------------------------------------------------------- private evaluator
    def private_state(self):
        sim = self.env.sim
        out = {}
        for name, body in self.env.item_body_ids.items():
            out[name] = {"position": sim.data.body_xpos[body].round(4).tolist(),
                         "z_axis": sim.data.body_xmat[body].reshape(3, 3)[:, 2].round(3).tolist()}
        return out

    def private_evaluate(self, goal):
        """Ground-truth success of object -> zone goals. Never shown to the agent."""
        state = self.private_state()
        zones = {z.name: z.xy for z in self.task.layout.zones}
        halves = {z.name: RECEPTACLE_HALF[RECEPTACLES[z.color][0]] for z in self.task.layout.zones}
        held = self.gripper_state()["holding_something"]
        details, ok_all = {}, True
        for name, zone in goal.items():
            pos = np.array(state[name]["position"])
            kind = name.split("_", 1)[1]
            half_h = KIND_HALF_HEIGHT[kind]
            inside = bool(np.all(np.abs(pos[:2] - np.array(zones[zone])) <= halves[zone] - .005))
            resting = bool(pos[2] <= TABLE_Z + .004 + half_h + .01)  # receptacle base is 4 mm thick
            upright = bool(abs(state[name]["z_axis"][2]) > .9)
            ok = inside and resting and upright
            details[name] = {"zone": zone, "inside": inside, "resting": resting, "upright": upright,
                             "position": state[name]["position"]}
            ok_all &= ok
        return {"success": bool(ok_all and not held), "objects": details, "gripper_holding": held}


def register():
    """Robosuite discovers environments by import; keep the import explicit."""
    return TabletopOrganize


register()

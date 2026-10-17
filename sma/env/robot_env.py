"""Pose-level robot API over the robosuite tabletop, plus a private evaluator.

Agent-visible: top-down RGB-D perception, gripper/end-effector proprioception, and the
results of its own commands. Private (evaluator only): simulator object poses.
"""
import os

import numpy as np
from scipy.spatial.transform import Rotation

from .perception import object_on_zone, perceive
from .scene import KINDS, TABLE_Z, TabletopOrganize, ZONE_HALF

HOME = np.array([-0.26, 0.0, 1.08])
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
        config = load_composite_controller_config(controller="BASIC")
        config["body_parts"]["right"]["input_ref_frame"] = "world"
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
        for _ in range(10):
            self._step(np.zeros(3))
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

    def _step(self, translation, rotation=None):
        if self.steps >= self.step_budget:
            raise BudgetExceeded("control_step_budget_exhausted", budget=self.step_budget)
        action = np.zeros(7)
        action[:3] = np.clip(translation / .05, -.8, .8)
        if rotation is not None:
            action[3:6] = np.clip(rotation / .5, -.2, .2)
        action[6] = self.grip
        self.obs, _, _, _ = self.env.step(action)
        self.steps += 1
        if self.video and self.steps % self.video_every == 0:
            self.frames.append(self.render("demo", 320, 240))

    def render(self, camera, width, height, depth=False):
        out = self.env.sim.render(camera_name=camera, width=width, height=height, depth=depth)
        if depth:
            rgb, d = out
            return rgb[::-1].copy(), d[::-1].copy()
        return out[::-1].copy()

    # ---------------------------------------------------------------- agent API
    def observe(self):
        from robosuite.utils.camera_utils import get_real_depth_map
        rgb, depth = self.render("topdown", self.image_size, self.image_size, depth=True)
        depth_m = get_real_depth_map(self.env.sim, depth)
        if depth_m.ndim == 3:
            depth_m = depth_m[..., 0]
        scene = perceive(rgb, depth_m, self.K, self.world_from_camera)
        scene["gripper"] = self.gripper_state()
        scene["end_effector"] = {"position": [round(float(v), 4) for v in self._ee()],
                                 "yaw": round(self._yaw(), 3)}
        self.last_rgb = rgb
        return scene

    def gripper_state(self):
        q = np.asarray(self.obs["robot0_gripper_qpos"], float)
        width = float(abs(q[0] - q[1]))
        return {"width": round(width, 4), "closed_command": self.grip > 0,
                "holding_something": bool(self.grip > 0 and width > .006)}

    def move_to(self, x, y, z, yaw=None, tolerance=.004, max_steps=300):
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
        # representative in [-pi/2, pi/2) so the wrist stays away from its joint limits.
        yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
        goal = self._target_rotation(yaw)
        start = self.steps
        for _ in range(max_steps):
            error = target - self._ee()
            rot_err = (goal * Rotation.from_quat(self.obs["robot0_eef_quat_site"]).inv()).as_rotvec()
            if np.linalg.norm(error) < tolerance and np.linalg.norm(rot_err) < .03:
                return {"ok": True, "position": self._ee().round(4).tolist(), "steps": self.steps - start}
            self._step(error, rot_err)
        raise RobotError("motion_timeout", target=target.round(4).tolist(),
                         reached=self._ee().round(4).tolist(),
                         remaining_error_m=round(float(np.linalg.norm(target - self._ee())), 4))

    def _gripper(self, command, steps=25):
        self.grip = command
        hold = self._ee()
        for _ in range(steps):
            self._step(hold - self._ee())
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
                self.move_to(here[0], here[1], HOME[2], tolerance=.03, max_steps=250)
            return self.move_to(*HOME, yaw=self.yaw0, tolerance=.03, max_steps=300)
        except RobotError as exc:
            if exc.reason == "motion_timeout" and exc.details["remaining_error_m"] < .06:
                return {"ok": True, "position": self._ee().round(4).tolist(), "approximate": True}
            raise

    def wait(self, steps=20):
        hold = self._ee()
        for _ in range(int(steps)):
            self._step(hold - self._ee())

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
        held = self.gripper_state()["holding_something"]
        details, ok_all = {}, True
        for name, zone in goal.items():
            pos = np.array(state[name]["position"])
            kind = name.split("_", 1)[1]
            shape, size = KINDS[kind]
            half_h = size[2] if shape == "box" else size[1]
            inside = bool(np.all(np.abs(pos[:2] - np.array(zones[zone])) <= ZONE_HALF - .005))
            resting = bool(pos[2] <= TABLE_Z + half_h + .01)
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

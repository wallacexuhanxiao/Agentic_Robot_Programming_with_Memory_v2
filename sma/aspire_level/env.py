"""ASPIRE-level environment: the unmodified ASPIRE CapApi (segmentation -> Contact-GraspNet grasps, PyRoki/cuRobo
IK and planning, trajectory execution) driving our TabletopOrganize scene.

Only embodiment conventions are translated here (controller codec, grip-site TCP, Contact-GraspNet gripper frame),
following the adapter that was already validated on the pen task in aspire_qwen/bridge. Our perception summary,
camera check, private evaluator and video recording are reused through TabletopEnv's methods.
"""
import os
from dataclasses import replace

import numpy as np

from cap_harness.api import CapApi
from cap_harness.contracts import ArmCommand, GraspSet, RobotAction, Segmentation, SegmentationSet
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import matrix_to_pose
from cap_harness.providers.curobo.client import CuRoboProvider
from cap_harness.providers.graspnet.client import ContactGraspNetProvider
from cap_harness.providers.pyroki.client import PyRokiProvider
from cap_harness.robosuite.adapter import RobosuiteAdapter
from cap_harness.robosuite.registry import RobosuiteTaskMetadata, RobosuiteTaskRegistry

from ..env.perception import OBJECT_HUES, ZONE_HUES, _largest_component, _palette_label
from ..env.robot_env import RobotError, TabletopEnv

CAMERA = "topdown"


def task_metadata(task):
    return RobosuiteTaskMetadata("tabletop_organize", "TabletopOrganize", task.instruction, ("Panda",), (CAMERA,),
                                 horizon=100000)


class SettlingCapApi(CapApi):
    """Finish tracking the commanded joints before rejecting a Cartesian goal (as in the pen bridge)."""

    def _validate_pose_execution(self, execution, target_pose, tolerance, arm):
        checked = super()._validate_pose_execution(execution, target_pose, tolerance, arm)
        if checked.ok:
            return checked
        target = self._adapter._joint_targets[arm].copy()
        for tick in range(20):
            step = self._adapter.step(RobotAction(arms={arm: ArmCommand(
                mode="joint_position", target=target, embodiment=self._adapter.embodiment)}))
            if not step.ok or step.observation is None:
                return checked
            execution = replace(execution, final_observation=step.observation, steps_executed=execution.steps_executed + 1)
            checked = super()._validate_pose_execution(execution, target_pose, tolerance, arm)
            if checked.ok:
                return replace(checked, diagnostics={**checked.diagnostics, "additional_settling_steps": tick + 1})
        return checked


class AbsoluteJointCodec:
    """Robosuite 1.5 JOINT_POSITION with absolute input (ASPIRE's codec targets 1.4 deltas)."""

    def encode(self, action, current_state, *, joint_targets, gripper_targets):
        command = action.arms.get("primary")
        target = joint_targets["primary"] if command is None else command.target
        grip = gripper_targets["primary"]
        if command is not None and command.gripper_position is not None:
            grip = command.gripper_position
        return np.r_[target, 1.0 - 2.0 * grip]


class GripSiteGraspProvider:
    """Map Contact-GraspNet's raw gripper frame to the Panda grip site (validated on the pen task)."""

    def __init__(self, upstream):
        self.upstream = upstream

    def generate_grasps(self, point_cloud, **kwargs):
        result = self.upstream.generate_grasps(point_cloud, **kwargs)
        if not result.ok:
            return result
        correction = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        grasps = []
        for grasp in result.grasps:
            transform = grasp.pose.as_matrix().copy()
            transform[:3, :3] = transform[:3, :3] @ correction
            transform[:3, 3] += transform[:3, :3] @ np.array([0., 0., .1034])
            grasps.append(replace(grasp, pose=matrix_to_pose(transform, frame=point_cloud.frame),
                                  metadata={**grasp.metadata, "calibration": "raw network to Panda grip site"}))
        return GraspSet(ok=True, grasps=tuple(grasps), diagnostics=result.diagnostics)


class TabletopSegmentation:
    """ASPIRE segment_text provider for this scene: colour + height masks, or the Grounded-SAM2 service."""

    COLORS = tuple(OBJECT_HUES) + tuple(ZONE_HUES)

    def segment_text(self, observation, query):
        q = query.lower()
        color = next((c for c in self.COLORS if c in q), None)
        # Container colours (orange, cyan, purple) and object colours (red, yellow, green, blue) are disjoint, so a
        # container colour alone identifies a container even when the query omits "tray"/"box".
        place = any(w in q for w in ("tray", "box", "bin", "basket")) or color in ZONE_HUES
        mask = None
        if os.environ.get("SMA_PERCEPTION") == "gsam" and os.environ.get("SMA_SEG_URL"):
            mask = self._gsam_mask(observation, q, color, place)
        if mask is None and color is not None:
            mask = self._color_mask(observation, color, place)
        if mask is None or mask.sum() < 12:
            return SegmentationSet(ok=False, error=ApiError(ErrorCode.NO_SEGMENTATION,
                                                            f"no segment for {query!r} in the current image"))
        return SegmentationSet(ok=True, segmentations=(Segmentation(
            mask=mask, label=query, score=1.0, camera_name=observation.frame.split("/")[-1], frame=observation.frame),),
            diagnostics={"provider": os.environ.get("SMA_PERCEPTION", "color")})

    @staticmethod
    def _heights(observation):
        depth = observation.depth_m
        k = observation.intrinsics
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        cam = np.stack([(u - k[0, 2]) / k[0, 0] * depth, (v - k[1, 2]) / k[1, 1] * depth, depth], -1)
        m = observation.camera_pose.as_matrix()
        z = (cam.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3])[:, 2].reshape(h, w)
        valid = depth > 0
        table = np.median(z[valid])
        return z - table, valid

    def _color_mask(self, observation, color, place):
        height, valid = self._heights(observation)
        hues = ZONE_HUES if place else OBJECT_HUES
        if color not in hues:
            return None
        labels, names = _palette_label(observation.rgb, hues)
        band = (np.abs(height) < .008) if place else (height > .006)
        return _largest_component((labels == names.index(color)) & valid & band, 12)

    def _gsam_mask(self, observation, q, color, place):
        from ..env.perception_gsam import _dominant_hue_name, _kind_from_label, _request
        reply = _request(os.environ["SMA_SEG_URL"], observation.rgb)
        wanted = next((k for k in ("block", "mug", "book", "tray", "box") if k in q or (k == "mug" and "cup" in q)), None)
        for det in reply["detections"]:
            kind = _kind_from_label(det["label"])
            if wanted and kind != wanted:
                continue
            if color and _dominant_hue_name(observation.rgb, det["mask"], ZONE_HUES if place else OBJECT_HUES) != color:
                continue
            return det["mask"]
        return None

    def segment_points(self, observation, points_px, *, point_labels=None):
        return SegmentationSet(ok=False, error=ApiError(ErrorCode.UNSUPPORTED, "point prompts are not supported"))


class TabletopAspireAdapter(RobosuiteAdapter):
    def __init__(self, task, *, on_step=None, **kwargs):
        self.task = task
        self.on_step = on_step
        super().__init__(registry=RobosuiteTaskRegistry((task_metadata(task),)), **kwargs)

    def _make_environment(self, metadata):
        import robosuite as suite
        import robosuite.macros as macros
        from robosuite import load_composite_controller_config
        from robosuite.controllers import load_part_controller_config
        from ..env import scene  # noqa: F401  registers TabletopOrganize
        macros.IMAGE_CONVENTION = "opengl"  # the ASPIRE adapter flips RGB/depth to OpenCV itself
        config = load_composite_controller_config(controller="BASIC")
        arm = load_part_controller_config(default_controller="JOINT_POSITION")
        arm.update(input_type="absolute", kp=400, damping_ratio=1)
        arm["gripper"] = config["body_parts"]["right"]["gripper"]
        config["body_parts"]["right"] = arm
        return suite.make("TabletopOrganize", robots="Panda", layout=self.task.layout, controller_configs=config,
                          has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, use_object_obs=False,
                          camera_names=list(metadata.camera_names), camera_heights=self.camera_height,
                          camera_widths=self.camera_width, camera_depths=True, camera_segmentations=None,
                          control_freq=20, render_gpu_device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", -1)),
                          hard_reset=False, horizon=self.horizon, ignore_done=True, seed=self._seed)

    def _build_codec(self, metadata):
        return AbsoluteJointCodec()

    def _camera_observation(self, name):
        raw = self._raw_observation
        depth = np.asarray(raw[f"{name}_depth"], dtype=float)
        bad = ~np.isfinite(depth) | (depth < 0) | (depth > 1)
        if bad.any():  # float epsilon at the far plane: clamp before ASPIRE's metric conversion
            self._raw_observation = {**raw, f"{name}_depth": np.clip(np.nan_to_num(depth, nan=1.0), 0.0, 1.0)}
            try:
                return super()._camera_observation(name)
            finally:
                self._raw_observation = raw
        return super()._camera_observation(name)

    @staticmethod
    def _gripper_qpos_indexes(robot):
        return robot._ref_gripper_joint_pos_indexes["right"]

    @staticmethod
    def _grip_site(robot):
        return robot.gripper["right"].important_sites["grip_site"]

    def ik_request(self, target_pose, state, arm):
        if arm != "primary" or target_pose.frame != self.BASE_FRAME:
            raise ValueError("This desk has one primary arm in robot0_base")
        env = self.native_env
        hand = self._primary_from_world() @ self._body_transform(env.sim, "robot0_right_hand")
        tool = state.end_effector_poses[arm].as_matrix()
        target_hand = target_pose.as_matrix() @ np.linalg.inv(np.linalg.inv(hand) @ tool)
        # The upstream PyRoki/cuRobo clients subtract 10 cm along local z; compensate their reference TCP.
        target_hand[:3, 3] += target_hand[:3, :3] @ np.array([0., 0., .1])
        return matrix_to_pose(target_hand, frame=self.BASE_FRAME), state

    def planning_workspace(self):
        return ([-1.5, -1.5, -.5], [1.5, 1.5, 2.0])

    def check_success(self):
        return False  # scoring belongs to the private evaluator

    def native_step(self, action):
        self.native_env.sim._render_context_offscreen.gl_ctx.make_current()
        result = super().native_step(action)
        if result.ok and self.on_step:
            self.on_step()
        return result


def build_api(adapter):
    grasp = GripSiteGraspProvider(ContactGraspNetProvider(base_url=os.environ["CAP_HARNESS_GRASPNET_URL"], max_retries=0))
    curobo = CuRoboProvider(base_url=os.environ["CAP_HARNESS_CUROBO_URL"], max_retries=0)
    pyroki = PyRokiProvider(base_url=os.environ["CAP_HARNESS_PYROKI_URL"], max_retries=0)
    return SettlingCapApi(adapter, segmentation_provider=TabletopSegmentation(),
                          grasp_providers={"contact-graspnet": grasp},
                          ik_providers={"pyroki": pyroki, "curobo": curobo},
                          trajectory_planning_providers={"curobo": curobo},
                          integrated_pose_planning_providers={"curobo-integrated": curobo},
                          default_camera=CAMERA)


class AspireTabletopEnv(TabletopEnv):
    """TabletopEnv whose robot is driven by ASPIRE's CapApi. Perception summary, camera check, private evaluator
    and video come from TabletopEnv and read the same native Robosuite simulator."""

    def reset(self):
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
        self.frames, self.steps = [], 0
        self.adapter = TabletopAspireAdapter(self.task, on_step=self._after_step, camera_height=self.image_size,
                                             camera_width=self.image_size, horizon=100000)
        self.adapter.reset(task_metadata(self.task), self.task.seed)
        self.api = build_api(self.adapter)
        self.env = self.adapter.native_env
        self.obs = self.adapter._raw_observation
        sim = self.env.sim
        self.K = get_camera_intrinsic_matrix(sim, CAMERA, self.image_size, self.image_size)
        self.world_from_camera = get_camera_extrinsic_matrix(sim, CAMERA)
        self.base_from_world = self.adapter._primary_from_world()
        self.down_quaternion = np.asarray(self.api.get_robot_state().end_effector_poses["primary"].quaternion_wxyz)
        self.go_home()
        self.steps = 0
        return self.observe()

    def _after_step(self):
        self.obs = self.adapter._raw_observation
        self.steps += 1
        if self.video and self.steps % self.video_every == 0:
            frame = self.render("demo", 480, 360)
            hook = getattr(self, "frame_hook", None)
            self.frames.append(hook(frame) if hook else frame)

    def gripper_state(self):
        q = np.asarray(self.obs["robot0_gripper_qpos"], float)
        width = float(abs(q[0] - q[1]))
        closed = self.adapter._gripper_targets.get("primary", 1.0) < .5  # ASPIRE: 1 = open, 0 = closed
        return {"width": round(width, 4), "closed_command": closed, "holding_something": bool(closed and width > .006)}

    def go_home(self):
        """Park at our camera-clearing pose (ASPIRE's own home pose hangs over the table centre)."""
        from cap_harness.contracts import Pose
        from ..env.robot_env import HOME
        home = (self.base_from_world @ np.r_[HOME, 1.0])[:3]
        result = self.api.move_to_pose(Pose(home, self.down_quaternion, self.adapter.BASE_FRAME),
                                       tolerance=.03, max_steps=300)
        if not result.ok:
            result = self.api.go_home()
        return {"ok": bool(result.ok)}

    def close(self):
        if getattr(self, "adapter", None) is not None:
            self.adapter.close()
            self.adapter = None
            self.env = None

    # ------------------------------------------------------------- LLM-facing API (ASPIRE level)
    def _check_budget(self):
        if self.steps >= self.step_budget:
            raise RobotError("control_step_budget_exhausted", budget=self.step_budget)

    def observe_base(self):
        """observe() expressed in the robot base frame used by every ASPIRE tool."""
        scene = self.observe()
        t = self.base_from_world

        def tf(p):
            v = t @ np.r_[p[:3], 1.0]
            return [round(float(x), 4) for x in v[:3]]
        out = {"frame": self.adapter.BASE_FRAME, "table_z": round(float((t @ np.r_[0, 0, scene["table_z"], 1])[2]), 4),
               "objects": {}, "zones": {}, "gripper": scene["gripper"],
               "end_effector": {"position": tf(scene["end_effector"]["position"]), "yaw": scene["end_effector"]["yaw"]}}
        for name, o in scene["objects"].items():
            out["objects"][name] = {**{k: o[k] for k in ("name", "kind", "height", "footprint_long", "footprint_short",
                                                         "long_axis_yaw", "short_axis_yaw")},
                                    "center": tf(o["center"]), "top_z": round(o["top_z"] + float(t[2, 3]), 4)}
        for name, z in scene["zones"].items():
            out["zones"][name] = {"name": name, "center": tf(z["center"]), "half_size": z["half_size"]}
        return out

    def aspire_api(self):
        api = self.api

        def checked(name, fn):
            def call(*args, **kwargs):
                self._check_budget()
                return fn(*args, **kwargs)
            call.__name__ = name
            return call

        def move_to_pose(target_pose, tolerance=0.01, max_steps=250, strategy=None):
            self._check_budget()
            result = api.move_to_pose(target_pose, tolerance=tolerance, max_steps=max_steps, strategy=strategy)
            if not result.ok:
                raise RobotError("move_to_pose_failed", message=str(result.error)[:300])
            return result

        def gripper(fn):
            def call():
                self._check_budget()
                fn()
                for _ in range(10):  # let the fingers settle before reporting
                    api.step(RobotAction(arms={"primary": ArmCommand(
                        mode="joint_position", target=self.adapter._joint_targets["primary"],
                        embodiment=self.adapter.embodiment)}))
                return self.gripper_state()
            return call

        from cap_harness.contracts import MotionStrategy, Pose
        return {"observe": self.observe_base, "get_robot_state": api.get_robot_state,
                "get_task_context": api.get_task_context,
                "localize_object": checked("localize_object", api.localize_object),
                "generate_grasps": checked("generate_grasps", api.generate_grasps),
                "select_grasp": api.select_grasp, "estimate_geometry": api.estimate_geometry,
                "crop_point_cloud": api.crop_point_cloud, "move_to_pose": move_to_pose,
                "open_gripper": gripper(api.open_gripper), "close_gripper": gripper(api.close_gripper),
                "go_home": checked("go_home", self.go_home), "Pose": Pose, "MotionStrategy": MotionStrategy}

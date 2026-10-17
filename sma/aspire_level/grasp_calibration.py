"""Contact-GraspNet gripper frame -> Panda grip site (validated on the pen task in aspire_qwen/bridge). Kept apart
from env.py so the mid-level API can use it without importing the ASPIRE robosuite adapter."""
from dataclasses import replace

import numpy as np

from cap_harness.contracts import GraspSet
from cap_harness.geometry import matrix_to_pose


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

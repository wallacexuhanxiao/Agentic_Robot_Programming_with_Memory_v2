"""Compare top-down perception with private object poses on a few layouts."""
import json, sys
import numpy as np
import imageio.v2 as imageio
from sma.env.robot_env import TabletopEnv
from sma.env.tasks import make_task

out = sys.argv[1]
for family, seed in (("place", 7), ("sort", 1000), ("clear", 1100)):
    task = make_task(family, seed)
    env = TabletopEnv(task, video=False)
    scene = env.reset()
    truth = env.private_state()
    imageio.imwrite(f"{out}/{task.task_id}_topdown.png", env.last_rgb)
    imageio.imwrite(f"{out}/{task.task_id}_demo.png", env.render("demo", 480, 360))
    print(task.task_id, task.instruction, task.goal)
    for name, t in truth.items():
        p = scene["objects"].get(name)
        err = None if p is None else np.round(1000 * (np.array(p["center"][:2]) - t["position"][:2]), 1).tolist()
        print("  ", name, "detected" if p else "MISSING", "xy_err_mm", err, p and {k: p[k] for k in ("height", "footprint_long", "footprint_short", "long_axis_yaw")})
    extra = set(scene["objects"]) - set(truth)
    print("   extra detections:", extra, "zones:", {k: v["center"][:2] for k, v in scene["zones"].items()}, "true zones:", {z.name: z.xy for z in task.layout.zones})
    print("   table_z", scene["table_z"], "ee", scene["end_effector"], scene["gripper"])
    env.close()

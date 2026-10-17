"""Handwritten pose-level pick-and-place (not an agent result): checks the API is physically sufficient."""
import json, sys, math, time
import imageio.v2 as imageio
from sma.env.robot_env import TabletopEnv
from sma.env.tasks import make_task

def place(env, obj_name, zone_name):
    scene = env.observe()
    obj, zone = scene["objects"][obj_name], scene["zones"][zone_name]
    grip_h = min(obj["height"] - .015, .045)          # grasp depth below the top surface
    yaw = obj["long_axis_yaw"] + math.pi / 2           # close across the short side
    x, y = obj["center"][:2]
    env.open_gripper()
    env.move_to(x, y, obj["top_z"] + .10, yaw)
    env.move_to(x, y, scene["table_z"] + grip_h, yaw)
    g = env.close_gripper()
    env.move_to(x, y, scene["table_z"] + .18, yaw)
    zx, zy = zone["center"][:2]
    env.move_to(zx, zy, scene["table_z"] + .18, yaw)
    env.move_to(zx, zy, scene["table_z"] + grip_h + .006, yaw)
    env.open_gripper()
    env.move_to(zx, zy, scene["table_z"] + .18, yaw)
    return g

out = sys.argv[1]
results = []
for kind, seed in (("block", 11), ("book", 12), ("mug", 13)):
    task = make_task("place", seed, kinds=[kind])
    env = TabletopEnv(task)
    env.reset()
    name, zone = next(iter(task.goal.items()))
    t = time.time()
    g = place(env, name, zone)
    check = env.check_on_zone(name, zone)
    truth = env.private_evaluate(task.goal)
    imageio.mimwrite(f"{out}/hand_{kind}.mp4", env.frames, fps=20)
    results.append({"kind": kind, "grip_after_close": g, "perception_check": check, "private": truth,
                    "steps": env.steps, "seconds": round(time.time() - t, 1)})
    print(kind, "closed", g, "| perception satisfied", check["satisfied"], "| GT", truth["success"], truth["objects"][name], "| steps", env.steps)
    env.close()
json.dump(results, open(f"{out}/handwritten_place.json", "w"), indent=1)

"""Can the simulator run without a GPU (Mesa software EGL)? Renders, perceives and runs one scripted pen placement
(geometry grasp + place_object), timing each part. No model, no GraspNet."""
import sys
import time

from sma.agent.loop import make_env
from sma.env.tasks import suite

task = suite("pen")[0]
obj, zone = next(iter(task.goal.items()))
env = make_env(task)
t0 = time.time()
scene = env.reset()
print(f"reset {time.time() - t0:.1f}s objects {sorted(scene['objects'])} zones {sorted(scene['zones'])}", flush=True)
try:
    from OpenGL import GL
    print("GL renderer:", GL.glGetString(GL.GL_RENDERER), GL.glGetString(GL.GL_VERSION), flush=True)
except Exception as exc:  # the context may not be current here; the timings below still tell
    print("GL renderer query failed:", type(exc).__name__, exc, flush=True)
t0 = time.time()
for _ in range(5):
    env.observe()
print(f"observe x5 {time.time() - t0:.2f}s", flush=True)
t0 = time.time()
for _ in range(20):
    env.render("demo", 480, 360)
print(f"demo render x20 {time.time() - t0:.2f}s", flush=True)
t0 = time.time()
cands = env.grasp_candidates(obj)
pick = next(c for c in cands if c["source"].startswith("geometry") and c["fits_open_fingers"])
g = env.grasp(pick)
s = env.free_spots(zone)["spots"][0]
r = env.place_object(obj, s["x"], s["y"], s["object_long_axis_yaw"])
gt = env.private_evaluate(task.goal)
print(f"scripted placement {time.time() - t0:.1f}s, {env.steps} control steps, {len(env.frames)} video frames; "
      f"hold {g['holding_something']} yaw_err {r.get('yaw_error_deg')} GT {gt['success']}", flush=True)
env.close()

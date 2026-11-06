"""Hue sweep for the cdesk palette: the scene lighting and the renderer pull hues together (nominal 145 and 160 both render
near 150), so the palette is chosen from measurements. 45 toy blocks with nominal OpenCV hues 0, 4, ..., 176 (the
palette's saturation and value) on a grid; the top-down camera's median hue over each block's top; then 12 nominal hues
whose rendered hues are as evenly spread as possible.
Usage: calibrate_hue_sweep.py OUT.json"""
import json
import sys

import cv2
import numpy as np

from sma.agent.loop import make_env
from sma.env.palette import _rgba
from sma.env.scene import Layout, ObjectSpec
from sma.env.tasks import Task

nominal = list(range(0, 180, 4))
xs, ys = np.linspace(-0.26, 0.22, 9), np.linspace(-0.24, 0.24, 5)
objects = [ObjectSpec(f"h{h}", "block", (round(float(xs[i % 9]), 3), round(float(ys[i // 9]), 3)), 0.0, rgba=_rgba(h))
           for i, h in enumerate(nominal)]
task = Task("hue_sweep", "calibration", "", Layout(objects, []), {}, 1)
env = make_env(task, video=False)
try:
    env.reset()
    rgb = env.render("topdown", 512, 512)
    from robosuite.utils.camera_utils import get_camera_intrinsic_matrix
    from sma.env.perception import back_project
    _, depth = env.render("topdown", 512, 512, depth=True)
    xyz = back_project(env._metric_depth(depth), get_camera_intrinsic_matrix(env.env.sim, "topdown", 512, 512),
                       env.world_from_camera)
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV).astype(float)
    table = float(np.nanmedian(xyz[..., 2]))
    state = env.private_state()
    rendered = {}
    for spec, h in zip(objects, nominal):
        c = state[spec.name]["position"]
        top = (np.hypot(xyz[..., 0] - c[0], xyz[..., 1] - c[1]) < .012) & (xyz[..., 2] > table + .03) & (hsv[..., 1] > 60)
        hues = hsv[..., 0][top]
        if h < 10 or h > 170:
            hues = np.where(hues > 90, hues - 180, hues)
        rendered[h] = {"median": round(float(np.median(hues)), 1) if hues.size else None, "pixels": int(hues.size),
                       "spread": round(float(np.percentile(hues, 95) - np.percentile(hues, 5)), 1) if hues.size else None,
                       "saturation": round(float(np.median(hsv[..., 1][top])), 1) if hues.size else None}
        print(h, rendered[h], flush=True)
finally:
    env.close()
points = [(h, r["median"] % 180) for h, r in rendered.items() if r["median"] is not None]
best = None
for offset in np.arange(0, 15, 1.0):  # 12 evenly spaced rendered targets; the nominal hue rendering closest to each
    targets = [(offset + 15 * k) % 180 for k in range(12)]
    pick = [min(points, key=lambda p: min(abs(p[1] - t), 180 - abs(p[1] - t))) for t in targets]
    got = sorted(p[1] for p in pick)
    gap = min([b - a for a, b in zip(got, got[1:])] + [got[0] + 180 - got[-1]])
    if len({p[0] for p in pick}) == 12 and (best is None or gap > best[0]):
        best = (gap, pick)
print("CHOSEN (nominal, rendered):", best[1] if best else None, "smallest rendered gap", best[0] if best else None, flush=True)
json.dump({"rendered": rendered, "chosen": best[1] if best else None, "smallest_gap": best[0] if best else None},
          open(sys.argv[1], "w"), indent=1)

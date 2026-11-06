"""Rendered hue of each palette colour on the colour-coded desk (cdesk), for perception's palette (palette.RENDERED_HUE).
For a few desks: the top-down camera's pixels over each thing (objects within 1.5 cm of their centre, receptacles over
their footprint, from the private layout), saturated ones only; per colour the median, 5th and 95th percentile hue and
the median saturation, and the smallest gap between the medians of two colours.
Usage: calibrate_palette.py OUT.json [SUITE]"""
import json
import sys

import cv2
import numpy as np

from sma.agent.loop import make_env
from sma.env.scene import RECEPTACLES, receptacle_half
from sma.env.tasks import suite

samples = {}
for task in suite(sys.argv[2] if len(sys.argv) > 2 else "cdesk@16000"):
    env = make_env(task, video=False)
    try:
        env.reset()
        rgb, xyz, table = env.last_rgb, env._masks["_xyz"], env._table_z
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV).astype(float)
        sat = hsv[..., 1] >= 80
        state = env.private_state()
        for spec in task.layout.objects:
            c = np.array(state[spec.name]["position"][:2])
            near = (np.hypot(xyz[..., 0] - c[0], xyz[..., 1] - c[1]) < .015) & (xyz[..., 2] > table + .003) & sat
            samples.setdefault(spec.color, []).extend(hsv[..., 0][near].tolist())
            samples.setdefault(spec.color + ":sat", []).extend(hsv[..., 1][near].tolist())
        for zone in task.layout.zones:
            hx, hy = receptacle_half(RECEPTACLES[zone.color][0], zone)
            near = (np.abs(xyz[..., 0] - zone.xy[0]) < hx + .008) & (np.abs(xyz[..., 1] - zone.xy[1]) < hy + .008) & sat
            samples.setdefault(zone.paint, []).extend(hsv[..., 0][near].tolist())
            samples.setdefault(zone.paint + ":sat", []).extend(hsv[..., 1][near].tolist())
    finally:
        env.close()
out = {}
for name, hues in samples.items():
    if name.endswith(":sat"):
        continue
    h = np.array(hues)
    if h.size and np.median(h) < 20 and (h > 150).any():  # red wraps around 180
        h = np.where(h > 90, h - 180, h)
    out[name] = {"median": round(float(np.median(h)), 1), "p5": round(float(np.percentile(h, 5)), 1),
                 "p95": round(float(np.percentile(h, 95)), 1), "pixels": int(h.size),
                 "saturation": round(float(np.median(samples[name + ":sat"])), 1)}
    print(name, out[name], flush=True)
medians = sorted((v["median"] % 180, k) for k, v in out.items())
gaps = [(round(b[0] - a[0], 1), a[1], b[1]) for a, b in zip(medians, medians[1:])] + \
       [(round(medians[0][0] + 180 - medians[-1][0], 1), medians[-1][1], medians[0][1])]
print("SMALLEST GAPS", sorted(gaps)[:4], flush=True)
json.dump({"colours": out, "gaps": sorted(gaps)}, open(sys.argv[1], "w"), indent=1)

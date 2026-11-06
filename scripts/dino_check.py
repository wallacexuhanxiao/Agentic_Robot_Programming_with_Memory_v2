"""Grounding DINO as a recogniser (no LLM): for every thing the camera reports on each desk, the vocabulary's labels are
scored by Grounding DINO (segmentation service /classify) on (a) the top-down camera cropped around the thing, (b) the
side (demo) camera cropped around it, (c) the whole top-down image with the thing's box; the best label is compared with
the private category. Things that are none of the task's items ("other": the laptop, a fragment) are reported apart:
the detector has no "other". Usage (segmentation service running, SMA_SEG_URL): dino_check.py OUT_DIR SUITE [SUITE ...]"""
import base64
import json
import os
import sys
import urllib.request

import cv2
import numpy as np

from sma.agent.identify import private_truth
from sma.agent.loop import make_env
from sma.env.naming import VOCABULARY
from sma.env.tasks import suite

out = sys.argv[1]
os.makedirs(f"{out}/crops", exist_ok=True)
URL = os.environ["SMA_SEG_URL"].rstrip("/") + "/classify"
RENDERS = {"topdown": (1024, 1024), "demo": (1280, 960)}


def classify(items):
    body = json.dumps({"labels": VOCABULARY, "items": items}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        reply = json.loads(r.read())
    if "error" in reply:
        raise RuntimeError(reply["error"])
    return reply["scores"]


def png(image):
    return base64.b64encode(cv2.imencode(".png", image[..., ::-1])[1].tobytes()).decode()


def crop_around(image, box, w, h, context=2.0, least=96):
    """A square crop around box (x0, y0, x1, y1), `context` times its larger side, and the box in the crop's pixels."""
    x0, y0, x1, y1 = box
    side = int(min(max(max(x1 - x0, y1 - y0) * context, least), w, h))
    a = int(np.clip((x0 + x1) / 2 - side / 2, 0, w - side))
    b = int(np.clip((y0 + y1) / 2 - side / 2, 0, h - side))
    return image[b:b + side, a:a + side], [x0 - a, y0 - b, x1 - a, y1 - b]


rows = []
for name in sys.argv[2:]:
    for task in suite(name):
        env = make_env(task, video=False)
        try:
            env.reset()
            scene = env.observe()
            truth, _ = private_truth(env, task, scene)
            images = {c: np.ascontiguousarray(env.render(c, w, h)) for c, (w, h) in RENDERS.items()}
            items, keys = [], []
            for thing in scene["things"]:
                for camera, (w, h) in RENDERS.items():
                    outline = env._thing_outline(thing, camera, h, w)
                    if outline is None:
                        continue
                    pts = np.vstack([c.reshape(-1, 2) for c in outline])
                    box = [*pts.min(0).tolist(), *pts.max(0).tolist()]
                    crop, inner = crop_around(images[camera], box, w, h)
                    items.append({"image_png_b64": png(crop), "box": inner})
                    keys.append((thing, f"{camera}_crop"))
                    cv2.imwrite(f"{out}/crops/{task.task_id}_{thing}_{camera}.png", crop[..., ::-1])
                    if camera == "topdown":
                        items.append({"image_png_b64": png(images[camera]), "box": box})
                        keys.append((thing, "topdown_whole"))
            scores = {}
            for (thing, view), s in zip(keys, classify(items)):
                scores.setdefault(thing, {})[view] = s
        finally:
            env.close()
        for thing, by_view in scores.items():
            row = {"task": task.task_id, "thing": thing, "truth": truth.get(thing, "other")}
            if "topdown_crop" in by_view and "demo_crop" in by_view:
                by_view["both_crops"] = {k: (by_view["topdown_crop"][k] + by_view["demo_crop"][k]) / 2 for k in VOCABULARY}
            for view, s in by_view.items():
                best = max(s, key=s.get)
                row[view] = {"said": best, "score": s[best], "truth_score": s.get(row["truth"])}
            rows.append(row)
            print(task.task_id, thing, row["truth"], {v: (row[v]["said"], row[v]["score"]) for v in by_view}, flush=True)

views = ("topdown_crop", "demo_crop", "both_crops", "topdown_whole")
known = [r for r in rows if r["truth"] != "other"]
summary = {"things": len(known), "other_things": len(rows) - len(known),
           "correct": {v: sum(r.get(v, {}).get("said") == r["truth"] for r in known) for v in views},
           "confusions": {v: {} for v in views}}
for r in known:
    for v in views:
        if v in r and r[v]["said"] != r["truth"]:
            key = f"{r['truth']} -> {r[v]['said']}"
            summary["confusions"][v][key] = summary["confusions"][v].get(key, 0) + 1
json.dump({"summary": summary, "rows": rows}, open(f"{out}/summary.json", "w"), indent=1)
print("SUMMARY", json.dumps(summary), flush=True)

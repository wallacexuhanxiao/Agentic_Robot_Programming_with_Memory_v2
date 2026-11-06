"""Where the things the camera reports come from (private diagnosis, no LLM). For one shared-colour desk, at the start
and after the act-level reference program has stood the out-of-place pen in the holder: every SAM2 mask of the last
look (the 16-point grid, the point prompts of the second look) and the depth-only regions, the things kept, and for each
thing the layer it came from (the input mask it matches best) and the simulator bodies its pixels show (MuJoCo
segmentation render of the top-down camera). For comparison, the masks of a denser 32-point grid on the same image.
Usage (SMA_API_LEVEL=act, segmentation service running; run through track_debug.py with SMA_TRACK_GATE=sam_debug.py):
sam_debug.py OUT_DIR INDEX"""
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np

from sma.agent.identify import private_truth, role_of
from sma.agent.loop import make_env
from sma.agent.sandbox import Sandbox
from sma.env import perception_sam
from sma.env.tasks import suite

out, index = sys.argv[1], int(sys.argv[2])
os.makedirs(out, exist_ok=True)
from reference_programs import PROGRAM  # noqa: E402  (scripts/ is on the path)

look = {"requests": [], "kept": None}
_request, _segment = perception_sam.request_masks, perception_sam.segment_things


def request(url, rgb, points_per_side=16, iou_threshold=.8, points=None):
    masks = _request(url, rgb, points_per_side, iou_threshold, points)
    look["requests"].append({"kind": "grid" if points is None else "points", "points": points, "masks": masks})
    return masks


def segment(rgb, height, valid, masks, **k):
    kept = _segment(rgb, height, valid, masks, **k)
    look["kept"], look["inputs"] = kept, masks
    return kept


perception_sam.request_masks, perception_sam.segment_things = request, segment


def bodies(env, seg, mask, top=3):
    m = env.env.sim.model._model
    ids = seg[mask]
    names = Counter(m.body(int(m.geom_bodyid[g])).name if t == 5 and g >= 0 else "none" for t, g in ids)
    total = sum(names.values())
    return [(n, round(c / total, 2)) for n, c in names.most_common(top)]


def iou(a, b):
    inter = int((a & b).sum())
    return inter / max(int((a | b).sum()), 1)


def source(mask):
    """The input mask a kept thing matches best: grid / points / depth-only region, with the IoU."""
    best = ("?", 0.0)
    for r in look["requests"]:
        for _, m in r["masks"]:
            v = iou(mask, m)
            if v > best[1]:
                best = (r["kind"], round(v, 2))
    for score, m in look.get("inputs") or []:
        if score is None:
            v = iou(mask, m)
            if v > best[1]:
                best = ("depth-only", round(v, 2))
    return best


def outline(image, masks, colour, labels=None):
    for k, m in enumerate(masks):
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image, contours, -1, colour, 1)
        if labels:
            ys, xs = np.nonzero(m)
            if len(xs):
                cv2.putText(image, str(labels[k]), (int(xs.mean()) - 8, int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, .32,
                            (255, 255, 255), 1, cv2.LINE_AA)
    return image


def report(env, task, tag):
    look["requests"].clear()
    scene = env.observe()
    rgb = env.last_rgb.copy()
    seg = env.env.sim.render(width=env.image_size, height=env.image_size, camera_name="topdown", segmentation=True)[::-1]
    truth, _ = private_truth(env, task, scene)
    rows = []
    for name in sorted(scene["things"], key=lambda n: int(n[5:]) if n[5:].isdigit() else 0):
        mask = env._masks.get(name)
        if mask is None:
            continue
        t = scene["things"][name]
        rows.append({"thing": name, "truth": truth.get(name), "pixels": int(mask.sum()), "source": source(mask),
                     "bodies": bodies(env, seg, mask), "height": t.get("height"),
                     "center": [round(float(v), 3) for v in t["center"]]})
    grid = [m for r in look["requests"] if r["kind"] == "grid" for _, m in r["masks"]]
    pts = [m for r in look["requests"] if r["kind"] == "points" for _, m in r["masks"]]
    depth = [m for s, m in (look.get("inputs") or []) if s is None]
    image = outline(outline(outline(rgb[..., ::-1].copy(), grid, (0, 200, 255)), pts, (255, 0, 255)), depth, (0, 255, 0))
    cv2.imwrite(f"{out}/{task.task_id}_{tag}_inputs.png", image)  # orange: grid, magenta: point prompts, green: depth
    names = [r["thing"] for r in rows]
    cv2.imwrite(f"{out}/{task.task_id}_{tag}_things.png",
                outline(rgb[..., ::-1].copy(), [env._masks[n] for n in names], (0, 255, 255), [n[5:] for n in names]))
    dense = _request(os.environ["SMA_SEG_URL"], rgb, 32)
    cv2.imwrite(f"{out}/{task.task_id}_{tag}_grid32.png", outline(rgb[..., ::-1].copy(), [m for _, m in dense], (0, 200, 255)))
    print(task.task_id, tag, f"grid masks {len(grid)}, point-prompt masks {len(pts)}, depth-only regions {len(depth)}, "
          f"32-point grid masks {len(dense)}", flush=True)
    for r in rows:
        print("  ", json.dumps(r), flush=True)
    return {"rows": rows, "masks": {"grid": len(grid), "points": len(pts), "depth_only": len(depth), "grid32": len(dense)}}


task = suite(os.environ.get("SMA_GATE_SUITE", "sdesk@17000"))[index]
env = make_env(task)
result = {"task": task.task_id}
try:
    env.reset()
    result["start"] = report(env, task, "start")
    scene = env.observe()
    truth, seen_as = private_truth(env, task, scene)
    env.set_identity({t: {"category": c, "role": role_of(c)} for t, c in truth.items()})
    kind_of = {o.name: o.kind for o in task.layout.objects}
    pen = next((o for o in task.meta["out_of_place"] if kind_of[o] == "pen" and o in seen_as), None)
    if pen and task.goal[pen] in seen_as:
        name, zone = seen_as[pen], seen_as[task.goal[pen]]
        env.steps, env.step_budget = 0, 6000
        ok, error = Sandbox(env).run(PROGRAM["pen"].format(obj=name, zone=zone))
        env.step_budget = env.steps + 3000
        check = env.check_on_zone(name, zone)
        gt = env.private_evaluate({pen: task.goal[pen]})["objects"][pen]
        print(task.task_id, "pen", name, "program_ok", ok, error, "camera", check.get("satisfied"),
              "gt", gt.get("inside") and gt.get("resting") and gt.get("upright"), flush=True)
        print("   check", json.dumps(check, default=str)[:500], flush=True)
        result["pen"] = {"thing": name, "zone": zone, "program_ok": ok, "check": check, "gt": gt}
        result["after"] = report(env, task, "after_pen")
finally:
    env.close()
json.dump(result, open(f"{out}/{task.task_id}.json", "w"), indent=1, default=str)

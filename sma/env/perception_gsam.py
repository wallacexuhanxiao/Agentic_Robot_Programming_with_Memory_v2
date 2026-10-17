"""Grounded-SAM2 perception: text-prompted detection + SAM2 masks from the segmentation service, then the same
depth geometry as the colour pipeline. Output has the same schema as perception.perceive().

Segmentation no longer relies on colour. Colour is used only as an attribute for naming (the dominant hue
inside a mask), so objects and containers keep names like red_mug or cyan_tray.
"""
import base64
import io
import json
import urllib.request

import cv2
import numpy as np
from PIL import Image

from .perception import OBJECT_HUES, ZONE_HUES, ZONE_NAMES, _footprint, back_project, classify_kind

PROMPTS = ["toy block", "mug", "book", "tray", "box"]
LABEL_KIND = {"block": "block", "cube": "block", "mug": "mug", "cup": "mug", "book": "book", "notebook": "book",
              "tray": "tray", "box": "box"}


def _request(url, rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    body = json.dumps({"image_png_b64": base64.b64encode(buf.getvalue()).decode(), "prompts": PROMPTS}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/segment", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        reply = json.loads(r.read())
    if "error" in reply:
        raise RuntimeError("segmentation service: " + reply["error"])
    for d in reply["detections"]:
        d["mask"] = np.asarray(Image.open(io.BytesIO(base64.b64decode(d.pop("mask_png_b64"))))) > 127
    return reply


def _dominant_hue_name(rgb, mask, hues, min_saturation=80, tolerance=12):
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    sel = mask & (hsv[..., 1] >= min_saturation)
    if sel.sum() < 10:
        return None
    h = hsv[..., 0][sel].astype(float)
    names = list(hues)
    ref = np.array([hues[n] for n in names], float)
    d = np.abs(h[:, None] - ref[None])
    d = np.minimum(d, 180 - d)
    votes = np.bincount(d.argmin(1)[d.min(1) <= tolerance], minlength=len(names))
    return names[int(votes.argmax())] if votes.max() > 0 else None


def _kind_from_label(label):
    for word, kind in LABEL_KIND.items():
        if word in label.lower():
            return kind
    return None


def perceive_gsam(rgb, depth_m, K, world_from_camera, url, *, min_pixels=25):
    xyz = back_project(depth_m, K, world_from_camera)
    valid = np.isfinite(xyz[..., 2])
    table_z = float(np.median(xyz[..., 2][valid]))
    height = xyz[..., 2] - table_z
    reply = _request(url, rgb)
    objects, zones, raw = {}, {}, []
    taken = np.zeros(rgb.shape[:2], bool)
    for det in reply["detections"]:  # highest score first
        mask = det["mask"] & valid & ~taken
        label_kind = _kind_from_label(det["label"])
        raw.append({"label": det["label"], "score": round(det["score"], 3), "pixels": int(mask.sum())})
        if mask.sum() < min_pixels:
            continue
        pts = xyz[mask]
        center_xy, long_side, short_side, yaw = _footprint(pts[:, :2])
        top_z = float(np.quantile(pts[:, 2], .9))
        h = top_z - table_z
        receptacle = long_side > .10 or label_kind in ("tray", "box") and long_side > .08
        if receptacle:
            base = mask & (np.abs(height) < .008)
            color = _dominant_hue_name(rgb, base if base.sum() >= 10 else mask, ZONE_HUES)
            name = ZONE_NAMES.get(color)
            if name is None or name in zones:
                continue
            floor = xyz[base] if base.sum() >= min_pixels else pts
            lo, hi = floor[:, :2].min(0), floor[:, :2].max(0)
            zones[name] = {"name": name, "center": [round(float((lo[0] + hi[0]) / 2), 4),
                                                   round(float((lo[1] + hi[1]) / 2), 4), round(table_z, 4)],
                           "half_size": [round(float((hi[0] - lo[0]) / 2), 4), round(float((hi[1] - lo[1]) / 2), 4)],
                           "detector_label": det["label"], "detector_score": round(det["score"], 3)}
            continue
        above = mask & (height > .006)  # the object itself, not the table or tray floor around it
        if above.sum() < min_pixels:
            continue
        pts = xyz[above]
        center_xy, long_side, short_side, yaw = _footprint(pts[:, :2])
        top_z = float(np.quantile(pts[:, 2], .9))
        h = top_z - table_z
        geo_kind = classify_kind(h, long_side, short_side)
        kind = label_kind if label_kind in ("block", "mug", "book") else geo_kind
        color = _dominant_hue_name(rgb, above, OBJECT_HUES) or "unknown"
        name = f"{color}_{kind}"
        if name in objects:
            continue
        taken |= above
        objects[name] = {
            "name": name, "kind": kind, "color": color,
            "center": [round(center_xy[0], 4), round(center_xy[1], 4), round(table_z + h / 2, 4)],
            "top_z": round(top_z, 4), "height": round(h, 4),
            "footprint_long": round(long_side, 4), "footprint_short": round(short_side, 4),
            "long_axis_yaw": round(yaw, 3), "short_axis_yaw": round((yaw + np.pi) % np.pi - np.pi / 2, 3),
            "detector_label": det["label"], "detector_score": round(det["score"], 3),
            "kind_from_label": label_kind, "kind_from_geometry": geo_kind, "visible_pixels": int(above.sum())}
    return {"table_z": round(table_z, 4), "objects": objects, "zones": zones,
            "perception": {"backend": "grounding-dino-tiny + sam2-hiera-large", "raw_detections": raw,
                           "service_seconds": reply.get("seconds")}}

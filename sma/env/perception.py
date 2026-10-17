"""Top-down RGB-D perception: colored objects above the table and colored zones on it.

Uses only the camera image, depth and calibration (available on a real calibrated camera).
Never reads simulator object state.
"""
import cv2
import numpy as np


KIND_HEIGHT = {"block": .04, "book": .025, "mug": .07, "pen": .012}
ZONE_NAMES = {"orange": "orange_tray", "cyan": "cyan_tray", "purple": "purple_box",
              "lime": "lime_case"}  # receptacle catalogue


def back_project(depth_m, K, world_from_camera):
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(depth_m)], -1)
    cam = rays * depth_m[..., None]
    world = cam.reshape(-1, 3) @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]
    return world.reshape(h, w, 3)


# Rendered hue (OpenCV 0-180) of each palette color under the scene lighting, measured once
# from a calibration render (scripts/calibrate_hues.py). Objects and zones are segmented in
# separate height bands, so the two sets only need to be distinct within themselves.
OBJECT_HUES = {"red": 1, "yellow": 30, "green": 62, "blue": 113}
ZONE_HUES = {"orange": 23, "cyan": 90, "purple": 144, "lime": 45}  # lime: pen case (calibrated by pen_gate)


def _palette_label(rgb, hues, tolerance=9, min_saturation=80):
    """Nearest calibrated hue per saturated pixel; -1 elsewhere."""
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV).astype(float)
    names = list(hues)
    ref = np.array([hues[n] for n in names], float)
    diff = np.abs(hsv[..., :1] - ref)
    diff = np.minimum(diff, 180 - diff)
    label = diff.argmin(-1)
    label[(diff.min(-1) > tolerance) | (hsv[..., 1] < min_saturation)] = -1
    return label, names


def _largest_component(mask, min_pixels):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    if count <= 1:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[index, cv2.CC_STAT_AREA] < min_pixels:
        return None
    return labels == index


def _footprint(points_xy):
    (cx, cy), (w, h), angle = cv2.minAreaRect(points_xy.astype(np.float32))
    if w < h:
        w, h, angle = h, w, angle + 90.
    yaw = np.deg2rad(angle)
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2  # long-axis direction, in (-pi/2, pi/2]
    return [float(cx), float(cy)], float(w), float(h), float(yaw)


def classify_kind(height, long_side, short_side):
    if long_side > .10 and short_side < .03 and height < .03:
        return "pen"
    if height > .055:
        return "mug"
    if long_side > .065:
        return "book"
    return "block"


def perceive(rgb, depth_m, K, world_from_camera, *, min_pixels=25, zone_min_pixels=150, masks_out=None):
    """masks_out (optional dict) receives each object's pixel mask and the back-projected point map."""
    xyz = back_project(depth_m, K, world_from_camera)
    if masks_out is not None:
        masks_out["_xyz"] = xyz
    valid = np.isfinite(xyz[..., 2])
    table_z = float(np.median(xyz[..., 2][valid]))
    height = xyz[..., 2] - table_z

    objects = {}
    labels, names = _palette_label(rgb, OBJECT_HUES)
    for i, color in enumerate(names):
        mask = _largest_component((labels == i) & valid & (height > .006), min_pixels)
        if mask is None:
            continue
        pts = xyz[mask]
        if masks_out is not None:
            masks_out[color] = mask
        top_z = float(np.quantile(pts[:, 2], .9))
        center_xy, long_side, short_side, yaw = _footprint(pts[:, :2])
        h = top_z - table_z
        kind = classify_kind(h, long_side, short_side)
        objects[f"{color}_{kind}"] = {
            "name": f"{color}_{kind}", "kind": kind, "color": color,
            "center": [round(center_xy[0], 4), round(center_xy[1], 4), round(table_z + h / 2, 4)],
            "top_z": round(top_z, 4), "height": round(h, 4),
            "footprint_long": round(long_side, 4), "footprint_short": round(short_side, 4),
            "long_axis_yaw": round(yaw, 3),
            "short_axis_yaw": round((yaw + np.pi) % np.pi - np.pi / 2, 3),  # perpendicular, in [-pi/2, pi/2)
            "visible_pixels": int(mask.sum())}

    zones = {}
    labels, names = _palette_label(rgb, ZONE_HUES)
    for i, color in enumerate(names):
        # All floor pixels of the colour (an object lying inside can split the floor into several parts).
        floor = cv2.morphologyEx(((labels == i) & valid & (np.abs(height) < .006)).astype(np.uint8), cv2.MORPH_OPEN,
                                 np.ones((2, 2), np.uint8)).astype(bool)
        if floor.sum() < zone_min_pixels:  # the smallest receptacle (pen case) shows ~340 floor pixels
            continue
        center_xy, long_side, short_side, yaw = _footprint(xyz[floor][:, :2])
        zones[ZONE_NAMES[color]] = {"name": ZONE_NAMES[color],
                                    "center": [round(center_xy[0], 4), round(center_xy[1], 4), round(table_z, 4)],
                                    "half_size": [round(long_side / 2, 4), round(short_side / 2, 4)],
                                    "long_axis_yaw": round(yaw, 3)}
    return {"table_z": round(table_z, 4), "objects": objects, "zones": zones}


def in_zone_frame(xy, zone):
    """(along the zone's long axis, across it) offset of a point from the zone centre."""
    yaw = zone.get("long_axis_yaw", 0.0)
    d = np.asarray(xy[:2], float) - np.asarray(zone["center"][:2], float)
    return float(d[0] * np.cos(yaw) + d[1] * np.sin(yaw)), float(-d[0] * np.sin(yaw) + d[1] * np.cos(yaw))


def inside_zone(xy, zone, margin=0.0):
    a, b = in_zone_frame(xy, zone)
    return abs(a) <= zone["half_size"][0] - margin and abs(b) <= zone["half_size"][1] - margin


def object_on_zone(obj, zone, table_z, margin=.01):
    """Perception-only check: object centre inside the zone (in the zone's own frame) and resting at its nominal
    height. A pen must lie inside with both ends."""
    inside = inside_zone(obj["center"], zone, margin)
    if obj["kind"] == "pen":
        u = np.array([np.cos(obj["long_axis_yaw"]), np.sin(obj["long_axis_yaw"])]) * obj["footprint_long"] / 2
        inside = inside and all(inside_zone(np.asarray(obj["center"][:2]) + sgn * u, zone, -.004) for sgn in (-1, 1))
    nominal = KIND_HEIGHT[obj["kind"]]
    resting = obj["top_z"] - table_z <= nominal + .012
    along, across = in_zone_frame(obj["center"], zone)
    out = {"inside_zone": bool(inside), "resting": bool(resting),
           "offset_from_zone_center": [round(obj["center"][0] - zone["center"][0], 4),
                                       round(obj["center"][1] - zone["center"][1], 4)],
           "offset_along_across_zone_axis": [round(along, 4), round(across, 4)],
           "zone_half_size": zone["half_size"], "top_height_above_table": round(obj["top_z"] - table_z, 4),
           "nominal_height": nominal}
    if obj["footprint_long"] - obj["footprint_short"] > .01 and "long_axis_yaw" in zone:
        diff = (obj["long_axis_yaw"] - zone["long_axis_yaw"] + np.pi / 2) % np.pi - np.pi / 2
        out["object_axis_vs_zone_axis_deg"] = round(float(np.degrees(abs(diff))), 1)
    return out

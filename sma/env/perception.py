"""Top-down RGB-D perception: colored objects above the table and colored zones on it.

Uses only the camera image, depth and calibration (available on a real calibrated camera).
Never reads simulator object state.
"""
import cv2
import numpy as np


WALL_THICKNESS = .008  # receptacle walls (assets.py: 4 mm half thickness)
KIND_HEIGHT = {"block": .04, "book": .025, "mug": .07, "pen": .012, "paper": .04, "notebook": .01}
ZONE_NAMES = {"orange": "orange_tray", "cyan": "cyan_tray", "purple": "purple_box",
              "green": "green_case", "magenta": "magenta_holder",
              "blue": "blue_coaster", "red": "red_bin"}  # receptacle catalogue (plus "book_pile", found from depth)


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
OBJECT_HUES = {"red": 1, "yellow": 30, "green": 62, "blue": 113,
               "cyan": 90, "purple": 144}  # cyan / purple objects (realdesk): the zone-base RGBA and its hue
ZONE_HUES = {"orange": 23, "cyan": 90, "purple": 144, "green": 62,  # green: pen case floor (checked by pen_gate)
             "magenta": 158,  # pen holder floor: rendered hue 157-158 (stand_gate calibration, 2026-11-04)
             "blue": 113, "red": 1}  # coaster / wastebasket floor (realdesk): the object blue / red


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


def object_pixels(color_mask, valid, height):
    """Pixels of one object colour above the table, without 1-pixel-wide lines: where two colours meet, the
    anti-aliased edge pixels can take a third hue (cyan next to orange reads as green), and the rim of a coloured
    floor next to a tall grey wall gets the wall's depth."""
    raw = (color_mask & valid & (height > .006)).astype(np.uint8)
    k = np.ones((2, 2), np.uint8)  # opening with matched anchors, so the mask does not shift by a pixel
    return cv2.dilate(cv2.erode(raw, k, anchor=(0, 0)), k, anchor=(1, 1)).astype(bool)


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


def classify_kind(height, long_side, short_side, fill=None, top_share=None):
    """top_share: share of the object's points within 2 mm of its top (90th percentile height). A toy block's flat
    top gives 0.85-0.99, a paper ball's dome 0.31-0.38, also when the ball is partly hidden by a wastebasket wall
    (scripts/shape_probe.py, 2026-11-05). fill (pixels over the footprint rectangle) does not separate them: blocks
    0.85-0.89, free balls 0.77-0.84, half-hidden balls 0.88-0.89; it is used only when no depth share is given."""
    if long_side > .15 and short_side > .09:  # an A5 notebook (21 x 14.8 cm), also when it lies on a pile of books
        return "notebook"
    if long_side > .10 and short_side < .03 and height < .03:
        return "pen"
    if height > .09 and long_side < .045:  # a pen standing on its end (e.g. in a holder): tall, tiny footprint
        return "pen"
    if height > .055:
        return "mug"
    if long_side > .065:
        return "book"
    if top_share is not None:
        return "paper" if top_share < .6 else "block"
    if fill is not None and fill < .82:
        return "paper"
    return "block"


def _fill_ratio(mask):
    """Pixels of the mask over the area of their minimum-area rectangle (in image pixels)."""
    rows, cols = np.nonzero(mask)
    if len(rows) < 5:
        return None
    _, (w, h), _ = cv2.minAreaRect(np.column_stack([cols, rows]).astype(np.float32))
    return float(len(rows) / max((w + 1) * (h + 1), 1.0))


def _rim_height(height, band, d, yaw, hx, hy):
    """Height of a receptacle's rim from the band of pixels just outside its floor: the 90th percentile on each of
    the four sides, then the median of the sides, so a taller neighbour next to one side (a wastebasket beside the
    pen holder) does not set the rim."""
    if band.sum() < 10:
        return None
    a = d[..., 0] * np.cos(yaw) + d[..., 1] * np.sin(yaw)
    b = -d[..., 0] * np.sin(yaw) + d[..., 1] * np.cos(yaw)
    across_a = np.abs(a) - hx > np.abs(b) - hy
    sides = [band & across_a & (a > 0), band & across_a & (a < 0), band & ~across_a & (b > 0), band & ~across_a & (b < 0)]
    tops = [float(np.quantile(height[m], .9)) for m in sides if m.sum() >= 5]
    return float(np.median(tops)) if tops else float(np.quantile(height[band], .9))


def perceive(rgb, depth_m, K, world_from_camera, *, min_pixels=8, zone_min_pixels=150, masks_out=None):
    # min_pixels: a pen standing in a holder shows only its 1.2 cm top and a sliver of its side (about 10-15 pixels
    # at 256 px; 15 missed one in the realdesk gate).
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
        mask = _largest_component(object_pixels(labels == i, valid, height), min_pixels)
        if mask is None:
            continue
        pts = xyz[mask]
        if masks_out is not None:
            masks_out[color] = mask
        top_z = float(np.quantile(pts[:, 2], .9))
        center_xy, long_side, short_side, yaw = _footprint(pts[:, :2])
        h = top_z - table_z
        top_share = float(np.mean(pts[:, 2] > top_z - .002))
        kind = classify_kind(h, long_side, short_side, _fill_ratio(mask), top_share)
        objects[f"{color}_{kind}"] = {
            "name": f"{color}_{kind}", "kind": kind, "color": color,
            "center": [round(center_xy[0], 4), round(center_xy[1], 4), round(table_z + h / 2, 4)],
            "top_z": round(top_z, 4), "height": round(h, 4),
            "footprint_long": round(long_side, 4), "footprint_short": round(short_side, 4),
            "long_axis_yaw": round(yaw, 3),
            "short_axis_yaw": round((yaw + np.pi) % np.pi - np.pi / 2, 3),  # perpendicular, in [-pi/2, pi/2)
            "visible_pixels": int(mask.sum())}

    zones = {}
    grey = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)[..., 1] < 80  # unsaturated
    labels, names = _palette_label(rgb, ZONE_HUES)
    for i, color in enumerate(names):
        # All floor pixels of the colour (an object lying inside can split the floor into several parts).
        floor = cv2.morphologyEx(((labels == i) & valid & (np.abs(height) < .006)).astype(np.uint8), cv2.MORPH_OPEN,
                                 np.ones((2, 2), np.uint8))
        # Drop small specks (e.g. the bottom edge of an object of the same colour elsewhere on the table): keep the
        # parts that are at least a quarter of the largest one.
        count, parts, stats, _ = cv2.connectedComponentsWithStats(floor, connectivity=4)
        if count <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        floor = np.isin(parts, 1 + np.nonzero(areas >= .25 * areas.max())[0])
        if floor.sum() < zone_min_pixels:  # the smallest receptacle (pen case) shows ~340 floor pixels
            continue
        center_xy, long_side, short_side, yaw = _footprint(xyz[floor][:, :2])
        # Rim / wall height: the highest points in a 1.5 cm band just outside the floor (from depth only).
        d = xyz[..., :2] - np.asarray(center_xy)
        a = np.abs(d[..., 0] * np.cos(yaw) + d[..., 1] * np.sin(yaw))
        b = np.abs(-d[..., 0] * np.sin(yaw) + d[..., 1] * np.cos(yaw))
        hx, hy = long_side / 2, short_side / 2
        band = valid & (a <= hx + .015) & (b <= hy + .015) & ~((a <= hx - .002) & (b <= hy - .002))
        rim = _rim_height(height, band, d, yaw, hx, hy)
        if rim is not None and rim > .06:
            # A tall receptacle (pen holder, wastebasket): its walls hide part of the floor from the camera (and so
            # do things standing or lying in it), so the floor's footprint is shifted and too small. The tops of the
            # walls are always in view: take the ring of grey wall-top pixels around the floor (a neighbour's walls
            # are a separate ring) and remove the wall thickness.
            ring = (valid & grey & (np.abs(height - rim) < .015) & (a <= hx + .05) & (b <= hy + .05)).astype(np.uint8)
            count, parts, stats, _ = cv2.connectedComponentsWithStats(ring, connectivity=8)
            rows, cols = np.nonzero(floor)
            r0, c0 = rows.mean(), cols.mean()
            around = [k for k in range(1, count) if stats[k, cv2.CC_STAT_AREA] >= 20
                      and stats[k, cv2.CC_STAT_LEFT] <= c0 <= stats[k, cv2.CC_STAT_LEFT] + stats[k, cv2.CC_STAT_WIDTH]
                      and stats[k, cv2.CC_STAT_TOP] <= r0 <= stats[k, cv2.CC_STAT_TOP] + stats[k, cv2.CC_STAT_HEIGHT]]
            if around:
                k = max(around, key=lambda k: stats[k, cv2.CC_STAT_AREA])
                center_xy, long_side, short_side, yaw = _footprint(xyz[parts == k][:, :2])
                long_side, short_side = long_side - 2 * WALL_THICKNESS, short_side - 2 * WALL_THICKNESS
        zones[ZONE_NAMES[color]] = {"name": ZONE_NAMES[color],
                                    "center": [round(center_xy[0], 4), round(center_xy[1], 4), round(table_z, 4)],
                                    "half_size": [round(long_side / 2, 4), round(short_side / 2, 4)],
                                    "long_axis_yaw": round(yaw, 3),
                                    "rim_height": None if rim is None else round(rim, 3)}
    pile = _book_pile(height, valid, grey, xyz, table_z)
    if pile is not None:
        zones["book_pile"] = pile
    return {"table_z": round(table_z, 4), "objects": objects, "zones": zones}


def _book_pile(height, valid, grey, xyz, table_z, min_pixels=600):
    """A pile of books: the largest flat grey top 5-25 cm above the table (a B5 textbook top is about 3300 pixels at
    256 px; a notebook lying on it still leaves a ring of the top in view). Its footprint is the pixels within 1.5 cm
    of the top's height (the books' sides are lower and do not widen it)."""
    mask = cv2.morphologyEx((grey & valid & (height > .05) & (height < .25)).astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8))
    part = _largest_component(mask, min_pixels)
    if part is None:
        return None
    top = float(np.median(height[part]))
    flat = part & (np.abs(height - top) < .015)
    if flat.sum() < min_pixels:
        return None
    center_xy, long_side, short_side, yaw = _footprint(xyz[flat][:, :2])
    return {"name": "book_pile", "center": [round(center_xy[0], 4), round(center_xy[1], 4), round(table_z + top, 4)],
            "half_size": [round(long_side / 2, 4), round(short_side / 2, 4)], "long_axis_yaw": round(yaw, 3),
            "rim_height": round(top, 3), "top_height": round(top, 3)}


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
    if obj["kind"] == "notebook" and zone.get("top_height") is None:  # in a tray: all four corners inside
        u = np.array([np.cos(obj["long_axis_yaw"]), np.sin(obj["long_axis_yaw"])])
        v = np.array([-u[1], u[0]])
        inside = inside and all(inside_zone(np.asarray(obj["center"][:2]) + a * obj["footprint_long"] / 2 * u +
                                            b * obj["footprint_short"] / 2 * v, zone, -.004) for a in (-1, 1) for b in (-1, 1))
    nominal = KIND_HEIGHT.get(obj["kind"]) or obj.get("height") or .02  # an unidentified thing: its own height
    # lying on the 4 mm floor, not on the 8 mm rim (a notebook partly on the rim is 4 mm higher)
    resting = obj["top_z"] - table_z <= nominal + (.007 if obj["kind"] == "notebook" else .012)
    if zone.get("top_height") is not None:  # on a pile of books: its centre over the pile, lying flat on the top
        inside = inside_zone(obj["center"], zone, .02)
        resting = abs(obj["top_z"] - table_z - zone["top_height"] - nominal) < .008
    elif zone.get("hollow"):  # in a hollow receptacle (colour-coded desk): down inside it, below its rim
        resting = obj["top_z"] - table_z < (zone.get("rim_height") or 0) + .01
    # A mug must also stand: one lying on its side is about 5 cm tall instead of 7 (a GPU realdesk run counted a
    # fallen mug on the coaster as done).
    upright = obj["kind"] != "mug" or obj["top_z"] - table_z >= nominal - .006  # 5.9-6.2 cm read as standing in the
    # GPU realdesk runs for mugs that ground truth found tipped; a standing mug measures 7.0 cm
    along, across = in_zone_frame(obj["center"], zone)
    out = {"inside_zone": bool(inside), "resting": bool(resting), "upright": bool(upright),
           "offset_from_zone_center": [round(obj["center"][0] - zone["center"][0], 4),
                                       round(obj["center"][1] - zone["center"][1], 4)],
           "offset_along_across_zone_axis": [round(along, 4), round(across, 4)],
           "zone_half_size": zone["half_size"], "top_height_above_table": round(obj["top_z"] - table_z, 4),
           "nominal_height": nominal}
    if obj["footprint_long"] - obj["footprint_short"] > .01 and "long_axis_yaw" in zone:
        diff = (obj["long_axis_yaw"] - zone["long_axis_yaw"] + np.pi / 2) % np.pi - np.pi / 2
        out["object_axis_vs_zone_axis_deg"] = round(float(np.degrees(abs(diff))), 1)
        out["object_long_axis_yaw"], out["zone_long_axis_yaw"] = obj["long_axis_yaw"], round(float(zone["long_axis_yaw"]), 3)
    return out


# ---------------------------------------------------------------- neutral perception (colour-coded desk)
# Every thing on the desk has its own colour. The camera reports each one under its colour with measurements only;
# what it is (and whether it is to be moved or is a place to put things) is left to whoever reads the report.

def perceive_things(rgb, depth_m, K, world_from_camera, hues, tolerance, *, min_pixels=8, masks_out=None):
    """{"table_z", "things": {colour: geometry}}: the largest region of each palette colour, floors and walls
    included (a coloured receptacle is one region from the floor up)."""
    xyz = back_project(depth_m, K, world_from_camera)
    if masks_out is not None:
        masks_out["_xyz"] = xyz
    valid = np.isfinite(xyz[..., 2])
    table_z = float(np.median(xyz[..., 2][valid]))
    height = xyz[..., 2] - table_z
    labels, names = _palette_label(rgb, hues, tolerance)
    things = {}
    for i, color in enumerate(names):
        raw = ((labels == i) & valid & (height > -.003)).astype(np.uint8)
        k = np.ones((2, 2), np.uint8)
        mask = _largest_component(cv2.dilate(cv2.erode(raw, k, anchor=(0, 0)), k, anchor=(1, 1)).astype(bool),
                                  min_pixels)
        if mask is None:
            continue
        if masks_out is not None:
            masks_out[color] = mask
        things[color] = describe_thing(color, mask, xyz, height, table_z)
    return {"table_z": round(table_z, 4), "things": things}


def describe_thing(color, mask, xyz, height, table_z):
    """Measurements of one thing: footprint, height of its top, outline shape (round / rectangular / irregular), whether
    it is hollow (its middle at least 2 cm below its rim: a cup, basket or holder seen from above) with the rim and
    inner floor heights, and whether its top is flat (most of it within 4 mm of the highest part) or domed."""
    pts, h = xyz[mask], height[mask]
    center_xy, long_side, short_side, yaw = _footprint(pts[:, :2])
    top = float(np.quantile(h, .95))
    filled = mask.astype(np.uint8).copy()
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)  # holes (what lies inside it) filled
    rows, cols = np.nonzero(filled)
    (_, _), (w, l), _ = cv2.minAreaRect(np.column_stack([cols, rows]).astype(np.float32))
    fill = float(filled.sum() / max((w + 1) * (l + 1), 1.0))
    ratio = long_side / max(short_side, 1e-3)
    shape = "round" if ratio < 1.25 and .68 < fill < .87 else "rectangular" if fill >= .87 else "irregular"
    d = xyz[..., :2] - np.asarray(center_xy)
    a = d[..., 0] * np.cos(yaw) + d[..., 1] * np.sin(yaw)
    b = -d[..., 0] * np.sin(yaw) + d[..., 1] * np.cos(yaw)
    inner = mask & (np.abs(a) < .3 * long_side) & (np.abs(b) < .3 * short_side)
    outer = mask & ((np.abs(a) > .35 * long_side) | (np.abs(b) > .35 * short_side))
    inner_h = float(np.median(height[inner])) if inner.sum() >= 4 else None
    rim = float(np.quantile(height[outer], .9)) if outer.sum() >= 4 else top
    hollow = inner_h is not None and rim - inner_h > .02
    out = {"name": color, "color": color,
           "center": [round(center_xy[0], 4), round(center_xy[1], 4), round(table_z + top / 2, 4)],
           "top_z": round(table_z + top, 4), "height": round(top, 4),
           "footprint_long": round(long_side, 4), "footprint_short": round(short_side, 4),
           "long_axis_yaw": round(yaw, 3), "short_axis_yaw": round((yaw + np.pi) % np.pi - np.pi / 2, 3),
           "shape": shape, "hollow": bool(hollow),
           "top_flat": bool(np.mean(h > top - .004) > .5), "visible_pixels": int(mask.sum())}
    if hollow:
        out.update({"rim_height": round(rim, 4), "inner_floor_height": round(inner_h, 4)})
    # The footprint of the top surface (or rim) alone: the sides of a tall thing seen in perspective widen the full one
    # (a coloured 11 cm pile of books read 1 cm wider on each side).
    upper = h > (rim if hollow else top) - .015
    if upper.sum() >= 5:
        (ux, uy), ul, us, uyaw = _footprint(pts[upper][:, :2])
        out["top_footprint"] = {"center": [round(ux, 4), round(uy, 4)], "long": round(ul, 4), "short": round(us, 4),
                                "long_axis_yaw": round(uyaw, 3)}
    return out

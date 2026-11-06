"""Class-agnostic perception: SAM2 masks from the segmentation service (/auto: a grid of point prompts, no text, no
categories) turned into things with depth geometry. No colour palette is involved: a thing is a region SAM2 separates
by its appearance, kept when the depth camera shows it standing on or rising above the desk.

When one mask lies inside another (or inside its outline, e.g. in a basket's hole), depth decides what it is: if it
rises at least 5 mm above the part of the other thing around it, it is a thing of its own lying in or on it (a paper
ball in a basket, a notebook on a pile, a pen standing in a holder); if not, it is a part of the other thing (the
inside of a mug, a basket's floor) and is dropped. Names are given by the tracker in the environment.

A grid of point prompts can miss small things (the top of a pen standing in a holder of the same colour is about 50
pixels here; a 16 x 16 grid, 32 pixels apart, missed it and a pen of the holder's colour laid across it, a 32 x 32 grid
finds both), and SAM2 can take two things for one (a mug on a coaster) or miss a thing that looks
like what it stands on (a white wastebasket on a white desk). Depth then points to what is left: parts of a thing rising
more than 2 cm above most of it (a pen standing out of a holder), parts rising more than 2 cm above its lowest level
that lie wholly inside it (a mug on a coaster; a hollow thing's walls reach its outline), and raised regions no thing
covers. SAM2 is prompted there; where it still gives no mask of its own, the raised region of the depth image itself
(with what it encloses) is taken as the thing: instance segmentation from depth alone.

A thing the robot has just let go of is looked for where the hand left it, at about the size it had (see
_expected_mask): laid on something it nearly covers, its mask and that of both together overlap too much to tell apart
from one thing at two granularities."""
import base64
import io
import json
import urllib.request

import cv2
import numpy as np
from PIL import Image

from .perception import back_project, describe_thing


def request_masks(url, rgb, points_per_side=16, iou_threshold=.8, points=None):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(buf, format="PNG")
    body = json.dumps({"image_png_b64": base64.b64encode(buf.getvalue()).decode(), "points_per_side": points_per_side,
                       "iou_threshold": iou_threshold, "points": points}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/auto", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        reply = json.loads(r.read())
    if "error" in reply:
        raise RuntimeError("segmentation service: " + reply["error"])
    return [(m["score"], np.asarray(Image.open(io.BytesIO(base64.b64decode(m["mask_png_b64"])))) > 127)
            for m in reply["masks"]]


def _filled(mask):
    filled = mask.astype(np.uint8).copy()
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    return filled > 0


def _rises(mask, other, height, by=.005):
    """Whether `mask` rises above `other` around it: its top (90th height percentile; a mug's median is its inner floor)
    at least `by` above the median height of `other` just around it."""
    return np.quantile(height[mask], .9) - _ring_height(mask, other, height) >= by


def _ring_height(mask, other, height, kernel=np.ones((7, 7), np.uint8)):
    """Median height of `other` just around `mask` (or of all of `other` when they do not touch)."""
    ring = (cv2.dilate(mask.astype(np.uint8), kernel) > 0) & ~mask & other
    return float(np.median(height[ring])) if ring.sum() >= 5 else float(np.median(height[other]))


def _side_of(part, thing, xyz, tol=.015):
    """Whether `part` is a side of `thing` seen in perspective (the page edges of a pile of books beside its cover, in
    another colour): it spans more than 2 cm in height and its points lie on the thing's outline on the desk (within tol
    of the hull of the thing's points); something standing or lying beside the thing lies outside that outline or is
    flat."""
    h = xyz[part][:, 2]
    if np.quantile(h, .9) - np.quantile(h, .1) < .02:
        return False
    hull = cv2.convexHull((xyz[thing][:, :2] * 1000).astype(np.float32))
    pts = xyz[part][:, :2] * 1000
    pts = pts[np.linspace(0, len(pts) - 1, min(len(pts), 200)).astype(int)]
    d = np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), True) for x, y in pts])
    return bool(np.mean(d >= -tol * 1000) > .8)


def _main_part(mask, gap=7, share=.3):
    """A mask without the specks SAM2 sometimes adds far from the thing (a few stray pixels stretch the footprint
    rectangle by many centimetres): its largest piece, with the pieces within `gap` pixels of it (about 1.5 cm) or at
    least `share` of its size (a thing cut in two by something lying across it)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    main = 1 + int(np.argmax(areas))
    near = cv2.dilate((labels == main).astype(np.uint8), np.ones((2 * gap + 1, 2 * gap + 1), np.uint8)) > 0
    keep = [k for k in range(1, n) if k == main or areas[k - 1] >= share * areas[main - 1] or near[labels == k].any()]
    return np.isin(labels, keep)


def segment_things(rgb, height, valid, masks, *, min_pixels=30, max_share=.15, expected=(), xyz=None):
    """Masks -> one mask per thing (see the module note). Returns a list of {"mask", "lab", "area"}. expected: masks
    of a thing the robot just let go of (see _expected_mask): never taken for another thing at another granularity,
    though depth must still show it rising above what it lies on. xyz: the points, to join a thing's sides seen in
    perspective to it (_side_of)."""
    total = height.size
    lab = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(float)
    cands = []
    for score, m in [*masks, *(("expected", m) for m in expected)]:
        depth_only = score is None  # a raised region of the depth image (see the module note)
        m = m & valid
        area = int(m.sum())
        if area < min_pixels or area > max_share * total:  # specks; the desk, the floor
            continue
        h = height[m]
        if np.quantile(h, .9) < .004:  # lying flat on the desk: texture, shadows, printing
            continue
        if np.median(h) > .2 or np.mean(h < -.05) > .5:  # the parked arm, a lamp head; the floor past the desk
            continue
        # Floor seen past the desk edge next to a thing that sticks out over it is no part of the thing (back-projected,
        # it lies far out and would stretch the footprint).
        m = _main_part(m & (height > -.03))
        area = int(m.sum())
        cands.append({"mask": m, "area": area, "score": score, "depth_only": depth_only})
    cands.sort(key=lambda c: -c["area"])
    kept = []
    for c in cands:
        drop = False
        for t in kept:
            inter = int((c["mask"] & t["mask"]).sum())
            if inter / (c["area"] + t["area"] - inter) > .6 and c["score"] != "expected":  # the same thing at
                # another granularity
                drop = True
                break
            inside = (c["mask"] & (t["mask"] | t["filled"])).sum() / c["area"]
            if inside > .8:  # within t: a part of it, or something in or on it
                # (a thing just let go of, found by its expected size, may miss a rim of its own pixels: measured
                # against t beyond that rim)
                other = t["mask"]
                if c["score"] == "expected":
                    beyond = other & ~(cv2.dilate(c["mask"].astype(np.uint8), np.ones((11, 11), np.uint8)) > 0)
                    other = beyond if beyond.sum() >= 5 else other
                if not _rises(c["mask"], other, height):
                    drop = True
                    break
            elif inter > .5 * c["area"] and c["score"] != "expected" and not _rises(c["mask"], t["mask"], height):
                # mostly on a thing kept already and not above it: another piece of that thing (the top of a pen
                # found by depth and the mask SAM2 gave when prompted there overlap by about half)
                t["mask"] = t["mask"] | c["mask"]
                t["area"] = int(t["mask"].sum())
                t["filled"] = _filled(t["mask"])
                drop = True
                break
        if not drop:
            c["filled"] = _filled(c["mask"])
            kept.append(c)
    for t in kept:  # a thing's own pixels: without what lies in or on it
        own = t["mask"].copy()
        for o in kept:
            if o is not t and (o["mask"] & t["filled"]).sum() > .8 * o["area"] and o["area"] < t["area"]:
                own &= ~o["mask"]
        t["mask"] = own if own.sum() >= min_pixels else t["mask"]
        t["area"] = int(t["mask"].sum())
        t["lab"] = lab[t["mask"]].mean(0)
    # A small part SAM2 split off beside a thing (a mug's handle): under a fifth of the size of a thing it touches, of
    # nearly the same colour or one of its sides seen in perspective (_side_of), and not rising above it, it joins that
    # thing; so does a raised region found by depth alone (a side face) under half the size of a thing it touches,
    # whatever its colour.
    joined = True
    while joined:
        joined = False
        for s in sorted(kept, key=lambda c: c["area"]):
            grown = cv2.dilate(s["mask"].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            for b in kept:
                if b is s or s["area"] > (.5 if s["depth_only"] else .2) * b["area"] or (grown & b["mask"]).sum() < 3:
                    continue
                d = s["lab"] - b["lab"]
                if (s["depth_only"] or np.sqrt((d[0] / 2) ** 2 + d[1] ** 2 + d[2] ** 2) < 12 or
                        (xyz is not None and _side_of(s["mask"], b["mask"], xyz))) and \
                        not _rises(s["mask"], b["mask"], height):
                    b["mask"] = b["mask"] | s["mask"]
                    b["area"] = int(b["mask"].sum())
                    b["lab"] = lab[b["mask"]].mean(0)
                    kept = [k for k in kept if k is not s]  # by identity: the dicts hold arrays
                    joined = True
                    break
            if joined:
                break
    return kept


def unexplained(kept, height, valid, min_pixels=15):
    """Raised regions of the depth image (see the module note), each a pixel mask with what it encloses: parts of a thing
    more than 2 cm above the level most of it reaches (its 80th height percentile); parts more than 2 cm above its 20th
    percentile; both only where they do not reach its outline (a hollow thing's walls do); pixels more than 6 mm above
    the desk that no thing covers. Regions are kept whole (a mug's 3 mm wall seen from above is 2 pixels wide)."""
    def parts(mask):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        return [labels == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_pixels]

    covered = np.zeros(height.shape, bool)
    regions = []
    for t in kept:
        m = t["mask"]
        covered |= m
        h = height[m]
        outline = m & ~(cv2.erode(_filled(m).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        for level in (np.quantile(h, .8), np.quantile(h, .2)):
            for p in parts(m & (height > level + .02)):
                if (p & outline).sum() < .02 * max(outline.sum(), 1):
                    regions.append(p)
    # Raised pixels more than 2 px from every thing (not the slivers along their edges), whole: a thin wall seen from
    # above (a wastebasket's) would break apart if opened.
    free = valid & ~(cv2.dilate(covered.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & (height > .006) & (height < .25)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=8)
    regions += [labels == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_pixels]
    return [_filled(r) & valid for r in regions]


def _inner_point(region):
    """The pixel deepest inside a region, as [x, y] for a point prompt."""
    dist = cv2.distanceTransform(np.pad(region.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    r, c = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return [int(c), int(r)]


def _expected_spot(expect, xyz, height, valid, camera_z):
    """Where a thing the robot just let go of should be seen and how big: expect holds where it should be (the hand's
    pose when it let go, with the thing's offset from the hand at the grasp), its size in the image on the desk and the
    height it was seen at. Returns (row, column, pixels) of the raised pixel nearest that place, None when nothing is
    raised within 3 cm of it."""
    raised = valid & (height > .004)
    dist = np.where(raised, np.hypot(xyz[..., 0] - expect["center"][0], xyz[..., 1] - expect["center"][1]), np.inf)
    v, u = np.unravel_index(int(np.argmin(dist)), dist.shape)
    if dist[v, u] > .03:
        return None
    # an image of a thing grows as the inverse square of its distance from the camera
    return v, u, expect["pixels"] * ((camera_z - expect["seen_z"]) / (camera_z - xyz[v, u, 2])) ** 2


def _expected_mask(kept, spot, height, valid, rgb, url, masks):
    """The mask of a thing the robot just let go of (spot: see _expected_spot), when SAM2 took it together with what it
    lies on: a notebook laid on a pile covers most of the pile's top, so its own mask and the mask of both overlap like
    one thing at two granularities. Where the thing found at the spot is much bigger than the expected size, the mask
    containing the spot that best matches the size (from the masks already found, or SAM2 prompted there; failing
    those, the part at the spot's height) is returned; None if that thing already has about the expected size, or
    nothing fits."""
    v, u, size = spot
    owner = next((t for t in kept if t["mask"][v, u]), None)
    if owner is not None and owner["area"] < 1.25 * size:
        return None
    pool = [m for _, m in masks if m[v, u]] + [m for _, m in request_masks(url, rgb, iou_threshold=.7, points=[[int(u), int(v)]])]
    fits = []
    for m in pool:
        m = m & valid & (height > -.03)
        area = int(m.sum())
        if .75 * size < area < 1.33 * size and (owner is None or (
                area < .9 * owner["area"] and (m & owner["mask"]).sum() > .8 * area)):
            fits.append((abs(np.log(area / size)), m))
    if fits:
        return min(fits, key=lambda f: f[0])[1]
    if owner is None:
        return None
    # No mask of that size: the part of the thing found there at the height of the point (the top of a notebook lies
    # a centimetre above the pile's top), with what it encloses
    level = (owner["mask"] & (np.abs(height - height[v, u]) < .004)).astype(np.uint8)
    labels = cv2.connectedComponents(level)[1]
    part = _filled(labels == labels[v, u]) & owner["mask"]
    area = int(part.sum())
    return part if .75 * size < area < 1.33 * size and area < .9 * owner["area"] else None


def perceive_sam(rgb, depth_m, K, world_from_camera, url, *, masks_out=None, exclude=None, expect=None):
    """{"table_z", "things": [geometry + "lab" (mean CIELAB colour, for tracking) + "mask"]} (unnamed). exclude: pixels
    that belong to no thing (the robot itself). expect: a thing the robot just let go of (see _expected_spot); the thing
    seen where it should be, at about its size, is marked "released_here"."""
    xyz = back_project(depth_m, K, world_from_camera)
    if masks_out is not None:
        masks_out["_xyz"] = xyz
    valid = np.isfinite(xyz[..., 2])
    if exclude is not None:
        valid = valid & ~exclude
    table_z = float(np.median(xyz[..., 2][valid]))  # the desk fills most of the view
    height = xyz[..., 2] - table_z
    masks = request_masks(url, rgb, points_per_side=32)
    kept = segment_things(rgb, height, valid, masks, xyz=xyz)
    regions = unexplained(kept, height, valid)
    if regions:  # a second look where depth says something is missing; the regions themselves as a last resort
        points = [_inner_point(r) for r in regions]
        masks = masks + request_masks(url, rgb, iou_threshold=.7, points=points) + [(None, r) for r in regions]
        kept = segment_things(rgb, height, valid, masks, xyz=xyz)
    spot = expect and _expected_spot(expect, xyz, height, valid, float(world_from_camera[2, 3]))
    here = None
    if spot:
        mask = _expected_mask(kept, spot, height, valid, rgb, url, masks)
        if mask is not None:
            kept = segment_things(rgb, height, valid, masks, expected=[mask], xyz=xyz)
        # The thing just put down: the one whose outline (convex: SAM2 may leave a mug's rim open where it split off the
        # inside) holds the spot and that has the expected size, taken together with the smaller pieces SAM2 split off
        # mostly within that outline (the inside of a mug, reaching its rim): nothing lies in a thing just put down. A
        # receptacle holding the spot (the pile under a notebook, a coaster) is much bigger than the thing put on it.
        v, u, size = spot
        best = None
        for t in kept:
            outline = np.zeros(t["mask"].shape, np.uint8)
            cv2.fillConvexPoly(outline, cv2.convexHull(np.column_stack(np.nonzero(t["mask"])[::-1]).astype(np.int32)), 1)
            outline = outline > 0
            if not outline[v, u]:
                continue
            parts = [o for o in kept if o is not t and o["area"] < t["area"] and (o["mask"] & outline).sum() > .5 * o["area"]]
            area = t["area"] + sum(o["area"] for o in parts)
            if .75 * size < area < 1.33 * size and (best is None or abs(np.log(area / size)) < best[0]):
                best = abs(np.log(area / size)), t, parts
        if best is not None:
            _, here, parts = best
            if parts:
                for o in parts:
                    here["mask"] = here["mask"] | o["mask"]
                kept = [t for t in kept if not any(t is o for o in parts)]
                here["area"] = int(here["mask"].sum())
                here["lab"] = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(float)[here["mask"]].mean(0)
    things = []
    for t in kept:
        d = describe_thing("?", t["mask"], xyz, height, table_z)
        d.update({"lab": [round(float(v), 1) for v in t["lab"]], "mask": t["mask"]})
        if t is here:
            d["released_here"] = True
        things.append(d)
    return {"table_z": round(table_z, 4), "things": things}

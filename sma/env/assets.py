"""Procedural MJCF for the realistic objects (mug with handle, book with covers and pages, pen) and the
static receptacles (shallow trays, a walled toy box and a narrow pen case).

Colors: object bodies use the object palette; receptacle *bases* use the receptacle palette while
rims and walls are an unsaturated wood tone, so perception never mistakes a rim for an object.

look "real" (sdesk): the same collision geoms, invisible, with visual-only geoms that look like the real things
(looks.py); the physics is unchanged.
"""
import math
import os
import tempfile
from pathlib import Path

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import new_geom

WOOD = [0.80, 0.79, 0.77, 1]  # neutral light grey: unsaturated, so never segmented as a coloured object
PAGES = [0.93, 0.91, 0.84, 1]
CACHE = Path(tempfile.gettempdir()) / "sma_assets"

# Half extents used by the scene sampler, the evaluator and perception.
BOOK_HALF = (0.0375, 0.0275, 0.0125)        # 7.5 x 5.5 x 2.5 cm pocket notebook (fits the 8 cm gripper either way)
NOTEBOOK_HALF = (0.105, 0.074, 0.005)        # A5 notebook, 21 x 14.8 x 1 cm, 160 g: wider than the 8 cm gripper
                                             # opens either way, so it is picked by pinching an edge that sticks out
MUG_RADIUS, MUG_HALF_HEIGHT = 0.024, 0.035  # 4.8 cm wide, 7 cm tall (+ handle)
BLOCK_HALF = 0.02                            # 4 cm toy block
PEN_HALF = (0.07, 0.006, 0.006)              # 14 cm pen with a square 1.2 cm section (does not roll, like a hex pencil)
PAPER_RADIUS = 0.022                         # 4.4 cm crumpled paper ball (light)
BIN_RADIUS, BIN_BOTTOM_RADIUS, BIN_HALF_HEIGHT = 0.085, 0.07, 0.10  # round wire-mesh wastebasket, inside: 17 cm
                                                                  # across the mouth, 14 cm at the floor; 20 cm tall


def _geom(name, type_, size, pos, rgba, mass, quat=None, solref="0.01 1"):
    q = f' quat="{" ".join(f"{v:.6f}" for v in quat)}"' if quat else ""
    return (f'<geom name="{name}" type="{type_}" size="{" ".join(f"{v:.5f}" for v in size)}" '
            f'pos="{" ".join(f"{v:.5f}" for v in pos)}" rgba="{" ".join(str(v) for v in rgba)}" mass="{mass}" '
            f'group="0" friction="1.5 0.005 0.0001" solref="{solref}" condim="4"{q}/>')


def _xml(model, geoms, half_height, radius, assets=""):
    return (f'<mujoco model="{model}"><asset>{assets}</asset><worldbody><body><body name="object">{"".join(geoms)}</body>'
            f'<site name="bottom_site" pos="0 0 {-half_height}" size="0.001" rgba="0 0 0 0"/>'
            f'<site name="top_site" pos="0 0 {half_height}" size="0.001" rgba="0 0 0 0"/>'
            f'<site name="horizontal_radius_site" pos="{radius} 0 0" size="0.001" rgba="0 0 0 0"/>'
            f'</body></worldbody></mujoco>')


def _real(model, geoms, half_height, radius, visual):
    """The collision geoms made invisible plus the visual-only geoms and assets of the real look."""
    import re
    geoms = [re.sub(r'rgba="[^"]*"', 'rgba="0 0 0 0"', g) for g in geoms]
    vis, assets = visual
    return _xml(model, geoms + vis, half_height, radius, assets)


def mug_xml(rgba, segments=20, wall=0.003, look="plain"):
    r, h = MUG_RADIUS, MUG_HALF_HEIGHT
    geoms = [_geom("bottom", "cylinder", [r, .003], [0, 0, -h + .003], rgba, .06)]
    chord = 2 * r * math.sin(math.pi / segments) * 1.05
    for i in range(segments):
        a = 2 * math.pi * i / segments
        geoms.append(_geom(f"wall_{i}", "box", [wall / 2, chord / 2, h], [(r - wall / 2) * math.cos(a),
                     (r - wall / 2) * math.sin(a), 0], rgba, .012, quat=[math.cos(a / 2), 0, 0, math.sin(a / 2)]))
    # Handle on the +x side: two arms and an outer bar.
    for z in (h * .55, -h * .45):
        geoms.append(_geom(f"handle_arm_{'top' if z > 0 else 'low'}", "box", [.007, .004, .003],
                           [r + .006, 0, z], rgba, .004))
    geoms.append(_geom("handle_bar", "box", [.003, .004, h * .5], [r + .012, 0, h * .05], rgba, .006))
    if look == "real":
        from .looks import mug_visual
        return _real("mug", geoms, h, r + .015, mug_visual(rgba))
    return _xml("mug", geoms, h, r + .015)


def book_xml(rgba):
    hx, hy, hz = BOOK_HALF
    cover = .0016
    geoms = [_geom("cover_top", "box", [hx, hy, cover], [0, 0, hz - cover], rgba, .03),
             _geom("cover_bottom", "box", [hx, hy, cover], [0, 0, -hz + cover], rgba, .03),
             _geom("spine", "box", [.0016, hy, hz], [-hx + .0016, 0, 0], rgba, .01),
             _geom("pages", "box", [hx - .004, hy - .002, hz - 2 * cover], [.0008, 0, 0], PAGES, .12)]
    return _xml("book", geoms, hz, math.hypot(hx, hy))


def notebook_xml(rgba, look="plain"):
    hx, hy, hz = NOTEBOOK_HALF
    cover = .0008
    geoms = [_geom("cover_top", "box", [hx, hy, cover], [0, 0, hz - cover], rgba, .025),
             _geom("cover_bottom", "box", [hx, hy, cover], [0, 0, -hz + cover], rgba, .025),
             _geom("spine", "box", [.0012, hy, hz], [-hx + .0012, 0, 0], rgba, .006),
             _geom("pages", "box", [hx - .003, hy - .002, hz - 2 * cover], [.0006, 0, 0], PAGES, .104)]
    if look == "real":
        from .looks import notebook_visual
        return _real("notebook", geoms, hz, math.hypot(hx, hy), notebook_visual(rgba))
    return _xml("notebook", geoms, hz, math.hypot(hx, hy))


def block_xml(rgba):
    s = BLOCK_HALF
    geoms = [_geom("body", "box", [s, s, s], [0, 0, 0], rgba, .05)]
    return _xml("block", geoms, s, s * math.sqrt(2))


def pen_xml(rgba, look="plain"):
    hx, hy, hz = PEN_HALF
    geoms = [_geom("body", "box", [hx, hy, hz], [0, 0, 0], rgba, .02)]
    if look == "real":
        from .looks import pen_visual
        return _real("pen", geoms, hz, hx, pen_visual(rgba))
    return _xml("pen", geoms, hz, hx)


def paper_xml(rgba):
    r = PAPER_RADIUS
    # A slightly squashed sphere plus two lumps, so it looks crumpled; light, with high friction so it does not roll far.
    geoms = [_geom("body", "ellipsoid", [r, r * .95, r * .9], [0, 0, 0], rgba, .006),
             _geom("lump_a", "sphere", [r * .45], [r * .55, r * .3, r * .35], rgba, .001),
             _geom("lump_b", "sphere", [r * .4], [-r * .45, -r * .4, r * .3], rgba, .001)]
    return _xml("paper", geoms, r * .9, r)


def crumpled_paper_xml(rgba, look="plain"):
    """A crumpled paper ball that looks crumpled (cdesk, where the model must recognise it from the camera): the same
    4.4 cm light body, wrapped in a dozen small flattened lumps lying along its surface, so the outline is lumpy and
    wrinkled (thin box facets looked like a toy block to the model in 6 of 16 views)."""
    import random
    r = PAPER_RADIUS
    rnd = random.Random(7)
    geoms = [_geom("body", "ellipsoid", [r * .88, r * .84, r * .8], [0, 0, 0], rgba, .005)]
    for i in range(14):
        theta, phi = rnd.uniform(0, 2 * math.pi), rnd.uniform(-.8, 1.3)
        n = [math.cos(phi) * math.cos(theta), math.cos(phi) * math.sin(theta), math.sin(phi)]  # outward normal
        pos = [r * .78 * n[0], r * .78 * n[1], r * .74 * n[2]]
        # a lump flattened along the normal: rotate the ellipsoid's z axis onto n, plus a random twist about it
        axis = [-n[1], n[0], 0.0]
        norm = math.hypot(axis[0], axis[1]) or 1.0
        angle = math.acos(max(-1.0, min(1.0, n[2])))
        twist = rnd.uniform(0, math.pi)
        q1 = [math.cos(angle / 2)] + [a / norm * math.sin(angle / 2) for a in axis]
        q2 = [math.cos(twist / 2), 0.0, 0.0, math.sin(twist / 2)]
        quat = [q1[0] * q2[0] - q1[3] * q2[3], q1[1] * q2[0] + q1[2] * q2[3], q1[2] * q2[0] - q1[1] * q2[3],
                q1[3] * q2[0] + q1[0] * q2[3]]
        geoms.append(_geom(f"lump_{i}", "ellipsoid", [r * rnd.uniform(.3, .45), r * rnd.uniform(.2, .32), r * .14], pos,
                           rgba, .0002, quat=quat))
    if look == "real":
        from .looks import paper_visual
        return _real("paper", geoms, r * .85, r, paper_visual(rgba))
    return _xml("paper", geoms, r * .85, r)


REAL_KINDS = ("mug", "pen", "paper", "notebook")  # kinds with a real look (the paper ball: crumpled)


def make_object(kind, name, rgba, variant=None, look="plain"):
    CACHE.mkdir(parents=True, exist_ok=True)
    builders = {"mug": mug_xml, "book": book_xml, "block": block_xml, "pen": pen_xml, "paper": paper_xml,
                "notebook": notebook_xml}
    if variant == "crumpled" and kind == "paper" or look == "real" and kind == "paper":
        builders["paper"] = crumpled_paper_xml
    real = look == "real" and kind in REAL_KINDS
    text = builders[kind](rgba, look="real") if real else builders[kind](rgba)
    path = CACHE / f"{kind}{'_' + variant if variant else ''}{'_real' if real else ''}_{'_'.join(f'{v:.2f}' for v in rgba[:3])}.xml"
    if not path.exists() or path.read_text() != text:
        tmp = path.with_suffix(f".{os.getpid()}.xml")
        tmp.write_text(text)
        os.replace(tmp, path)
    # the real look brings its own visual geoms: the collision geoms are not copied as visuals
    return MujocoXMLObject(str(path), name=name, joints=[{"type": "free", "damping": "0.0005"}],
                           obj_type="all", duplicate_collision_geoms=not real)


def _static(worldbody, name, size, pos, rgba, friction, yaw=0.0, visual=True, quat=None):
    # Robosuite renders only the visual group (1); collision geoms (group 0) are invisible.
    quat = quat or [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    worldbody.append(new_geom(name=name, type="box", size=size, pos=pos, quat=quat, rgba=[0, 0, 0, 0], group=0,
                              friction=friction))
    if visual:  # (the real look draws its own)
        worldbody.append(new_geom(name=name + "_visual", type="box", size=size, pos=pos, quat=quat, rgba=rgba, group=1,
                                  contype="0", conaffinity="0"))


def add_receptacle(worldbody, name, style, base_rgba, xy, table_z, half=(0.065, 0.065), yaw=0.0,
                   wall_half_height=None, wall_rgba=None, look="plain", asset=None):
    """Static receptacle: a coloured base plate plus wooden rims (tray) or walls (toy box, pen case), rotated by
    yaw about its centre. half: inner half extents along the receptacle's own x and y axes."""
    hx, hy = (half, half) if isinstance(half, (int, float)) else half
    c, s = math.cos(yaw), math.sin(yaw)

    def at(dx, dy, z):
        return [xy[0] + c * dx - s * dy, xy[1] + s * dx + c * dy, z]

    base_h = .002
    real = look == "real" and style in ("holder", "coaster")
    _static(worldbody, f"{name}_base", [hx, hy, base_h], at(0, 0, table_z + base_h), base_rgba,
            [1.5, .005, .0001], yaw, visual=not real)
    wall_half_h = wall_half_height or (.004 if style == "tray" else .025)
    t = .004
    for i, (dx, dy, sx, sy) in enumerate(((hx + t, 0, t, hy + 2 * t), (-hx - t, 0, t, hy + 2 * t),
                                          (0, hy + t, hx, t), (0, -hy - t, hx, t))):
        _static(worldbody, f"{name}_wall_{i}", [sx, sy, wall_half_h], at(dx, dy, table_z + wall_half_h),
                wall_rgba or WOOD, [1.0, .005, .0001], yaw, visual=not real)
    if real:
        from .looks import coaster_look, holder_look
        (holder_look if style == "holder" else coaster_look)(worldbody, asset, name, base_rgba, xy, table_z, (hx, hy),
                                                             wall_half_h, yaw)
    return table_z + 2 * base_h  # surface height inside the receptacle


def add_bin(worldbody, name, base_rgba, xy, table_z, radius=BIN_RADIUS, half_height=BIN_HALF_HEIGHT,
            bottom_radius=BIN_BOTTOM_RADIUS, segments=24, wall_rgba=None, look="plain", asset=None):
    """Round wastebasket, wider at the mouth (inner radius `radius`) than at the floor (`bottom_radius`): a coloured
    floor disc and a ring of thin wall panels leaning out (open top)."""
    real = look == "real"
    worldbody.append(new_geom(name=f"{name}_base", type="cylinder", size=[bottom_radius, .002],
                              pos=[xy[0], xy[1], table_z + .002], rgba=[0, 0, 0, 0], group=0, friction=[1.5, .005, .0001]))
    if not real:
        worldbody.append(new_geom(name=f"{name}_base_visual", type="cylinder", size=[bottom_radius, .002],
                                  pos=[xy[0], xy[1], table_z + .002], rgba=base_rgba, group=1, contype="0",
                                  conaffinity="0"))
    t = .004
    lean = math.atan2(radius - bottom_radius, 2 * half_height)
    chord = 2 * (radius + t) * math.sin(math.pi / segments) * 1.08  # wide enough at the mouth (they overlap lower down)
    mid = (radius + bottom_radius) / 2 + t
    cb, sb = math.cos(lean / 2), math.sin(lean / 2)
    for i in range(segments):
        a = 2 * math.pi * i / segments
        ca, sa = math.cos(a / 2), math.sin(a / 2)
        _static(worldbody, f"{name}_wall_{i}", [t, chord / 2, half_height / math.cos(lean)],
                [xy[0] + mid * math.cos(a), xy[1] + mid * math.sin(a), table_z + half_height],
                wall_rgba or WOOD, [1.0, .005, .0001], visual=not real,
                quat=[ca * cb, -sa * sb, ca * sb, sa * cb])  # turned to face out by a, its top tipped out by lean
    if real:
        from .looks import bin_look
        bin_look(worldbody, asset, name, base_rgba, xy, table_z, radius, half_height, bottom_radius)
    return table_z + .004


def add_decor(worldbody, table_z, books=True, look="plain", asset=None, phone_side=1):
    """Static clutter on the desk, out of the arm's reach (x > 0.23 or at the far corners): a closed laptop, a stack of
    books, a desk lamp and a phone. Low and unsaturated (greys, beige), so perception never takes them for objects
    and they do not hide the workspace from the demo camera (which looks from the far side of the table).
    phone_side: +1 the phone at the left near corner (+y), -1 at the right near corner (a varied desk whose notebook
    lies at the left edge)."""
    fr = [1, .005, .0001]
    real = look == "real"
    _static(worldbody, "decor_laptop", [.075, .11, .008], [.315, .03, table_z + .008], [.30, .30, .32, 1], fr,
            visual=not real)
    _static(worldbody, "decor_laptop_logo", [.012, .012, .0005], [.315, .03, table_z + .0165], [.75, .75, .77, 1], fr,
            visual=not real)
    for i, (w, d) in enumerate(((.075, .055), (.07, .05), (.08, .058)) if books else ()):
        _static(worldbody, f"decor_books_{i}", [w, d, .011], [.31, -.30, table_z + .011 + .022 * i],
                [.62 - .1 * i, .58 - .09 * i, .52 - .07 * i, 1], fr, .25 * i - .2)
    _static(worldbody, "decor_lamp_base", [.045, .045, .008], [.32, .31, table_z + .008], [.25, .25, .27, 1], fr)
    _static(worldbody, "decor_lamp_pole", [.006, .006, .14], [.32, .31, table_z + .156], [.6, .6, .62, 1], fr)
    _static(worldbody, "decor_lamp_head", [.06, .025, .015], [.28, .31, table_z + .30], [.25, .25, .27, 1], fr)
    _static(worldbody, "decor_phone", [.035, .07, .004], [-.31, .32 * phone_side, table_z + .004], [.12, .12, .14, 1], fr,
            .3 * phone_side, visual=not real)
    if real:
        from .looks import decor_look
        decor_look(worldbody, asset, table_z, phone_side)


def add_stack(worldbody, name, xy, table_z, height, half=(.125, .09), yaw=0.0, covers=None, look="plain", asset=None,
              paint=None):
    """A pile of closed books (static; default: B5 textbooks, 25 x 18 cm) in dull unsaturated covers, e.g. the pile a
    notebook goes on. Returns the height of its top above the table."""
    covers = covers or ([.55, .52, .47, 1], [.40, .42, .45, 1], [.62, .60, .55, 1], [.35, .33, .31, 1])
    n = max(1, round(height / .028))
    t = height / n
    c, s = math.cos(yaw), math.sin(yaw)
    books = []
    for i in range(n):
        dx, dy = .004 * ((i % 2) * 2 - 1), .003 * (1 - (i % 3))  # slightly uneven, like a real pile
        size = [half[0] - .004 * (i % 2), half[1] - .003 * (i % 3), t / 2]
        centre = [xy[0] + c * dx - s * dy, xy[1] + s * dx + c * dy, table_z + t * (i + .5)]
        _static(worldbody, f"{name}_{i}", size, centre, covers[i % 4], [1.0, .005, .0001], yaw + .02 * ((i % 3) - 1),
                visual=look != "real")
        books.append((size, centre, yaw + .02 * ((i % 3) - 1)))
    if look == "real":
        from .looks import stack_look
        stack_look(worldbody, asset, name, books, paint)
    return height

"""Procedural MJCF for the realistic objects (mug with handle, book with covers and pages) and the
static receptacles (shallow trays and a walled toy box).

Colors: object bodies use the object palette; receptacle *bases* use the receptacle palette while
rims and walls are an unsaturated wood tone, so perception never mistakes a rim for an object.
"""
import math
import tempfile
from pathlib import Path

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import new_geom

WOOD = [0.74, 0.66, 0.56, 1]
PAGES = [0.93, 0.91, 0.84, 1]
CACHE = Path(tempfile.gettempdir()) / "sma_assets"

# Half extents used by the scene sampler, the evaluator and perception.
BOOK_HALF = (0.045, 0.030, 0.0125)          # 9 x 6 x 2.5 cm
MUG_RADIUS, MUG_HALF_HEIGHT = 0.024, 0.035  # 4.8 cm wide, 7 cm tall (+ handle)
BLOCK_HALF = 0.02                            # 4 cm toy block


def _geom(name, type_, size, pos, rgba, mass, quat=None, solref="0.01 1"):
    q = f' quat="{" ".join(f"{v:.6f}" for v in quat)}"' if quat else ""
    return (f'<geom name="{name}" type="{type_}" size="{" ".join(f"{v:.5f}" for v in size)}" '
            f'pos="{" ".join(f"{v:.5f}" for v in pos)}" rgba="{" ".join(str(v) for v in rgba)}" mass="{mass}" '
            f'group="0" friction="1.5 0.005 0.0001" solref="{solref}" condim="4"{q}/>')


def _xml(model, geoms, half_height, radius):
    return (f'<mujoco model="{model}"><asset/><worldbody><body><body name="object">{"".join(geoms)}</body>'
            f'<site name="bottom_site" pos="0 0 {-half_height}" size="0.001" rgba="0 0 0 0"/>'
            f'<site name="top_site" pos="0 0 {half_height}" size="0.001" rgba="0 0 0 0"/>'
            f'<site name="horizontal_radius_site" pos="{radius} 0 0" size="0.001" rgba="0 0 0 0"/>'
            f'</body></worldbody></mujoco>')


def mug_xml(rgba, segments=20, wall=0.003):
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
    return _xml("mug", geoms, h, r + .015)


def book_xml(rgba):
    hx, hy, hz = BOOK_HALF
    cover = .0016
    geoms = [_geom("cover_top", "box", [hx, hy, cover], [0, 0, hz - cover], rgba, .03),
             _geom("cover_bottom", "box", [hx, hy, cover], [0, 0, -hz + cover], rgba, .03),
             _geom("spine", "box", [.0016, hy, hz], [-hx + .0016, 0, 0], rgba, .01),
             _geom("pages", "box", [hx - .004, hy - .002, hz - 2 * cover], [.0008, 0, 0], PAGES, .12)]
    return _xml("book", geoms, hz, math.hypot(hx, hy))


def block_xml(rgba):
    s = BLOCK_HALF
    geoms = [_geom("body", "box", [s, s, s], [0, 0, 0], rgba, .05)]
    return _xml("block", geoms, s, s * math.sqrt(2))


def make_object(kind, name, rgba):
    CACHE.mkdir(parents=True, exist_ok=True)
    text = {"mug": mug_xml, "book": book_xml, "block": block_xml}[kind](rgba)
    path = CACHE / f"{kind}_{'_'.join(f'{v:.2f}' for v in rgba[:3])}.xml"
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return MujocoXMLObject(str(path), name=name, joints=[{"type": "free", "damping": "0.0005"}],
                           obj_type="all", duplicate_collision_geoms=True)


def _static(worldbody, name, size, pos, rgba, friction):
    # Robosuite renders only the visual group (1); collision geoms (group 0) are invisible.
    worldbody.append(new_geom(name=name, type="box", size=size, pos=pos, rgba=[0, 0, 0, 0], group=0,
                              friction=friction))
    worldbody.append(new_geom(name=name + "_visual", type="box", size=size, pos=pos, rgba=rgba, group=1,
                              contype="0", conaffinity="0"))


def add_receptacle(worldbody, name, style, base_rgba, xy, table_z, half=0.065):
    """Static receptacle: a coloured base plate plus wooden rims (tray) or walls (toy box)."""
    base_h = .002
    _static(worldbody, f"{name}_base", [half, half, base_h], [xy[0], xy[1], table_z + base_h], base_rgba,
            [1.5, .005, .0001])
    wall_half_h = .009 if style == "tray" else .025
    t = .004
    for i, (dx, dy, sx, sy) in enumerate(((half + t, 0, t, half + 2 * t), (-half - t, 0, t, half + 2 * t),
                                          (0, half + t, half, t), (0, -half - t, half, t))):
        _static(worldbody, f"{name}_wall_{i}", [sx, sy, wall_half_h],
                [xy[0] + dx, xy[1] + dy, table_z + wall_half_h], WOOD, [1.0, .005, .0001])
    return table_z + 2 * base_h  # surface height inside the receptacle

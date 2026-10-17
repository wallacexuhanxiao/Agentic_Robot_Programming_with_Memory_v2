"""Procedural MJCF for the realistic objects (mug with handle, book with covers and pages, pen) and the
static receptacles (shallow trays, a walled toy box and a narrow pen case).

Colors: object bodies use the object palette; receptacle *bases* use the receptacle palette while
rims and walls are an unsaturated wood tone, so perception never mistakes a rim for an object.
"""
import math
import tempfile
from pathlib import Path

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import new_geom

WOOD = [0.80, 0.79, 0.77, 1]  # neutral light grey: unsaturated, so never segmented as a coloured object
PAGES = [0.93, 0.91, 0.84, 1]
CACHE = Path(tempfile.gettempdir()) / "sma_assets"

# Half extents used by the scene sampler, the evaluator and perception.
BOOK_HALF = (0.0375, 0.0275, 0.0125)        # 7.5 x 5.5 x 2.5 cm pocket notebook (fits the 8 cm gripper either way)
MUG_RADIUS, MUG_HALF_HEIGHT = 0.024, 0.035  # 4.8 cm wide, 7 cm tall (+ handle)
BLOCK_HALF = 0.02                            # 4 cm toy block
PEN_HALF = (0.07, 0.006, 0.006)              # 14 cm pen with a square 1.2 cm section (does not roll, like a hex pencil)


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


def pen_xml(rgba):
    hx, hy, hz = PEN_HALF
    geoms = [_geom("body", "box", [hx, hy, hz], [0, 0, 0], rgba, .02)]
    return _xml("pen", geoms, hz, hx)


def make_object(kind, name, rgba):
    CACHE.mkdir(parents=True, exist_ok=True)
    text = {"mug": mug_xml, "book": book_xml, "block": block_xml, "pen": pen_xml}[kind](rgba)
    path = CACHE / f"{kind}_{'_'.join(f'{v:.2f}' for v in rgba[:3])}.xml"
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return MujocoXMLObject(str(path), name=name, joints=[{"type": "free", "damping": "0.0005"}],
                           obj_type="all", duplicate_collision_geoms=True)


def _static(worldbody, name, size, pos, rgba, friction, yaw=0.0):
    # Robosuite renders only the visual group (1); collision geoms (group 0) are invisible.
    quat = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    worldbody.append(new_geom(name=name, type="box", size=size, pos=pos, quat=quat, rgba=[0, 0, 0, 0], group=0,
                              friction=friction))
    worldbody.append(new_geom(name=name + "_visual", type="box", size=size, pos=pos, quat=quat, rgba=rgba, group=1,
                              contype="0", conaffinity="0"))


def add_receptacle(worldbody, name, style, base_rgba, xy, table_z, half=(0.065, 0.065), yaw=0.0,
                   wall_half_height=None):
    """Static receptacle: a coloured base plate plus wooden rims (tray) or walls (toy box, pen case), rotated by
    yaw about its centre. half: inner half extents along the receptacle's own x and y axes."""
    hx, hy = (half, half) if isinstance(half, (int, float)) else half
    c, s = math.cos(yaw), math.sin(yaw)

    def at(dx, dy, z):
        return [xy[0] + c * dx - s * dy, xy[1] + s * dx + c * dy, z]

    base_h = .002
    _static(worldbody, f"{name}_base", [hx, hy, base_h], at(0, 0, table_z + base_h), base_rgba,
            [1.5, .005, .0001], yaw)
    wall_half_h = wall_half_height or (.004 if style == "tray" else .025)
    t = .004
    for i, (dx, dy, sx, sy) in enumerate(((hx + t, 0, t, hy + 2 * t), (-hx - t, 0, t, hy + 2 * t),
                                          (0, hy + t, hx, t), (0, -hy - t, hx, t))):
        _static(worldbody, f"{name}_wall_{i}", [sx, sy, wall_half_h], at(dx, dy, table_z + wall_half_h), WOOD,
                [1.0, .005, .0001], yaw)
    return table_z + 2 * base_h  # surface height inside the receptacle

"""The "real" look (sdesk): what the things on the desk look like, with their physics unchanged.

Every collision geom stays exactly as in assets.py (same shapes, masses, friction) and is made invisible; what the
cameras see is a separate set of visual-only geoms (group 1, no contacts, negligible mass) shaped and textured like the
real thing: a pen with a cap, clip and tip; a glazed mug with a curved handle and a print; a crumpled paper ball folded
into facets, with ruled lines; textbooks with printed covers, spines and cream page edges; a notebook with a label and a
wire spiral; a cork coaster; a perforated metal pen holder; a wire-mesh wastebasket the camera sees through; a
phone with a lit screen and an aluminium laptop. The visual shapes follow the collision shapes to within a few
millimetres (the depth camera sees the visual ones).

Textures are drawn here (PIL) once into the asset cache; meshes are written inline in the MJCF."""
import glob
import math
import os

import numpy as np

from .assets import CACHE, MUG_HALF_HEIGHT, MUG_RADIUS, NOTEBOOK_HALF, PAPER_RADIUS, PEN_HALF

VERSION = "v1"  # part of every texture file name: change it to redraw them
TEX = CACHE / "textures"
INVISIBLE = [0, 0, 0, 0]
TINY = "0.000001"  # mass of a visual geom (MuJoCo gives every geom some mass)


# ------------------------------------------------------------------ textures
def _font(size):
    from PIL import ImageFont
    for path in sorted(glob.glob("/usr/share/fonts/**/DejaVuSans-Bold.ttf", recursive=True)) + \
            sorted(glob.glob("/usr/share/fonts/**/LiberationSans-Bold.ttf", recursive=True)) + \
            ["/System/Library/Fonts/Supplemental/Arial Bold.ttf"]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def _rgb(rgba):
    return tuple(int(round(255 * max(0.0, min(1.0, c)))) for c in rgba[:3])


def _texture(name, draw, size=(512, 512)):
    """Path of a cached texture PNG, drawn by draw(width, height) -> PIL image the first time."""
    path = TEX / f"{name}_{VERSION}.png"
    if not path.exists():
        TEX.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.png")
        draw(*size).convert("RGB").save(tmp)
        os.replace(tmp, path)  # several processes may draw the same texture at once
    return str(path)


def _noise(h, w, sigma, seed):
    return np.random.default_rng(seed).normal(0, sigma, (h, w, 1))


def _array_image(arr):
    from PIL import Image
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def tex_paper():
    """Ruled notebook paper, crumpled: off-white, faint blue lines, a red margin, grey creases."""
    def draw(w, h):
        from PIL import ImageDraw
        arr = np.full((h, w, 3), [246.0, 245, 239]) + _noise(h, w, 2.5, 1)
        arr[::22] = [172, 192, 226]
        arr[1::22] = [190, 206, 232]
        arr[:, 66:68] = [226, 140, 140]
        img = _array_image(arr)
        d = ImageDraw.Draw(img)
        rng = np.random.default_rng(2)
        for _ in range(26):
            x0, y0 = rng.uniform(0, w, 2)
            a = rng.uniform(0, np.pi)
            length = rng.uniform(60, 260)
            x1, y1 = x0 + length * np.cos(a), y0 + length * np.sin(a)
            d.line([(x0, y0), (x1, y1)], fill=(196, 196, 190), width=2)
            d.line([(x0 + 2, y0 + 1), (x1 + 2, y1 + 1)], fill=(253, 253, 251), width=1)
        return img
    return _texture("paper", draw)


def tex_pages():
    """Page edges of a closed book: cream, with fine lines along the cover (image rows)."""
    def draw(w, h):
        arr = np.full((h, w, 3), [236.0, 230, 211]) + _noise(h, w, 2, 3)
        arr[::3] -= 16
        arr[1::7] -= 8
        return _array_image(arr)
    return _texture("pages", draw, (256, 256))


def tex_cork():
    def draw(w, h):
        rng = np.random.default_rng(4)
        arr = np.full((h, w, 3), [192.0, 154, 106]) + _noise(h, w, 9, 5)
        for colour, n in (([118, 84, 50], 2600), ([222, 192, 146], 1800), ([150, 110, 70], 2600)):
            ys, xs = rng.integers(0, h - 3, n), rng.integers(0, w - 3, n)
            sizes = rng.integers(1, 4, n)
            for y, x, s in zip(ys, xs, sizes):
                arr[y:y + s, x:x + s] = colour
        return _array_image(arr)
    return _texture("cork", draw)


def tex_perforated():
    """Perforated sheet metal (tinted by the material colour): a light sheet with a grid of dark round holes, plain
    bands at the top and bottom."""
    def draw(w, h):
        from PIL import ImageDraw
        img = _array_image(np.full((h, w, 3), [232.0, 232, 232]) + _noise(h, w, 4, 6))
        d = ImageDraw.Draw(img)
        step, r = 18, 5
        for j, y in enumerate(range(40, h - 40, step)):
            for x in range((j % 2) * step // 2, w + step, step):
                d.ellipse([x - r, y - r, x + r, y + r], fill=(38, 38, 40))
        return img
    return _texture("perforated", draw)


def tex_mug_print():
    """The outside of a mug, white (tinted by the glaze colour), with a small dark print: a cup with steam and a word."""
    def draw(w, h):
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (w, h), (255, 255, 255))
        d = ImageDraw.Draw(img)
        cx, cy = int(w * .5), int(h * .55)
        d.rounded_rectangle([cx - 34, cy - 10, cx + 26, cy + 50], radius=10, outline=(60, 60, 60), width=6)
        d.arc([cx + 14, cy + 2, cx + 46, cy + 34], -90, 90, fill=(60, 60, 60), width=6)
        for dx in (-18, 0, 14):
            d.arc([cx + dx - 6, cy - 52, cx + dx + 6, cy - 30], 90, 270, fill=(60, 60, 60), width=4)
        d.text((cx, cy + 80), "MORNING", font=_font(34), fill=(60, 60, 60), anchor="mm")
        return img
    return _texture("mug_print", draw, (512, 256))


def tex_cover(rgba, title, subtitle, key):
    """A textbook cover in its own colour: a band, a title, a picture block and an author line."""
    def draw(w, h):
        from PIL import ImageDraw
        c = np.array(_rgb(rgba), float)
        arr = np.full((h, w, 3), c) + _noise(h, w, 4, abs(hash(key)) % 1000)
        img = _array_image(arr)
        d = ImageDraw.Draw(img)
        light = c.mean() > 140
        ink = (25, 25, 30) if light else (245, 242, 232)
        band = tuple(int(v) for v in (c * .7 if light else np.minimum(255, c * 1.35 + 30)))
        d.rectangle([0, int(h * .06), w, int(h * .30)], fill=band)
        d.text((w // 2, int(h * .18)), title, font=_font(int(h * .13)), fill=ink, anchor="mm")
        d.text((w // 2, int(h * .40)), subtitle, font=_font(int(h * .055)), fill=ink, anchor="mm")
        d.rectangle([int(w * .2), int(h * .50), int(w * .8), int(h * .82)], fill=band, outline=ink, width=3)
        d.ellipse([int(w * .38), int(h * .55), int(w * .62), int(h * .78)], outline=ink, width=4)
        d.text((w // 2, int(h * .91)), "Third Edition", font=_font(int(h * .045)), fill=ink, anchor="mm")
        return img
    return _texture(f"cover_{key}", draw, (512, 384))


def tex_spine(rgba, key):
    def draw(w, h):
        c = np.array(_rgb(rgba), float)
        arr = np.full((h, w, 3), c) + _noise(h, w, 3, 11)
        arr[:, int(w * .12):int(w * .16)] = [200, 170, 90]
        arr[:, int(w * .84):int(w * .88)] = [200, 170, 90]
        return _array_image(arr)
    return _texture(f"spine_{key}", draw, (256, 64))


def tex_notebook_cover(rgba, key):
    """A notebook cover: its colour with a fine linen grain and a white label with lines to write on."""
    def draw(w, h):
        from PIL import ImageDraw
        c = np.array(_rgb(rgba), float)
        arr = np.full((h, w, 3), c) + _noise(h, w, 5, 12)
        arr[::4] *= .96
        img = _array_image(arr)
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = int(w * .52), int(h * .14), int(w * .90), int(h * .42)
        d.rounded_rectangle([x0, y0, x1, y1], radius=12, fill=(248, 247, 242), outline=(90, 90, 90), width=2)
        d.text(((x0 + x1) // 2, y0 + 26), "NOTES", font=_font(26), fill=(60, 60, 60), anchor="mm")
        for k in range(3):
            y = y0 + 58 + 22 * k
            d.line([(x0 + 14, y), (x1 - 14, y)], fill=(150, 150, 150), width=2)
        return img
    return _texture(f"notebook_{key}", draw, (512, 384))


def tex_aluminium():
    def draw(w, h):
        from PIL import ImageDraw
        rng = np.random.default_rng(13)
        rows = rng.normal(0, 5, (h, 1, 1))
        arr = np.full((h, w, 3), [192.0, 194, 198]) + rows + _noise(h, w, 2, 14)
        img = _array_image(arr)
        d = ImageDraw.Draw(img)
        d.ellipse([w // 2 - 34, h // 2 - 34, w // 2 + 34, h // 2 + 34], fill=(214, 216, 220))
        return img
    return _texture("aluminium", draw)


def tex_phone_screen():
    """A phone's lock screen in portrait: a dark gradient, the time, a grid of app icons and a dock."""
    def draw(w, h):
        from PIL import ImageDraw
        t = np.linspace(0, 1, h)[:, None, None]
        arr = (1 - t) * np.array([30.0, 40, 90]) + t * np.array([90.0, 40, 110]) + np.zeros((h, w, 3))
        img = _array_image(arr)
        d = ImageDraw.Draw(img)
        d.text((w // 2, int(h * .10)), "9:41", font=_font(int(w * .22)), fill=(240, 240, 245), anchor="mm")
        rng = np.random.default_rng(15)
        s = w // 6
        for row in range(4):
            for col in range(4):
                x, y = int(w * .09) + col * int(w * .22), int(h * .24) + row * int(w * .24)
                colour = tuple(int(v) for v in rng.integers(60, 250, 3))
                d.rounded_rectangle([x, y, x + s, y + s], radius=s // 4, fill=colour)
        d.rounded_rectangle([int(w * .05), int(h * .86), int(w * .95), int(h * .97)], radius=20, fill=(120, 110, 140))
        return img
    return _texture("phone_screen", draw, (256, 512))


# ------------------------------------------------------------------ MJCF pieces
def material(name, texture_path=None, rgba=(1, 1, 1, 1), specular=.3, shininess=.3, texrepeat=None, cube=False):
    """(texture element, material element) as MJCF strings for an object's <asset>."""
    tex = ""
    attr = ""
    if texture_path:
        tex = f'<texture name="{name}_tex" type="{"cube" if cube else "2d"}" file="{texture_path}"/>'
        attr = f' texture="{name}_tex"' + (f' texrepeat="{texrepeat[0]} {texrepeat[1]}"' if texrepeat else "")
    return tex + (f'<material name="{name}"{attr} rgba="{" ".join(f"{v:.3f}" for v in rgba)}" '
                  f'specular="{specular}" shininess="{shininess}"/>')


def vgeom(name, type_, size, pos, mat, quat=None):
    """A visual-only geom (group 1, no contacts, negligible mass). No rgba: a geom's own rgba would override its
    material's colour."""
    q = f' quat="{" ".join(f"{v:.6f}" for v in quat)}"' if quat else ""
    return (f'<geom name="{name}" type="{type_}" size="{" ".join(f"{v:.5f}" for v in size)}" '
            f'pos="{" ".join(f"{v:.5f}" for v in pos)}" material="{mat}" group="1" contype="0" '
            f'conaffinity="0" mass="{TINY}"{q}/>')


def vmesh(name, mesh, mat):
    """A visual-only mesh geom; no pos or quat, so the mesh sits where its vertices say."""
    return (f'<geom name="{name}" type="mesh" mesh="{mesh}" material="{mat}" group="1" contype="0" '
            f'conaffinity="0" mass="{TINY}"/>')


def mesh_asset(name, verts, faces, texcoords=None):
    v = " ".join(f"{x:.5f}" for p in verts for x in p)
    f = " ".join(str(int(i)) for t in faces for i in t)
    tc = f' texcoord="{" ".join(f"{x:.4f}" for p in texcoords for x in p)}"' if texcoords is not None else ""
    return f'<mesh name="{name}" vertex="{v}" face="{f}"{tc} inertia="convex"/>'


X_AXIS = (math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0)  # quat turning a cylinder's axis (z) onto x


# ------------------------------------------------------------------ meshes
class _Mesh:
    def __init__(self):
        self.v, self.f, self.t = [], [], []

    def strip(self, ring_a, ring_b, tc_a=None, tc_b=None, outward=True):
        """Quads between two rings of equal length (open strip: the last point does not join the first)."""
        base, n = len(self.v), len(ring_a)
        self.v += list(ring_a) + list(ring_b)
        self.t += list(tc_a or [(0, 0)] * n) + list(tc_b or [(0, 0)] * n)
        for i in range(n - 1):
            a, b, c, d = base + i, base + i + 1, base + n + i, base + n + i + 1
            self.f += [(a, b, d), (a, d, c)] if outward else [(a, d, b), (a, c, d)]

    def fan(self, centre, ring, up=True):
        base = len(self.v)
        self.v += [centre] + list(ring)
        self.t += [(0, 0)] * (len(ring) + 1)
        for i in range(len(ring) - 1):
            self.f.append((base, base + 1 + i, base + 2 + i) if up else (base, base + 2 + i, base + 1 + i))


def _circle(r, z, n, cx=0.0, cy=0.0):
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n), z) for i in range(n + 1)]


def cup_mesh(r_out, r_in, z0, z1, floor, n=48):
    """A closed cup: outer wall (texture u around, v up), rim, inner wall, inner floor, bottom."""
    m = _Mesh()
    m.strip(_circle(r_out, z0, n), _circle(r_out, z1, n), [(i / n, 0) for i in range(n + 1)],
            [(i / n, 1) for i in range(n + 1)])
    m.strip(_circle(r_out, z1, n), _circle(r_in, z1, n))
    m.strip(_circle(r_in, floor, n), _circle(r_in, z1, n), outward=False)
    m.fan((0, 0, floor), _circle(r_in, floor, n), up=True)
    m.fan((0, 0, z0), _circle(r_out, z0, n), up=False)
    return m


def tube_mesh(r_out, r_in, z0, z1, n=48, flare=0.0):
    """A closed thick tube (no floor): outer wall (texture u around, v up), top, inner wall, bottom; flare widens it
    from z0 to z1 by that much (a wall leaning out)."""
    m = _Mesh()
    m.strip(_circle(r_out, z0, n), _circle(r_out + flare, z1, n), [(i / n, 0) for i in range(n + 1)],
            [(i / n, 1) for i in range(n + 1)])
    m.strip(_circle(r_out + flare, z1, n), _circle(r_in + flare, z1, n))
    m.strip(_circle(r_in, z0, n), _circle(r_in + flare, z1, n), outward=False)
    m.strip(_circle(r_in, z0, n), _circle(r_out, z0, n))
    return m


def _merge(meshes):
    m = _Mesh()
    for part in meshes:
        base = len(m.v)
        m.v += part.v
        m.t += part.t
        m.f += [tuple(base + i for i in f) for f in part.f]
    return m


def sweep_mesh(path, radius, n=10, m=None):
    """A tube of the given radius along a path of points (its ends left open: they sit inside something)."""
    m = m or _Mesh()
    path = np.asarray(path, float)
    rings = []
    for k, p in enumerate(path):
        t = path[min(k + 1, len(path) - 1)] - path[max(k - 1, 0)]
        t /= np.linalg.norm(t)
        ref = np.array([0, 0, 1.0]) if abs(t[2]) < .9 else np.array([1.0, 0, 0])
        a = np.cross(t, ref)
        a /= np.linalg.norm(a)
        b = np.cross(t, a)
        rings.append([tuple(p + radius * (math.cos(2 * math.pi * j / n) * a + math.sin(2 * math.pi * j / n) * b))
                      for j in range(n + 1)])
    for k in range(len(rings) - 1):
        m.strip(rings[k], rings[k + 1])
    # winding: the first face's normal must point away from the path
    a, b, c = (np.array(m.v[i]) for i in m.f[-1])
    centre = path[-2]
    if np.dot(np.cross(b - a, c - a), (a + b + c) / 3 - centre) < 0:
        m.f = [(a_, c_, b_) for a_, b_, c_ in m.f]
    return m


def torus_mesh(R, r, centre, axis="z", n_major=48, n_minor=10, m=None):
    """A closed torus about the given axis through centre."""
    m = m or _Mesh()
    cx, cy, cz = centre
    rings = []
    for i in range(n_major + 1):
        th = 2 * math.pi * i / n_major
        ring = []
        for j in range(n_minor + 1):
            ph = 2 * math.pi * j / n_minor
            u, w = (R + r * math.cos(ph)), r * math.sin(ph)
            local = (u * math.cos(th), u * math.sin(th), w)  # torus about z
            if axis == "y":  # about y: the ring lies in the x-z plane
                local = (local[0], local[2], local[1])
            ring.append((cx + local[0], cy + local[1], cz + local[2]))
        rings.append(ring)
    for i in range(n_major):
        # torus about z: (theta, phi) order gives outward normals; the y version is mirrored, so flip it
        m.strip(rings[i], rings[i + 1], outward=(axis == "z"))
    return m


def crumpled_mesh(r, seed=7, subdiv=3):
    """A crumpled paper ball: an icosphere pushed in and out along random folds (|d . n_k - c_k| creases), each face
    with its own vertices so it is shaded flat, like folded paper; texture coordinates from the direction."""
    t = (1 + 5 ** .5) / 2
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    verts = [np.array(v, float) / np.linalg.norm(v) for v in verts]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
             (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
             (8, 6, 7), (9, 8, 1)]
    for _ in range(subdiv):
        cache, new = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                v = verts[a] + verts[b]
                verts.append(v / np.linalg.norm(v))
                cache[key] = len(verts) - 1
            return cache[key]
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new
    rng = np.random.default_rng(seed)
    folds = rng.normal(size=(14, 3))
    folds /= np.linalg.norm(folds, axis=1, keepdims=True)
    offsets, weights = rng.uniform(-.5, .5, 14), rng.uniform(.5, 1.0, 14)
    d = np.array(verts)
    s = (weights * np.abs(d @ folds.T - offsets)).sum(1)
    s = (s - s.min()) / (s.max() - s.min())
    radius = r * (.80 + .28 * s + rng.uniform(-.03, .03, len(d)))
    pts = d * radius[:, None] * np.array([1.0, .96, .9])
    m = _Mesh()
    for a, b, c in faces:
        base = len(m.v)
        for i in (a, b, c):
            m.v.append(tuple(pts[i]))
            m.t.append((math.atan2(d[i, 1], d[i, 0]) / (2 * math.pi) + .5, math.acos(max(-1, min(1, d[i, 2]))) / math.pi))
        m.f.append((base, base + 1, base + 2))
    return m


# ------------------------------------------------------------------ objects (inside an object's MJCF)
def _shade(rgba, f):
    return [min(1.0, c * f) for c in rgba[:3]] + [1]


def pen_visual(rgba):
    """A capped pen along x (14 cm): the barrel in the pen's colour, a black grip and a metal tip at -x, the cap in a
    darker shade with a metal clip at +x."""
    hx = PEN_HALF[0]
    assets = (material("barrel", rgba=rgba, specular=.5, shininess=.6) +
              material("cap", rgba=_shade(rgba, .75), specular=.5, shininess=.6) +
              material("grip", rgba=(.08, .08, .09, 1), specular=.1) +
              material("metal", rgba=(.78, .78, .80, 1), specular=.9, shininess=.9))
    geoms = [vgeom("v_barrel", "cylinder", [.0056, .040], [-.014, 0, 0], "barrel", X_AXIS),
             vgeom("v_grip", "cylinder", [.0060, .012], [-.046, 0, 0], "grip", X_AXIS),
             vgeom("v_cone", "cylinder", [.0036, .003], [-hx + .007, 0, 0], "metal", X_AXIS),
             vgeom("v_tip", "sphere", [.0016], [-hx + .0025, 0, 0], "metal"),
             vgeom("v_cap", "cylinder", [.0062, .022], [hx - .024, 0, 0], "cap", X_AXIS),
             vgeom("v_cap_end", "cylinder", [.0050, .0015], [hx - .0015, 0, 0], "cap", X_AXIS),
             vgeom("v_clip", "box", [.017, .0012, .0007], [hx - .022, 0, .0068], "metal"),
             vgeom("v_clip_end", "sphere", [.0016], [hx - .039, 0, .0064], "metal")]
    return geoms, assets


def mug_visual(rgba):
    """A glazed mug: a smooth cup (4.8 cm, 7 cm tall) with a print on its side and a curved handle on +x."""
    r, h = MUG_RADIUS, MUG_HALF_HEIGHT
    cup = cup_mesh(r, r - .0028, -h, h, -h + .006)
    path = [(r - .002 + .013 * math.cos(t), 0, h * .05 + .021 * math.sin(t))
            for t in np.linspace(-math.pi / 2 - .15, math.pi / 2 + .15, 24)]
    handle = sweep_mesh(path, .0034)
    assets = (material("glaze", tex_mug_print(), rgba=rgba, specular=.6, shininess=.7) +
              material("glaze_plain", rgba=rgba, specular=.6, shininess=.7) +
              mesh_asset("cup", cup.v, cup.f, cup.t) + mesh_asset("handle", handle.v, handle.f))
    return [vmesh("v_cup", "cup", "glaze"), vmesh("v_handle", "handle", "glaze_plain")], assets


def paper_visual(rgba):
    """A crumpled sheet of ruled paper (the paper's colour tints it)."""
    m = crumpled_mesh(PAPER_RADIUS * .97)
    assets = material("sheet", tex_paper(), rgba=rgba, specular=.05, shininess=.1) + mesh_asset("ball", m.v, m.f, m.t)
    return [vmesh("v_ball", "ball", "sheet")], assets


def notebook_visual(rgba):
    """An A5 notebook: printed covers, cream page edges and a wire spiral along its short edge at -x."""
    hx, hy, hz = NOTEBOOK_HALF
    key = "_".join(f"{v:.2f}" for v in rgba[:3])
    spiral = None
    for y in np.linspace(-hy + .008, hy - .008, 17):
        spiral = torus_mesh(.0045, .0007, (-hx + .002, y, 0), axis="y", n_major=16, n_minor=6, m=spiral)
    assets = (material("cover", tex_notebook_cover(rgba, key), specular=.15) +
              material("back", rgba=_shade(rgba, .9), specular=.15) +
              material("pages", tex_pages(), specular=.05) +
              material("wire", rgba=(.75, .75, .78, 1), specular=.9, shininess=.9) +
              mesh_asset("spiral", spiral.v, spiral.f))
    cover = .0008
    geoms = [vgeom("v_cover", "box", [hx, hy, cover], [0, 0, hz - cover], "cover"),
             vgeom("v_back", "box", [hx, hy, cover], [0, 0, -hz + cover], "back"),
             vgeom("v_pages", "box", [hx - .002, hy - .002, hz - 2 * cover], [.001, 0, 0], "pages"),
             vmesh("v_spiral", "spiral", "wire")]
    return geoms, assets


# ------------------------------------------------------------------ static scenery (into the arena)
def _asset(asset_root, xml):
    """Add MJCF asset elements (a string of them) to the arena's <asset>, once per name."""
    import xml.etree.ElementTree as ET
    for el in ET.fromstring(f"<a>{xml}</a>"):
        if asset_root.find(f"./{el.tag}[@name='{el.get('name')}']") is None:
            asset_root.append(el)


def _key(rgba):
    return "_".join(f"{v:.2f}" for v in rgba[:3])


def _vstatic(worldbody, name, type_, size, pos, mat, yaw=0.0, quat=None):
    from robosuite.utils.mjcf_utils import new_geom
    q = quat or [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    worldbody.append(new_geom(name=name, type=type_, size=size, pos=pos, quat=q, group=1, contype="0",
                              conaffinity="0", material=mat))


def _vmesh_static(worldbody, name, mesh):
    import xml.etree.ElementTree as ET
    worldbody.append(ET.fromstring(f'<geom name="{name}" type="mesh" mesh="{mesh}" material="{mesh}_mat" '
                                   f'group="1" contype="0" conaffinity="0"/>'))


def holder_look(worldbody, asset, name, rgba, xy, table_z, half, wall_half_h, yaw):
    """A perforated metal pen holder over the collision walls of add_receptacle (wall thickness 4 mm)."""
    _asset(asset, material(f"{name}_mesh_mat", tex_perforated(), rgba=rgba, specular=.6, shininess=.5) +
           material(f"{name}_base_mat", rgba=_shade(rgba, .7), specular=.4))
    hx, hy = half
    c, s = math.cos(yaw), math.sin(yaw)

    def at(dx, dy, z):
        return [xy[0] + c * dx - s * dy, xy[1] + s * dx + c * dy, z]
    t = .004
    _vstatic(worldbody, f"{name}_v_base", "box", [hx + 2 * t, hy + 2 * t, .002], at(0, 0, table_z + .002),
             f"{name}_base_mat", yaw)
    for i, (dx, dy, sx, sy) in enumerate(((hx + t, 0, t, hy + 2 * t), (-hx - t, 0, t, hy + 2 * t),
                                          (0, hy + t, hx, t), (0, -hy - t, hx, t))):
        _vstatic(worldbody, f"{name}_v_wall_{i}", "box", [sx, sy, wall_half_h], at(dx, dy, table_z + wall_half_h),
                 f"{name}_mesh_mat", yaw)


def coaster_look(worldbody, asset, name, rgba, xy, table_z, half, wall_half_h, yaw):
    """A cork coaster in a frame of the coaster's colour."""
    _asset(asset, material(f"{name}_cork_mat", tex_cork(), specular=.05) +
           material(f"{name}_frame_mat", rgba=rgba, specular=.3))
    hx, hy = half
    c, s = math.cos(yaw), math.sin(yaw)

    def at(dx, dy, z):
        return [xy[0] + c * dx - s * dy, xy[1] + s * dx + c * dy, z]
    t = .004
    _vstatic(worldbody, f"{name}_v_base", "box", [hx, hy, .002], at(0, 0, table_z + .002), f"{name}_cork_mat", yaw)
    for i, (dx, dy, sx, sy) in enumerate(((hx + t, 0, t, hy + 2 * t), (-hx - t, 0, t, hy + 2 * t),
                                          (0, hy + t, hx, t), (0, -hy - t, hx, t))):
        _vstatic(worldbody, f"{name}_v_wall_{i}", "box", [sx, sy, wall_half_h], at(dx, dy, table_z + wall_half_h),
                 f"{name}_frame_mat", yaw)


def bin_look(worldbody, asset, name, rgba, xy, table_z, radius, half_height, bottom_radius):
    """A wire-mesh wastebasket over the collision panels of add_bin (8 mm thick, inner face at bottom_radius at the
    floor and radius at the mouth): a diamond mesh of thin wires crossing between a solid foot band and a solid band
    under a rolled rim, so the camera sees through its sides; a solid metal floor."""
    t, top = .004, table_z + 2 * half_height
    flare = radius - bottom_radius

    def middle(z):  # the middle of the wall at height z
        return bottom_radius + t + flare * (z - table_z) / (2 * half_height)
    foot, band = table_z + .02, top - .012
    foot_mesh = tube_mesh(bottom_radius + 2 * t, bottom_radius, table_z, foot, n=72,
                          flare=flare * (foot - table_z) / (2 * half_height))
    band_mesh = tube_mesh(middle(band) + t, middle(band) - t, band, top, n=72, flare=flare * (top - band) / (2 * half_height))
    rim = torus_mesh(radius + t, .0045, (0, 0, top), n_major=72, n_minor=10)
    # 36 wires each way at 45 degrees (one radian around per radius of height): diamonds about 1.4 cm wide
    wires, n, turn = [], 36, 1.0 / middle((foot + band) / 2)
    for sense in (1, -1):
        for i in range(n):
            a0 = 2 * math.pi * i / n
            path = [(middle(z) * math.cos(a0 + sense * turn * (z - foot)), middle(z) * math.sin(a0 + sense * turn * (z - foot)), z)
                    for z in np.linspace(foot - .003, band + .003, 18)]  # the ends inside the bands
            wires.append(sweep_mesh(path, .0012, n=5))
    parts = {"bands": _merge([foot_mesh, band_mesh, rim]), "mesh": _merge(wires)}
    for part in parts.values():
        part.v = [(x + xy[0], y + xy[1], z) for x, y, z in part.v]
    _asset(asset, material(f"{name}_metal_mat", rgba=rgba, specular=.6, shininess=.5) +
           material(f"{name}_floor_mat", rgba=_shade(rgba, .85), specular=.4, shininess=.4) +
           "".join(mesh_asset(f"{name}_{k}", m.v, m.f) for k, m in parts.items()))
    import xml.etree.ElementTree as ET
    for k in parts:
        worldbody.append(ET.fromstring(f'<geom name="{name}_v_{k}" type="mesh" mesh="{name}_{k}" '
                                       f'material="{name}_metal_mat" group="1" contype="0" conaffinity="0"/>'))
    _vstatic(worldbody, f"{name}_v_floor", "cylinder", [bottom_radius, .002], [xy[0], xy[1], table_z + .002],
             f"{name}_floor_mat")


TITLES = [("PHYSICS", "Mechanics and Waves"), ("CALCULUS", "Early Transcendentals"), ("BIOLOGY", "Cells to Systems"),
          ("HISTORY", "The Modern World"), ("CHEMISTRY", "Atoms First"), ("ECONOMICS", "Principles")]
OTHER_COVERS = [(.16, .34, .24, 1), (.48, .12, .13, 1), (.82, .75, .60, 1), (.20, .24, .40, 1)]


def stack_look(worldbody, asset, name, books, paint=None):
    """Textbooks over the collision boxes of add_stack: books = [(half size, centre, yaw)] from the bottom up. Each has
    printed covers, a spine on its -x side and cream page edges; the top one is in the pile's colour (paint), if any."""
    for i, (half, centre, yaw) in enumerate(books):
        colour = paint if (paint is not None and i == len(books) - 1) else OTHER_COVERS[(i + len(name)) % 4]
        title, subtitle = TITLES[(i + len(name)) % len(TITLES)]
        key = f"{_key(colour)}_{i}"
        _asset(asset, material(f"{name}_{i}_cover_mat", tex_cover(colour, title, subtitle, f"{key}_{title}"), specular=.3) +
               material(f"{name}_{i}_back_mat", rgba=colour, specular=.3) +
               material(f"{name}_{i}_spine_mat", tex_spine(colour, key), specular=.3) +
               material(f"{name}_pages_mat", tex_pages(), specular=.05))
        hx, hy, hz = half
        c, s = math.cos(yaw), math.sin(yaw)

        def at(dx, dy, dz):
            return [centre[0] + c * dx - s * dy, centre[1] + s * dx + c * dy, centre[2] + dz]
        cover = .0012
        _vstatic(worldbody, f"{name}_{i}_v_cover", "box", [hx, hy, cover], at(0, 0, hz - cover), f"{name}_{i}_cover_mat", yaw)
        _vstatic(worldbody, f"{name}_{i}_v_back", "box", [hx, hy, cover], at(0, 0, -hz + cover), f"{name}_{i}_back_mat", yaw)
        _vstatic(worldbody, f"{name}_{i}_v_spine", "box", [.002, hy, hz], at(-hx + .002, 0, 0), f"{name}_{i}_spine_mat", yaw)
        _vstatic(worldbody, f"{name}_{i}_v_pages", "box", [hx - .004, hy - .003, hz - 2 * cover], at(.0015, 0, 0),
                 f"{name}_pages_mat", yaw)


def decor_look(worldbody, asset, table_z):
    """Visuals for add_decor's laptop (aluminium lid, dark seam) and phone (a lit screen in a black frame)."""
    _asset(asset, material("decor_alu_mat", tex_aluminium(), specular=.7, shininess=.6) +
           material("decor_seam_mat", rgba=(.10, .10, .11, 1), specular=.2) +
           material("decor_phone_frame_mat", rgba=(.06, .06, .07, 1), specular=.8, shininess=.8) +
           material("decor_screen_mat", tex_phone_screen(), specular=.9, shininess=.9))
    _vstatic(worldbody, "decor_laptop_v_lid", "box", [.075, .11, .0035], [.315, .03, table_z + .0125], "decor_alu_mat")
    _vstatic(worldbody, "decor_laptop_v_seam", "box", [.0745, .1095, .001], [.315, .03, table_z + .0085], "decor_seam_mat")
    _vstatic(worldbody, "decor_laptop_v_base", "box", [.075, .11, .0035], [.315, .03, table_z + .0035], "decor_alu_mat")
    _vstatic(worldbody, "decor_phone_v_frame", "box", [.035, .07, .0035], [-.31, .32, table_z + .0035],
             "decor_phone_frame_mat", .3)
    _vstatic(worldbody, "decor_phone_v_screen", "box", [.032, .066, .0006], [-.31, .32, table_z + .0074],
             "decor_screen_mat", .3)

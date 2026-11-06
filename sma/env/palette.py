"""Colours for the colour-coded desk (cdesk): every thing on it, objects and receptacles alike, has its own hue, so the
camera can tell the things apart by colour without being told what any of them is.

The scene lighting and the renderer pull hues together (nominal OpenCV hues 24-36 all render as 30, 84-100 as 90, 144-156
as 150), so the colours were chosen from a sweep of 45 nominal hues (scripts/calibrate_hue_sweep.py, 2026-11-05): their
rendered hues are at least 15 apart. RENDERED_HUE is what the top-down camera measured; perception labels a pixel
with the nearest of these within PALETTE_TOLERANCE."""
import colorsys

# name -> nominal OpenCV hue (0-180) of the material. Nine colours with rendered hues 15-35 apart, none near 150: purples
# there render bluer on curved, shaded surfaces (a magenta paper ball read as violet and was lost in the first trial).
# Twelve colours 13 apart let a notebook's shaded side read as its neighbour's colour. A desk has at most nine things.
NOMINAL_HUE = {"red": 4, "orange": 16, "yellow": 44, "green": 60, "teal": 76, "azure": 104, "blue": 116, "violet": 128,
               "rose": 172}
PALETTE_TOLERANCE = 7


def _rgba(hue, saturation=.88, value=.82):
    r, g, b = colorsys.hsv_to_rgb(hue / 180.0, saturation, value)
    return [round(r, 3), round(g, 3), round(b, 3), 1]


PALETTE = {name: _rgba(h) for name, h in NOMINAL_HUE.items()}
# Measured on the top faces of toy blocks under the scene lighting (hue sweep, 2026-11-05).
RENDERED_HUE = {"red": 5, "orange": 20, "yellow": 39, "green": 60, "teal": 81, "azure": 97, "blue": 115, "violet": 133,
                "rose": 168}


# sdesk: ordinary desk colours, several things sharing each (the camera cannot tell things apart by colour there).
NATURAL = {"white": [.86, .86, .83, 1], "black": [.11, .11, .12, 1], "navy": [.13, .18, .42, 1],
           "red": [.70, .13, .12, 1], "grey": [.48, .49, .51, 1]}


def shade(rgba, factor):
    """The same hue, darker (factor < 1) or lighter: walls, rims and the books of a pile."""
    return [round(min(1.0, c * factor), 3) for c in rgba[:3]] + [1]

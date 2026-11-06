"""What a thing is, for the harness's own bookkeeping (skill keys, grouping, which camera check applies).

Named scenes ("green_pen", "red_bin") carry it in the name. On the colour-coded desk the camera names things by colour
only, and the model's identification (category words from VOCABULARY) is the source; the harness maps those words to
its short keys here."""

VOCABULARY = ["pen", "pencil", "mug", "notebook", "book", "crumpled paper ball", "toy block", "wastebasket",
              "pen holder", "coaster", "pile of books", "tray", "box", "pen case"]
# category word -> the harness's object kind or receptacle style
KIND_OF = {"pen": "pen", "pencil": "pen", "mug": "mug", "notebook": "notebook", "book": "book",
           "crumpled paper ball": "paper", "toy block": "block"}
STYLE_OF = {"wastebasket": "bin", "pen holder": "holder", "coaster": "coaster", "pile of books": "pile", "tray": "tray",
            "box": "box", "pen case": "case"}


def kind_of(name, scene=None):
    """An object's kind: from the identification on a colour-coded desk, else from its name ("green_pen" -> "pen")."""
    obj = ((scene or {}).get("objects") or {}).get(name) or {}
    if obj.get("category"):
        return KIND_OF.get(obj["category"], obj["category"])
    return name.split("_", 1)[1] if "_" in name else "unknown"


def zone_style(name, scene=None):
    """A receptacle's style: from the identification on a colour-coded desk, else from its name ("red_bin" -> "bin")."""
    zone = ((scene or {}).get("zones") or {}).get(name) or {}
    if zone.get("category"):
        return STYLE_OF.get(zone["category"], zone["category"])
    return name.split("_")[-1]

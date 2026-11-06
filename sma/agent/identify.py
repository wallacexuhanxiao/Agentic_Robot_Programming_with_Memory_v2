"""Recognition by the model: on the colour-coded desk the camera reports things by colour with measurements only, and
the model says what each is from two labelled camera images and the measurements. The harness then uses the model's
own reading (TabletopEnv.set_identity); nothing it is told names a thing's category. Whether a thing is moved or is a
place for others follows from the category the model gives (role_of)."""
import base64
import json
import os
import re

from ..env.naming import VOCABULARY

IDENTIFY = """You are the vision module of a tabletop robot. Image 1 shows a desk from straight above (image right = +x,
image up = +y); image 2 shows the same desk from the side. {marked}{closeups}
The depth camera measured the marked things (metres; heights above the desk; "hollow": its middle is at least 2 cm
below its rim, with the rim and inner floor heights; "top_flat": most of its top is level):
{things}

For each marked thing, say what it is: one of {vocabulary}, or "other: <a few words>".
Reply with JSON only, keyed by the numbers:
{{"things": {{"<number>": {{"category": "...", "why": "<few words>"}}}}}}"""

MARKED = {"color": "Every coloured thing on it is marked with a number in a small white tag beside it, joined to it by a "
                   "thin line (the tag is not part of the thing). Grey and black things are not marked and are not part of "
                   "this.",
          "sam": "Every separate thing the camera found on it is marked with a number in a small white tag beside it, joined "
                 "to it by a thin line (the tag is not part of the thing). A marked region may also be only a part of "
                 "something, or something that is not for tidying (say so with \"other: ...\")."}
SOM = {"color": "Every coloured thing on it is outlined and marked with a number in a small white tag beside it, joined "
                "to it by a thin line (outline and tag are not part of the thing). Grey and black things are not marked "
                "and are not part of this.",
       "sam": "Every separate thing the camera found on it is outlined and marked with a number in a small white tag "
              "beside it, joined to it by a thin line (outline and tag are not part of the thing). A marked region may "
              "also be only a part of something, or something that is not for tidying (say so with \"other: ...\")."}
CLOSEUPS = (" Image 3 shows each marked thing close up: its number, then the view from above and the view from the side, "
            "cropped around it, with its outline (\"not visible\": hidden from that side). Things hidden from the side are "
            "not outlined in image 2.")
# Model calls the latest identify() needed. With a dozen marked things deepseek-flash thought through 9,500 tokens and
# said nothing in 5 of 5 calls (sdesk, 2026-11-05): the first call may think for up to 16,000 tokens; an empty or
# unreadable reply is asked again once, without thinking.
LAST = {"attempts": 0}
KEEP = ("footprint_long", "footprint_short", "height", "shape", "hollow", "rim_height", "inner_floor_height", "top_flat")


def identify(llm, env):
    """Ask the model what each coloured thing is; set and return the identity ({name: {"category", "role", "why"}})."""
    env.set_identity({})
    scene = env.observe()
    # the colours (drawn at random per desk) or the tracker's thing names: the numbers say nothing about what it is
    names = sorted(scene["things"], key=lambda n: (len(n), n))
    number = {n: str(i + 1) for i, n in enumerate(names)}
    things = {number[n]: {k: scene["things"][n][k] for k in KEEP if k in scene["things"][n]} for n in names}
    marks = os.environ.get("SMA_MARKS", "tags")
    views = env.labelled_views(labels=number, marks=marks)
    source = "sam" if getattr(env, "perception", "color") == "sam" else "color"
    marked = (SOM if marks == "som" else MARKED)[source]
    closeups = CLOSEUPS if any(v["camera"] == "closeups" for v in views) else ""
    content = [{"type": "text", "text": IDENTIFY.format(marked=marked, closeups=closeups, things=json.dumps(things),
                                                        vocabulary=", ".join(VOCABULARY))}]
    for v in views:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + base64.b64encode(v["png"]).decode()}})
    for attempt in (1, 2):
        text = llm.chat([{"role": "user", "content": content}], purpose="identify", max_tokens=1500,
                        think_room=16000 if attempt == 1 else 8000, no_thinking=attempt == 2)
        match = re.search(r"\{.*\}", text or "", re.S)
        try:
            reply = json.loads(match.group(0)).get("things", {}) if match else {}
        except (json.JSONDecodeError, AttributeError):
            reply = {}
        LAST["attempts"] = attempt
        if reply:
            break
    identity = {}
    for name in names:
        r = reply.get(number[name]) or {}
        category = str(r.get("category", "other")).strip().lower()
        identity[name] = {"category": category, "role": role_of(category), "why": str(r.get("why", ""))[:120],
                          "tag": number[name]}
    env.set_identity(identity)
    return identity, views


def role_of(category):
    """What the harness does with a thing of this category: move it (an object), put things in or on it (a receptacle:
    a mug is a container, but on a desk it is something to put away), or leave it."""
    from ..env.naming import KIND_OF, STYLE_OF
    return "object" if category in KIND_OF else "receptacle" if category in STYLE_OF else "other"


def private_truth(env, task, scene=None):
    """Private, for scoring and reference programs only: ({thing the camera reports: what it is}, {private item name:
    the thing it is reported as}). On the colour-coded desk both come with the task; with SAM perception the things are
    matched to the task's items by position (TabletopEnv.private_thing_truth), and a thing matched to none is "other"."""
    meta = task.meta or {}
    if getattr(env, "perception", "color") != "sam":
        return meta.get("true_category", {}), meta.get("perceived_name", {})
    match = env.private_thing_truth(scene)
    category = meta.get("category") or {}
    return ({t: category.get(k, "other") if k else "other" for t, k in match.items()},
            {k: t for t, k in match.items() if k})


def score_identity(identity, true_category):
    """Private, for the report: how many things the model named with the right category (and role). A thing that is
    none of the task's items ("other": the laptop, a fragment) is right when the model says "other: ..."."""
    rows = []
    for name, truth in true_category.items():
        got = (identity.get(name) or {}).get("category")
        role = (identity.get(name) or {}).get("role")
        ok = got == truth or (truth == "other" and str(got or "").startswith("other"))
        rows.append({"thing": name, "truth": truth, "said": got, "role": role, "category_ok": ok,
                     "role_ok": role == role_of(truth)})
    return {"things": len(rows), "category_ok": sum(r["category_ok"] for r in rows),
            "role_ok": sum(r["role_ok"] for r in rows), "rows": rows}

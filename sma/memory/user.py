"""Long-term user memory: placement preferences the user stated, with their evidence.

Separate from the skill memory (how to manipulate). Only explicit lasting statements
("always", "from now on", "I like ... to be ...") become long-term rules; task-only requests and
robot successes never do. Rules are keyed by object kind ("block") or a specific object ("red_block").
"""
import json
import re
from pathlib import Path

LEARN = """Extract lasting placement preferences from ONE user's message to a tabletop robot.
Message: "{message}"
Object kinds: block (toy block), mug, book. Objects currently visible: {objects}. Places: {zones}.
Return ONLY JSON: {{"updates": [{{"subject": "<kind or object name>", "zone": "<place name, e.g. purple_box>",
"scope": "long_term" | "task_only", "evidence": "<verbatim quote from the message>"}}]}}
A habit stated with words like always / from now on / usually / I like / I prefer is long_term.
A request only for this time is task_only. A message without placement statements gives no updates.
Never invent a placement that the message does not state."""


def resolve_place(text, scene):
    """Map a free-form place mention ('purple toy box', 'cyan_tray') to a perceived place name."""
    t = text.lower().strip().replace(" ", "_")
    if t in scene["zones"]:
        return t
    matches = [z for z in scene["zones"] if z.split("_")[0] in t]
    return matches[0] if len(matches) == 1 else t


KIND_WORDS = {"block": ("block", "blocks", "cube", "cubes"), "mug": ("mug", "mugs", "cup", "cups"),
              "book": ("book", "books", "notebook", "notebooks")}
LASTING = ("always", "from now on", "usually", "prefer", "every time", "in future", "in the future")


def supported(message, subject, zone, scene):
    """The message itself must name the object (kind word or exact name) and the place's colour word.
    Wording may differ from the quote; a placement the message never mentions is rejected."""
    import re as _re
    m = message.lower()
    if zone not in scene["zones"]:
        return False
    words = KIND_WORDS.get(subject, (subject.replace("_", " "), subject.split("_")[-1]))
    color = zone.split("_")[0]
    colors = {z.split("_")[0] for z in scene["zones"]} | {"orange", "cyan", "purple"}
    # The first colour word after the object word must be this place's colour ("my mugs on the cyan
    # tray"), so a mug is never paired with a colour the message attaches to a different object.
    for w in words:
        for hit in _re.finditer(rf"\b{_re.escape(w)}\b", m):
            later = _re.search(r"\b(" + "|".join(sorted(colors)) + r")\b", m[hit.end():hit.end() + 60])
            if later and later.group(1) == color:
                return True
    return False


def _json(text):
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0)) if match else {}


class UserMemory:
    def __init__(self, path, user_id="user_a"):
        self.path = Path(path)
        self.user_id = user_id
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.rules = data.get("rules", {})      # subject -> rule
        self.events = data.get("events", [])

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"user_id": self.user_id, "rules": self.rules,
                                         "events": self.events}, indent=1, ensure_ascii=False))

    def learn(self, llm, message, scene, task_id):
        raw = llm.chat([{"role": "user", "content": LEARN.format(
            message=message, objects=sorted(scene["objects"]), zones=sorted(scene["zones"]))}],
            purpose="user_memory_write", max_tokens=600)
        try:
            updates = _json(raw).get("updates", [])
        except json.JSONDecodeError:
            updates = []
        kinds = {"block", "mug", "book"} | {k + "s" for k in ("block", "mug", "book")}
        aliases = {"toy block": "block", "toy blocks": "block", "cup": "mug", "cups": "mug"}
        accepted, rejected = [], []
        for u in updates:
            raw = aliases.get(str(u.get("subject", "")).lower().strip(), str(u.get("subject", "")).lower().strip())
            subject = raw.rstrip("s") if raw in kinds else raw
            zone = resolve_place(str(u.get("zone", "")), scene)
            evidence = str(u.get("evidence", ""))
            ok = (subject in {"block", "mug", "book"} or subject in scene["objects"]) and \
                supported(message, subject, zone, scene)
            if ok and u.get("scope") == "long_term" and not any(w in message.lower() for w in LASTING):
                u = {**u, "scope": "task_only"}  # no lasting wording in the message: not a long-term habit
            (accepted if ok else rejected).append({"subject": subject, "zone": zone,
                                                   "scope": u.get("scope"), "evidence": evidence})
        for u in accepted:
            if u["scope"] == "long_term":
                self.rules[u["subject"]] = {"zone": u["zone"], "evidence": u["evidence"], "source_task": task_id}
        self.events.append({"task": task_id, "message": message, "accepted": accepted, "rejected": rejected})
        self.save()
        return {"accepted": accepted, "rejected": rejected}

    def retrieve(self, scene):
        """Rules that apply to an object currently in view (specific rules override kind rules)."""
        out = []
        for name, obj in scene["objects"].items():
            rule = self.rules.get(name) or self.rules.get(obj["kind"])
            if rule:
                out.append({"object": name, "zone": rule["zone"],
                            "rule_subject": name if name in self.rules else obj["kind"],
                            "evidence": rule["evidence"], "source_task": rule["source_task"]})
        return out

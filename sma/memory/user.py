"""Long-term user memory: placement preferences and handling requests ("how") the user stated, with evidence.

Separate from the skill memory (how to manipulate). Only explicit lasting statements
("always", "from now on", "I like ... to be ...") become long-term rules; task-only requests and
robot successes never do. Rules are keyed by object kind ("block") or a specific object ("red_block").
"""
import json
import re
from pathlib import Path

LEARN = """Extract lasting preferences from ONE user's message to a tabletop robot.
Message: "{message}"
Object kinds: block (toy block), mug, book, pen. Objects currently visible: {objects}. Places: {zones}.
Return ONLY JSON: {{"updates": [{{"subject": "<kind or object name>", "zone": "<place name, e.g. purple_box>",
"scope": "long_term" | "task_only", "evidence": "<verbatim quote from the message>"}}],
"handling": [{{"subject": "<kind or object name>", "request": "<how the user wants it handled, in their terms>",
"scope": "long_term" | "task_only", "evidence": "<verbatim quote from the message>"}}]}}
"updates" are placements (which place an object goes to); "handling" are requests about HOW to handle an object
(for example how to orient it), not where it goes.
A habit stated with words like always / from now on / whenever / usually / I like / I prefer is long_term.
A request only for this time is task_only. A message without such statements gives empty lists.
Never invent anything that the message does not state."""


def resolve_place(text, scene):
    """Map a free-form place mention ('purple toy box', 'cyan_tray') to a perceived place name."""
    t = text.lower().strip().replace(" ", "_")
    if t in scene["zones"]:
        return t
    matches = [z for z in scene["zones"] if z.split("_")[0] in t]
    return matches[0] if len(matches) == 1 else t


KIND_WORDS = {"block": ("block", "blocks", "cube", "cubes"), "mug": ("mug", "mugs", "cup", "cups"),
              "book": ("book", "books", "notebook", "notebooks"), "pen": ("pen", "pens")}
KINDS = set(KIND_WORDS)
LASTING = ("always", "from now on", "usually", "prefer", "every time", "in future", "in the future", "whenever")


def quoted(evidence, message):
    """The evidence must be the user's own words: every '...'-separated fragment occurs in the message."""
    norm = lambda t: re.sub(r"\s+", " ", t.lower()).strip(" .,'\"")
    fragments = [norm(f) for f in str(evidence).split("...") if norm(f)]
    return bool(fragments) and all(f in norm(message) for f in fragments)


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
        self.rules = data.get("rules", {})      # subject -> placement rule
        self.handling = data.get("handling", {})  # subject -> how-to request
        self.events = data.get("events", [])

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"user_id": self.user_id, "rules": self.rules, "handling": self.handling,
                                         "events": self.events}, indent=1, ensure_ascii=False))

    def learn(self, llm, message, scene, task_id):
        raw = llm.chat([{"role": "user", "content": LEARN.format(
            message=message, objects=sorted(scene["objects"]), zones=sorted(scene["zones"]))}],
            purpose="user_memory_write", max_tokens=600)
        try:
            parsed = _json(raw)
        except json.JSONDecodeError:
            parsed = {}
        updates, handling = parsed.get("updates", []) or [], parsed.get("handling", []) or []
        kinds = KINDS | {k + "s" for k in KINDS}
        aliases = {"toy block": "block", "toy blocks": "block", "cup": "mug", "cups": "mug"}
        accepted, rejected = [], []
        for u in updates:
            raw = aliases.get(str(u.get("subject", "")).lower().strip(), str(u.get("subject", "")).lower().strip())
            subject = raw.rstrip("s") if raw in kinds else raw
            zone = resolve_place(str(u.get("zone", "")), scene)
            evidence = str(u.get("evidence", ""))
            ok = (subject in KINDS or subject in scene["objects"]) and \
                supported(message, subject, zone, scene)
            if ok and u.get("scope") == "long_term" and not any(w in message.lower() for w in LASTING):
                u = {**u, "scope": "task_only"}  # no lasting wording in the message: not a long-term habit
            (accepted if ok else rejected).append({"subject": subject, "zone": zone,
                                                   "scope": u.get("scope"), "evidence": evidence})
        for u in accepted:
            if u["scope"] == "long_term":
                self.rules[u["subject"]] = {"zone": u["zone"], "evidence": u["evidence"], "source_task": task_id}
        # Handling requests: kept only if the quote is the user's own words and names the object kind.
        handled = []
        for h in handling:
            raw = aliases.get(str(h.get("subject", "")).lower().strip(), str(h.get("subject", "")).lower().strip())
            subject = raw.rstrip("s") if raw in kinds else raw
            words = KIND_WORDS.get(subject, (subject.replace("_", " "),))
            ok = (subject in KINDS or subject in scene["objects"]) and quoted(h.get("evidence", ""), message) and \
                any(re.search(rf"\b{re.escape(w)}\b", message.lower()) for w in words)
            scope = h.get("scope")
            if ok and scope == "long_term" and not any(w in message.lower() for w in LASTING):
                scope = "task_only"
            entry = {"subject": subject, "request": str(h.get("request", "")), "scope": scope,
                     "evidence": str(h.get("evidence", "")), "accepted": bool(ok)}
            handled.append(entry)
            if ok and scope == "long_term":
                self.handling[subject] = {"request": entry["request"], "evidence": entry["evidence"],
                                          "source_task": task_id}
        self.events.append({"task": task_id, "message": message, "accepted": accepted, "rejected": rejected,
                            "handling": handled})
        self.save()
        return {"accepted": accepted, "rejected": rejected, "handling": handled}

    def handling_for(self, scene):
        """Stored how-to requests for objects in view, quoted as the user said them."""
        out = []
        for name, obj in scene["objects"].items():
            rule = self.handling.get(name) or self.handling.get(obj["kind"])
            if rule:
                out.append({"object": name, "request": rule["request"], "user_said": rule["evidence"],
                            "source_task": rule["source_task"]})
        return out

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

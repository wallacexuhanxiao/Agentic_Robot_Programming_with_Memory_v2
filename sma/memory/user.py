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
    if len(matches) == 1:
        return matches[0]
    # colour-coded desk: a place named by what it was identified as ("pen_holder"), if exactly one place is that
    words = t.replace("_", " ")
    by_category = [n for n, z in scene["zones"].items() if z.get("category") and z["category"] in words]
    return by_category[0] if len(by_category) == 1 else t


KIND_WORDS = {"block": ("block", "blocks", "cube", "cubes"), "mug": ("mug", "mugs", "cup", "cups"),
              "paper": ("paper", "paper ball", "paper balls"),
              "book": ("book", "books", "notebook", "notebooks"), "pen": ("pen", "pens"),
              "notebook": ("notebook", "notebooks")}
PLACE_WORDS = {"book_pile": ("pile", "stack")}  # places without a colour: the words that name them
CATEGORY_PLACE_WORDS = {"wastebasket": ("wastebasket", "basket", "bin"), "pen holder": ("holder", "pen holder"),
                        "coaster": ("coaster",), "pile of books": ("pile", "stack"), "tray": ("tray",), "box": ("box",),
                        "pen case": ("case", "pen case")}
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
    if scene["zones"][zone].get("category"):  # colour-coded desk: places are named by what the model saw them as
        own = CATEGORY_PLACE_WORDS.get(scene["zones"][zone]["category"], (scene["zones"][zone]["category"],))
        every = {w for z in scene["zones"].values() for w in CATEGORY_PLACE_WORDS.get(z.get("category"), ())} | set(own)
        for w in words:
            for hit in _re.finditer(rf"\b{_re.escape(w)}", m):
                later = _re.search(r"\b(" + "|".join(sorted(map(_re.escape, every))) + r")\b", m[hit.end():hit.end() + 60])
                if later and later.group(1) in own:
                    return True
        return False
    color = zone.split("_")[0]
    colors = {z.split("_")[0] for z in scene["zones"]} | {"orange", "cyan", "purple"} | \
        {w for ws in PLACE_WORDS.values() for w in ws}
    # The first colour word after the object word must be this place's colour ("my mugs on the cyan
    # tray"), so a mug is never paired with a colour the message attaches to a different object.
    for w in words:
        for hit in _re.finditer(rf"\b{_re.escape(w)}\b", m):
            later = _re.search(r"\b(" + "|".join(sorted(colors)) + r")\b", m[hit.end():hit.end() + 60])
            if later and (later.group(1) == color or later.group(1) in PLACE_WORDS.get(zone, ())):
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
        # Observed habits (from the scene grouping): category -> place -> {"count", "tasks"}; a place is kept as what it
        # is (the model's category on a colour-coded desk, where colours change from desk to desk; else its name).
        self.habits = data.get("habits", {})

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"user_id": self.user_id, "rules": self.rules, "handling": self.handling,
                                         "events": self.events, "habits": self.habits}, indent=1, ensure_ascii=False))

    @staticmethod
    def _place_label(zone_name, scene):
        """How a place is remembered: what it is on a colour-coded desk (its name, a colour, changes per desk)."""
        return (scene["zones"].get(zone_name) or {}).get("category") or zone_name

    @staticmethod
    def _zone_for(label, scene):
        """The zone on this desk that a remembered place stands for: by name, or the one place of that category."""
        if label in scene["zones"]:
            return label
        same = [n for n, z in scene["zones"].items() if z.get("category") == label]
        return same[0] if len(same) == 1 else None

    def observe_habits(self, grouping, scene, task_id):
        """Write what this desk shows about where things are kept into long-term memory: each category seen in a place
        (a thing already there, or the group a place stands for, e.g. the books of a pile) is one observation, at most
        once per desk. Returns the observations added."""
        added = []
        for m in (grouping or {}).get("members", []):
            if m["place"] is None or not m["category"]:
                continue
            place = self._place_label(m["place"], scene)
            entry = self.habits.setdefault(m["category"], {}).setdefault(place, {"count": 0, "tasks": []})
            if task_id not in entry["tasks"]:
                entry["count"] += 1
                entry["tasks"].append(task_id)
                added.append({"category": m["category"], "place": place})
        self.save()
        return added

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
            if u["scope"] == "long_term":  # the place as what it is, so the rule carries over to another desk
                self.rules[u["subject"]] = {"zone": self._place_label(u["zone"], scene), "evidence": u["evidence"],
                                            "source_task": task_id}
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
        """Long-term memory for the objects in view: first a preference the user stated (a specific object's rule over
        its kind's), else an observed habit: where things of this or a similar category (clustered as in the scene
        grouping) were kept on earlier desks, the place with the most similarity-weighted observations."""
        from ..agent.grouping import THRESHOLD, similarity
        out = []
        for name, obj in scene["objects"].items():
            rule = self.rules.get(name) or self.rules.get(obj["kind"])
            zone = self._zone_for(rule["zone"], scene) if rule else None
            if zone:
                out.append({"object": name, "zone": zone, "origin": "stated by the user",
                            "rule_subject": name if name in self.rules else obj["kind"],
                            "evidence": rule["evidence"], "source_task": rule["source_task"]})
                continue
            votes, seen = {}, {}
            for category, places in self.habits.items():
                weight = similarity(obj["kind"], category) if obj.get("kind") else 0
                if weight < THRESHOLD:
                    continue
                for place, e in places.items():
                    votes[place] = votes.get(place, 0.0) + weight * e["count"]
                    seen.setdefault(place, []).append(f"{category} x{e['count']}")
            for place, _ in sorted(votes.items(), key=lambda kv: -kv[1]):
                zone = self._zone_for(place, scene)
                if zone:
                    out.append({"object": name, "zone": zone, "origin": "observed habit",
                                "evidence": f"on earlier desks things like it were kept in {place} "
                                            f"({', '.join(seen[place])})"})
                    break
        return out

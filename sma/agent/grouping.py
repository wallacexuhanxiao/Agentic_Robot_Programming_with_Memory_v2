"""Scene grouping: where a loose object belongs, from where similar things already are on this desk.

Like TidyBot (Wu et al., 2023), which generalises a user's placement examples with an LLM, except that the examples
here are the desk itself: things of a kind are kept together. The perceived categories (object kinds, and a pile of
books, which is itself a group of books) are clustered by the overlap of their category chains (a notebook and a book
share "book" and "paper product"; a pen and a notebook only "stationery"), single-link above a similarity threshold.
A loose object's home is the place already holding the most members of its cluster; with no member placed anywhere
there is no home, and the decision is left to the instruction or to common sense. Only what observe() reports is used
(no simulator state, no LLM)."""
from ..env.perception import inside_zone

# Each category with its broader categories (hand-written, WordNet-like). Unknown kinds form their own cluster.
CHAINS = {
    "pen": ("pen", "writing tool", "stationery"),
    "pencil": ("pencil", "writing tool", "stationery"),
    "notebook": ("notebook", "book", "paper product", "stationery"),
    "book": ("book", "paper product"),
    "paper": ("crumpled paper", "waste", "paper product"),
    "mug": ("mug", "drinkware", "kitchenware"),
    "block": ("toy block", "toy"),
}
# Zones that are themselves a group of things: zone -> (category, thickness of one item, to count them from the height).
GROUP_ZONES = {"book_pile": ("book", .028), "pile of books": ("book", .028)}  # by name, or by the model's category
THRESHOLD = .4


def similarity(a, b):
    """Overlap (Jaccard) of two categories' chains: 1 for the same category."""
    ca, cb = set(CHAINS.get(a, (a,))), set(CHAINS.get(b, (b,)))
    return len(ca & cb) / len(ca | cb)


def cluster(categories, threshold=THRESHOLD):
    """Single-link clusters of categories: {category: cluster index}."""
    cats = sorted(set(categories))
    label = {c: i for i, c in enumerate(cats)}
    for i, a in enumerate(cats):
        for b in cats[i + 1:]:
            if similarity(a, b) >= threshold and label[a] != label[b]:
                old, new = label[b], label[a]
                label = {c: (new if l == old else l) for c, l in label.items()}
    return label


def place_of(obj, scene):
    """The zone an object is in or on (its centre inside the zone's footprint), or None if it is loose on the desk."""
    for name, zone in scene["zones"].items():
        if inside_zone(obj["center"], zone, .005):
            return name
    return None


def infer_homes(scene, threshold=THRESHOLD):
    """{"members": [...], "clusters": [[categories]], "homes": {loose object: {"zone", "evidence", "similar"}}}."""
    members = []  # (category, place or None, object name or None)
    for name, obj in scene["objects"].items():
        members.append((obj["kind"], place_of(obj, scene), name))
    for zone_name, zone in scene["zones"].items():
        key = zone.get("category") or zone_name
        if key in GROUP_ZONES:
            category, thickness = GROUP_ZONES[key]
            count = max(1, round((zone.get("top_height") or thickness) / thickness))
            members += [(category, zone_name, None)] * count
    label = cluster([m[0] for m in members], threshold)
    groups = {}
    for c, l in label.items():
        groups.setdefault(l, []).append(c)
    homes = {}
    for kind, place, name in members:
        if name is None or place is not None:
            continue
        mates = [m for m in members if label[m[0]] == label[kind] and m[2] != name]
        counts = {}
        for c, p, _ in mates:
            if p is not None:
                counts[p] = counts.get(p, 0) + 1
        loose = sum(1 for _, p, _ in mates if p is None)
        if not counts:
            homes[name] = {"zone": None, "similar": sorted({c for c, _, _ in mates}),
                           "evidence": f"no thing like it ({'/'.join(groups[label[kind]])}) is kept in a place yet"}
            continue
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            homes[name] = {"zone": None, "similar": sorted({c for c, _, _ in mates}),
                           "evidence": f"things like it are split evenly: {dict(ranked)}"}
            continue
        zone, n = ranked[0]
        kinds = sorted({c for c, p, _ in mates if p == zone})
        homes[name] = {"zone": zone, "similar": kinds,
                       "evidence": f"{n} of the {len(mates)} other things like it ({', '.join(kinds)}) are in {zone}"
                                   + (f"; {loose} lie loose" if loose else "")}
    return {"members": [{"category": c, "place": p, "object": n} for c, p, n in members],
            "clusters": sorted(sorted(g) for g in groups.values()), "homes": homes}

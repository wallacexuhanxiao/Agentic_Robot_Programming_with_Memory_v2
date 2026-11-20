"""Tidiness metrics, defined once and computed from the downloaded runs (no simulator, no LLM).

Definition (offline, ground truth; the robot never sees any of it):
  O        the movable things on the desk: the ones out of place at the start (O_out) and the ones already in
           their place (O_in).
  h(o)     the home of thing o: the receptacle the task generator assigns (things of a kind are kept together).
  Placed   a geometric test of o against h(o) from the simulator pose (TabletopEnv.private_evaluate), per kind of
           receptacle:
             wastebasket   centre within the basket radius - 5 mm
             coaster       mug centre within 3 cm of the coaster centre, upright (tilt < 25.8 deg), resting
             pen holder    lower end inside the cup and on its floor (within 2.4 cm), centre within 1 cm of the cup,
                           tilt from vertical <= 40 deg
             pile of books centre at least 2 cm inside the top of the pile, lying on it (height within 6 mm), flat
                           (tilt < 8.1 deg)
  T(s)     tidiness of a desk state: the share of O that is Placed.  T0 at the start (= |O_in| / |O|), T1 at the end.
  G        tidiness gain, the share of the initial mess removed, net of damage: (T1 - T0) / (1 - T0)
           = (placed things of O_out - displaced things of O_in) / |O_out|.  1 = all fixed, nothing broken.
  P, K     placement rate on O_out, keep rate on O_in.
  S        whole-desk success: T1 = 1 and the gripper empty.
  residual for a thing not Placed: how far its centre is outside the accepted area of its home (0 when the centre is
           inside and another condition failed: not resting, tilted).
Writes results/dev_stats/tidy.csv (one row per desk) and tidy_objects.csv (one row per thing), prints a summary.
Usage: python3 scripts/tidy_metrics.py"""
import csv
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "dev_stats"
sys.path.insert(0, str(ROOT))


def _stub_robosuite():
    """The task generator imports robosuite for the scene classes; the layouts need none of it."""
    class Dummy:
        def __init__(self, *a, **k):
            pass
    names = ["Lift", "ManipulationEnv", "TableArena", "ManipulationTask", "new_geom", "SequentialCompositeSampler",
             "UniformRandomSampler", "MujocoXMLObject", "CompositeObject", "BoxObject", "CylinderObject",
             "MujocoObject", "array_to_string", "new_body", "new_joint", "new_site", "CustomMaterial", "find_elements",
             "string_to_array", "add_prefix"]
    for m in ["robosuite", "robosuite.environments", "robosuite.environments.manipulation",
              "robosuite.environments.manipulation.lift", "robosuite.environments.manipulation.manipulation_env",
              "robosuite.models", "robosuite.models.arenas", "robosuite.models.tasks", "robosuite.utils",
              "robosuite.utils.mjcf_utils", "robosuite.utils.placement_samplers", "robosuite.models.objects", "mujoco"]:
        if m not in sys.modules:
            mod = types.ModuleType(m)
            for n in names:
                setattr(mod, n, Dummy)
            sys.modules[m] = mod


# Paired skill-memory vs no-memory runs (as in scripts/dev_figures.py) and the SAM2 desks
PAIRS = {
    "Gemini colour 20+20": (["actgemini_realdesk2_none_20261104T144951Z", "actgemini_realdesk3_none_20261104T155202Z",
                             "actgemini_realdesk4_none10_20261104T164331Z"],
                            ["actgemini_realdesk3_skill10_20261104T155202Z", "actgemini_realdesk4_skill10_20261104T164331Z"]),
    "Gemini + GraspNet/cuRobo 10+10": (["actgemini_gpu2_realdesk_none10_20261104T182458Z"],
                                       ["actgemini_gpu3_realdesk_skill10_20261104T184856Z"]),
    "DeepSeek SAM2 3+3": (["ds_cmp_none_20261106T115457Z"], ["ds_cmp_skill_20261106T115457Z"]),
}
SAM2_COMMON = {"sdesk_vague_anchors_17100", "sdesk_vague_anchors_17101", "sdesk_vague_anchors_17102"}  # both groups done


def residual(pos, zone, style, half):
    """Distance (m) of the thing's centre outside the accepted area of its home, 0 inside."""
    c, yaw = np.asarray(zone.xy, float), float(getattr(zone, "yaw", 0.0) or 0.0)
    d = np.asarray(pos[:2], float) - c
    if style == "bin":
        return max(0.0, float(np.hypot(*d)) - (half[0] - .005))
    a = np.array([np.cos(yaw), np.sin(yaw)]) @ d, np.array([-np.sin(yaw), np.cos(yaw)]) @ d
    margin = {"coaster": .015, "pile": .02, "holder": -.01}.get(style, .005)
    over = [max(0.0, abs(a[i]) - (half[i] - margin)) for i in (0, 1)]
    return float(np.hypot(*over))


def layout_for(task_id):
    """Rebuild a SAM2 desk's layout from its task id (sdesk only)."""
    from sma.env.tasks import make_sdesk_task
    seed = int(task_id.rsplit("_", 1)[1])
    return make_sdesk_task(seed, varied="varied" in task_id)


def desk_metrics(e, with_residual=False):
    meta, ev = e.get("task_meta") or {}, (e.get("private_eval") or {})
    objs = ev.get("objects") or {}
    out_, in_ = list(meta.get("out_of_place") or []), list(meta.get("already_tidy") or [])
    if not objs or not (out_ or in_):
        return None, []

    def placed(o):
        d = objs.get(o) or {}
        return bool(d.get("inside") and d.get("resting") and d.get("upright"))
    n = len(out_) + len(in_)
    p_out, k_in = sum(map(placed, out_)), sum(map(placed, in_))
    t0, t1 = len(in_) / n, (p_out + k_in) / n
    row = {"task": e.get("task_id"), "things": n, "out_of_place": len(out_), "already_tidy": len(in_),
           "placed": p_out, "kept": k_in, "T0": round(t0, 3), "T1": round(t1, 3),
           "G": round((t1 - t0) / (1 - t0), 3) if t0 < 1 else None,
           "S": bool(ev.get("success")), "gripper_holding": bool(ev.get("gripper_holding"))}
    things = []
    task = None
    if with_residual:
        from sma.env.scene import RECEPTACLES, receptacle_half
        task = layout_for(e["task_id"])
        zones = {z.name: z for z in task.layout.zones}
        assert all(task.goal.get(o) == objs[o]["zone"] for o in objs), "rebuilt layout differs from the run"
    for o in out_ + in_:
        d = objs.get(o) or {}
        t = {"task": e.get("task_id"), "thing": o, "kind": o.split("_", 1)[1], "start": "out" if o in out_ else "in",
             "home": d.get("zone"), "placed": placed(o), "inside": d.get("inside"), "resting": d.get("resting"),
             "upright": d.get("upright"), "tilt_deg": d.get("tilt_from_vertical_deg")}
        if task is not None and d.get("position"):
            z = zones[d["zone"]]
            style = RECEPTACLES[z.color][0]
            t["residual_cm"] = round(100 * residual(d["position"], z, style, receptacle_half(style, z)), 1)
        things.append(t)
    return row, things


def main():
    _stub_robosuite()
    OUT.mkdir(parents=True, exist_ok=True)
    eps = list(csv.DictReader(open(OUT / "episodes.csv")))
    rows, things = [], []
    # SAM2 desks: every finished one; "main" = up to 2026-11-06 (before the post-comparison fixes), "later" = 11-08
    for r in eps:
        if r["family"] != "sdesk" or r["success_gt"] not in ("True", "False"):
            continue
        e = json.load(open(ROOT / "results" / r["path"] / "episode.json"))
        row, th = desk_metrics(e, with_residual=True)
        if row:
            group = "SAM2 main (<=11-06)" if r["date"] <= "2026-11-06" else "SAM2 later (11-08)"
            rows.append({"group": group, "condition": r["condition"], "run": r["run"], **row})
            things += [{"group": group, "condition": r["condition"], "run": r["run"], **t} for t in th]
    # Paired memory comparisons: only desks both groups finished
    for name, (none_runs, skill_runs) in PAIRS.items():
        per = {}
        for cond, runs in (("none", none_runs), ("skill", skill_runs)):
            for r in eps:
                if r["run"] in runs and r["success_gt"] in ("True", "False"):
                    per.setdefault(cond, {})[r["task"]] = r
        common = set(per.get("none", {})) & set(per.get("skill", {}))
        if name.startswith("DeepSeek"):
            common &= SAM2_COMMON
        for cond in ("none", "skill"):
            for task in sorted(common):
                r = per[cond][task]
                row, _ = desk_metrics(json.load(open(ROOT / "results" / r["path"] / "episode.json")))
                if row:
                    rows.append({"group": "pair: " + name, "condition": cond, "run": r["run"], **row})
    for fname, data in (("tidy.csv", rows), ("tidy_objects.csv", things)):
        keys = list(dict.fromkeys(k for d in data for k in d))
        with open(OUT / fname, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(data)
    summary = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = r["group"] if r["group"].startswith("SAM2") else f'{r["group"]} [{r["condition"]}]'
        for m in ("T0", "T1", "G", "S", "placed", "out_of_place", "kept", "already_tidy", "things"):
            summary[key][m].append(r[m])
    report = {}
    for key, m in summary.items():
        n = len(m["T0"])
        report[key] = {"desks": n, "T0": round(float(np.mean(m["T0"])), 3), "T1": round(float(np.mean(m["T1"])), 3),
                       "G": round(float(np.mean([g for g in m["G"] if g is not None])), 3),
                       "S": f'{sum(m["S"])}/{n}', "P": f'{sum(m["placed"])}/{sum(m["out_of_place"])}',
                       "K": f'{sum(m["kept"])}/{sum(m["already_tidy"])}',
                       "G_min": round(float(min(g for g in m["G"] if g is not None)), 3)}
    fails = [t for t in things if not t["placed"] and t["group"].startswith("SAM2 main")]
    report["SAM2 main: things not placed"] = [
        {k: t.get(k) for k in ("task", "condition", "kind", "start", "inside", "resting", "upright", "residual_cm")}
        for t in fails]
    (OUT / "tidy_summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

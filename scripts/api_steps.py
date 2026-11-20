"""Which steps of a robot program are repeated from desk to desk, and which of them the skill memory saves? Every
act-level attempt's sandbox trace (episode.json) is read as a sequence of API calls, grouped into stages:
  candidates   grasp_candidates (geometry, or Contact-GraspNet when the service runs)
  grasp        grasp, close_gripper
  reorient     solve_ik, move_gripper, rotate_by (6-DoF: turning the held object, e.g. a pen upright)
  transport    carry_to (straight line, or a cuRobo plan when cuRobo runs)
  place        lower_until_contact, move_by, open_gripper
For each kind of object: how often programs contain each stage, at which stage a failed attempt stopped (the call that
raised the error) or, for a program that ran to the end, whether the result was wrong; per condition (no memory /
skill memory) on the paired runs, and over all act-level runs. No simulator, no LLM.
Usage: python3 scripts/api_steps.py   (writes results/dev_stats/api_steps.json)"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
STAGE = {"grasp_candidates": "candidates", "grasp": "grasp", "close_gripper": "grasp",
         "solve_ik": "reorient", "move_gripper": "reorient", "rotate_by": "reorient",
         "carry_to": "transport", "lower_until_contact": "place", "move_by": "place", "open_gripper": "place"}
STAGES = ["candidates", "grasp", "reorient", "transport", "place"]
PAIRED = {"none": ["actgemini_realdesk2_none_20261104T144951Z", "actgemini_realdesk3_none_20261104T155202Z",
                   "actgemini_realdesk4_none10_20261104T164331Z", "actgemini_gpu2_realdesk_none10_20261104T182458Z",
                   "ds_cmp_none_20261106T115457Z"],
          "skill": ["actgemini_realdesk3_skill10_20261104T155202Z", "actgemini_realdesk4_skill10_20261104T164331Z",
                    "actgemini_gpu3_realdesk_skill10_20261104T184856Z", "ds_cmp_skill_20261106T115457Z"]}


def kind_of(name, e):
    ident = e.get("identification") or {}
    if name in ident:
        cat = str(ident[name].get("category", "?"))
        return {"crumpled paper ball": "paper", "drink can": "can", "sticky notes": "sticky", "tape roll": "tape",
                "pencil": "pen"}.get(cat, cat)
    return name.rsplit("_", 1)[-1] if "_" in name else name


def flatten(trace):
    """The primitive API calls in order, with the helper or skill each came from (calls made inside a helper or skill
    appear in the trace too, after the helper's own entry)."""
    return [t for t in trace if not t["call"].startswith(("helpers.", "skills."))]


def attempt_row(a, kind, cond):
    trace = a.get("trace") or []
    prims = flatten(trace)
    stages = [STAGE[t["call"]] for t in prims if t["call"] in STAGE]
    ok = bool((a.get("check") or {}).get("satisfied"))
    err = a.get("error") if isinstance(a.get("error"), dict) else None
    stop = None
    if err and err.get("type") == "robot_error":
        # the call that raised: the last primitive in the trace without a result
        last = next((t for t in reversed(prims) if "result" not in t and t["call"] in STAGE), None)
        stop = STAGE.get(last["call"]) if last else "other"
    elif err:
        stop = "program error"
    elif not ok:
        stop = "ran, wrong result" if a.get("code") is not None else "not run"
    grasp = next((t for t in prims if t["call"] == "grasp" and t.get("args")), None)
    gsrc = None
    if grasp and isinstance(grasp["args"][0], dict):
        gsrc = str(grasp["args"][0].get("source", "?")).split(" ")[0]
    carry = [t for t in prims if t["call"] == "carry_to" and isinstance(t.get("result"), dict)]
    return {"kind": kind, "cond": cond, "ok": ok, "direct": bool(a.get("direct_reuse")),
            "has_code": a.get("code") is not None, "stages": stages, "stop": stop,
            "reason": err and (err.get("reason") or err.get("exception")), "grasp_source": gsrc,
            "curobo_carry": any("curobo" in str(t["result"].get("planner", "")).lower() and
                                not str(t["result"].get("planner", "")).startswith("straight") for t in carry),
            "helpers": [t["call"][8:] for t in trace if t["call"].startswith("helpers.")],
            "n_calls": len(prims)}


def collect(runs, cond):
    rows = []
    for run in runs:
        for f in sorted((RES / run).glob(f"**/{cond}/*/episode.json")):
            e = json.loads(f.read_text())
            for s in e.get("subgoals") or []:
                k = kind_of(s.get("object", ""), e)
                for a in s.get("attempts") or []:
                    if a.get("code") is None and not a.get("direct_reuse"):
                        continue  # skipped (object not seen)
                    rows.append(attempt_row(a, k, cond))
    return rows


def table(rows, kinds):
    out = {}
    for k in kinds:
        rs = [r for r in rows if r["kind"] == k]
        if not rs:
            continue
        has = {st: sum(st in r["stages"] for r in rs) for st in STAGES}
        stop = Counter(r["stop"] for r in rs if not r["ok"])
        out[k] = {"attempts": len(rs), "ok": sum(r["ok"] for r in rs), "direct": sum(r["direct"] for r in rs),
                  "with_stage": has, "failed_at": dict(stop),
                  "ok_with_reorient": sum(r["ok"] for r in rs if "reorient" in r["stages"]),
                  "with_reorient": sum("reorient" in r["stages"] for r in rs),
                  "ok_without_reorient": sum(r["ok"] for r in rs if "reorient" not in r["stages"]),
                  "grasp_source": {f"{src} {'ok' if ok else 'failed'}": n for (src, ok), n in
                                   Counter((r["grasp_source"], r["ok"]) for r in rs if r["grasp_source"]).items()},
                  "curobo_carry": sum(r["curobo_carry"] for r in rs),
                  "reasons": dict(Counter(r["reason"] for r in rs if r["reason"]).most_common(5)),
                  "helpers": dict(Counter(h for r in rs for h in r["helpers"]).most_common(6))}
    return out


def main():
    paired = {c: collect(runs, c) for c, runs in PAIRED.items()}
    kinds = ["pen", "mug", "paper", "book", "notebook", "block"]
    result = {"paired": {c: table(rows, kinds) for c, rows in paired.items()}}
    # every act-level attempt (all runs, all conditions), for the stage at which each kind fails
    allrows = []
    for r in csv.DictReader(open(RES / "dev_stats" / "episodes.csv")):
        if r["api_level"] != "act":
            continue
        f = RES / r["path"] / "episode.json"
        try:
            e = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        cond = "skill" if r["condition"] in ("skill", "both", "memory") else "none"
        for s in e.get("subgoals") or []:
            k = kind_of(s.get("object", ""), e)
            for a in s.get("attempts") or []:
                if a.get("code") is None and not a.get("direct_reuse"):
                    continue
                allrows.append(attempt_row(a, k, cond))
    result["all_act"] = table(allrows, kinds + ["can", "eraser", "sticky", "tape", "marker"])
    (RES / "dev_stats" / "api_steps.json").write_text(json.dumps(result, indent=1))
    for c in ("none", "skill"):
        print(f"\n===== paired runs, {c}")
        for k, v in result["paired"][c].items():
            print(f"{k:9s} attempts {v['attempts']:3d} ok {v['ok']:3d} direct {v['direct']:3d} | with stage {v['with_stage']}"
                  f" | reorient ok {v['ok_with_reorient']}/{v['with_reorient']}, without {v['ok_without_reorient']}/"
                  f"{v['attempts'] - v['with_reorient']} | failed at {v['failed_at']}")
            print(f"          grasp source (src, ok): {v['grasp_source']} curobo carries {v['curobo_carry']} reasons {v['reasons']}")
            if v["helpers"]:
                print(f"          helpers {v['helpers']}")
    print("\n===== all act-level attempts")
    for k, v in result["all_act"].items():
        print(f"{k:9s} attempts {v['attempts']:4d} ok {v['ok']:4d} | reorient ok {v['ok_with_reorient']}/{v['with_reorient']}"
              f" | failed at {v['failed_at']} | reasons {v['reasons']}")


if __name__ == "__main__":
    main()

"""Where does the skill memory save LLM calls? On the paired skill-memory vs no-memory runs (the same desks), every LLM
call is counted by its purpose (from the call logs: identify, refine, plan, replan, code, reflect, extract, ...), and the
per-object work by kind and attempt (episode.json): attempts, programs the LLM wrote, stored skills executed directly,
first-attempt successes. No simulator, no LLM. Writes results/dev_stats/llm_savings.json and figures in docs/figures/dev/.
Usage: python3 scripts/llm_savings.py"""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "dev_stats"
FIG = ROOT / "docs" / "figures" / "dev"

PAIRS = {  # skill memory vs no memory (as in scripts/dev_figures.py), plus the new desks
    "Gemini colour 20+20": (["actgemini_realdesk2_none_20261104T144951Z", "actgemini_realdesk3_none_20261104T155202Z",
                             "actgemini_realdesk4_none10_20261104T164331Z"],
                            ["actgemini_realdesk3_skill10_20261104T155202Z", "actgemini_realdesk4_skill10_20261104T164331Z"]),
    "Gemini + GraspNet/cuRobo 10+10": (["actgemini_gpu2_realdesk_none10_20261104T182458Z"],
                                       ["actgemini_gpu3_realdesk_skill10_20261104T184856Z"]),
    "DeepSeek SAM2 3+3": (["ds_cmp_none_20261106T115457Z"], ["ds_cmp_skill_20261106T115457Z"]),
    "DeepSeek new desks 3+3": (["xdesk_par_20261111T1440Z", "xdesk_none2_20261111T153824Z"],
                               ["xdesk_run_20261111T121104Z", "xdesk_par_20261111T1440Z"]),
}
PURPOSES = ["identify", "refine", "plan", "replan", "code", "reflect", "extract", "other"]


def episodes(runs, condition):
    """{task: (episode dir, order in its run)} for the finished episodes of this condition in these runs."""
    out = {}
    for run in runs:
        for summ in sorted((RES / run).glob(f"**/{condition}/summary.json")):
            tasks = json.loads(summ.read_text()).get("tasks", [])
            for i, t in enumerate(tasks):
                d = summ.parent / t["task"]
                if (d / "episode.json").exists() and t.get("success_gt") is not None and not t.get("harness_error"):
                    out[t["task"]] = (d, i)
    return out


def calls_by_purpose(d):
    c = defaultdict(int)
    for f in (d / "llm").glob("call_*.json"):
        p = re.sub(r"^call_\d+_", "", f.stem)
        c[p if p in PURPOSES else "other"] += 1
    return c


def kind_of(name, e):
    ident = e.get("identification") or {}
    if name in ident:
        cat = str(ident[name].get("category", "?"))
        return {"crumpled paper ball": "paper", "drink can": "can", "sticky notes": "sticky", "tape roll": "tape",
                "pencil": "pen"}.get(cat, cat)
    return name.rsplit("_", 1)[-1] if "_" in name else name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, (none_runs, skill_runs) in PAIRS.items():
        eps = {"none": episodes(none_runs, "none"), "skill": episodes(skill_runs, "skill")}
        common = sorted(set(eps["none"]) & set(eps["skill"]))
        if name.startswith("DeepSeek SAM2"):
            common = [t for t in common if t.endswith(("17100", "17101", "17102"))]  # finished by both (17103: cap)
        r = {"desks": len(common)}
        for cond in ("none", "skill"):
            by_p = defaultdict(int)
            kinds = defaultdict(lambda: {"subgoals": 0, "attempts": 0, "programs": 0, "direct": 0, "direct_ok": 0,
                                         "first_ok": 0, "failed_attempts": 0, "ok": 0})
            per_desk = []
            for t in common:
                d, order = eps[cond][t]
                e = json.loads((d / "episode.json").read_text())
                c = calls_by_purpose(d)
                for p, v in c.items():
                    by_p[p] += v
                per_desk.append({"task": t, "order": order, "calls": sum(c.values()), **dict(c)})
                for s in e.get("subgoals") or []:
                    k = kinds[kind_of(s.get("object", ""), e)]
                    att = s.get("attempts") or []
                    k["subgoals"] += 1
                    k["attempts"] += len(att)
                    k["programs"] += sum(1 for a in att if a.get("code") is not None and not a.get("direct_reuse"))
                    k["direct"] += sum(bool(a.get("direct_reuse")) for a in att)
                    k["direct_ok"] += sum(bool(a.get("direct_reuse")) and bool((a.get("check") or {}).get("satisfied"))
                                          for a in att)
                    k["first_ok"] += bool(att and (att[0].get("check") or {}).get("satisfied") and not att[0].get("error"))
                    k["failed_attempts"] += sum(1 for a in att if not (a.get("check") or {}).get("satisfied")
                                                and a.get("code") is not None)
                    k["ok"] += bool(s.get("success"))
            n = max(1, len(common))
            r[cond] = {"calls_per_desk": round(sum(by_p.values()) / n, 1),
                       "by_purpose_per_desk": {p: round(by_p.get(p, 0) / n, 2) for p in PURPOSES},
                       "by_kind": {k: v for k, v in sorted(kinds.items())}, "per_desk": per_desk}
        r["saving_by_purpose_per_desk"] = {p: round(r["none"]["by_purpose_per_desk"][p] - r["skill"]["by_purpose_per_desk"][p], 2)
                                           for p in PURPOSES}
        report[name] = r
    (OUT / "llm_savings.json").write_text(json.dumps(report, indent=1))
    for name, r in report.items():
        print(f"\n== {name}: {r['desks']} desks; calls/desk none {r['none']['calls_per_desk']} -> skill {r['skill']['calls_per_desk']}")
        print("   per desk by purpose (none -> skill):",
              {p: (r["none"]["by_purpose_per_desk"][p], r["skill"]["by_purpose_per_desk"][p]) for p in PURPOSES
               if r["none"]["by_purpose_per_desk"][p] or r["skill"]["by_purpose_per_desk"][p]})
        for cond in ("none", "skill"):
            for k, v in r[cond]["by_kind"].items():
                if v["subgoals"] >= 2:
                    print(f"   {cond:5s} {k:9s} subgoals {v['subgoals']:3d} attempts/sub {v['attempts']/v['subgoals']:.2f} "
                          f"programs/sub {v['programs']/v['subgoals']:.2f} direct {v['direct_ok']}/{v['direct']} "
                          f"first-try ok {v['first_ok']}/{v['subgoals']} failed attempts/sub {v['failed_attempts']/v['subgoals']:.2f}")
    return report


def figures(report):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dev_figures import GRID, INK, INK2, NEUTRAL, SLOTS, SURFACE, style
    FIG.mkdir(parents=True, exist_ok=True)

    # 1. calls per desk by purpose, no memory vs skill memory, per setting
    parts = [("Identify, infer goals, plan (per desk)", ("identify", "refine", "plan", "other"), NEUTRAL),
             ("Re-plan", ("replan",), SLOTS[3]), ("Write a program", ("code",), SLOTS[0]),
             ("Reflect on a failure", ("reflect",), SLOTS[1]), ("Extract a skill (learning)", ("extract",), SLOTS[2])]
    labels = {"Gemini colour 20+20": "Gemini, colour perception\n20 + 20 desks",
              "Gemini + GraspNet/cuRobo 10+10": "Gemini + GraspNet/cuRobo\n10 + 10 desks",
              "DeepSeek SAM2 3+3": "DeepSeek, SAM2 perception\n3 + 3 desks",
              "DeepSeek new desks 3+3": "DeepSeek, new desks and objects\n3 + 3 desks"}
    fig = plt.figure(figsize=(9, 5.4), dpi=200, facecolor=SURFACE)
    ax = fig.add_axes([.27, .21, .70, .60])
    style(ax)
    y, ticks, names = 0, [], []
    for name, r in report.items():
        for cond, tag in (("none", "No memory"), ("skill", "Skill memory")):
            left = 0.0
            for _, keys, col in parts:
                v = sum(r[cond]["by_purpose_per_desk"].get(k, 0) for k in keys)
                ax.barh(y, v, left=left, color=col, edgecolor=SURFACE, linewidth=1.2, height=.72)
                left += v
            ax.text(left + .6, y, f"{left:.1f}", va="center", fontsize=8.5, color=INK)
            ticks.append(y)
            names.append(f"{labels[name]}  ·  {tag}" if cond == "none" else tag)
            y += 1
        y += .6
    ax.set_yticks(ticks)
    ax.set_yticklabels(names, fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("LLM calls per desk", fontsize=9, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts]
    fig.legend(handles, [p[0] for p in parts], loc="lower center", ncol=3, fontsize=7.8, frameon=False,
               bbox_to_anchor=(.6, .0))
    fig.text(.03, .93, "Where skill memory saves LLM calls: fewer programs and fewer reflections", fontsize=12.5,
             color=INK, weight="bold")
    fig.text(.03, .875, "Same desks with and without memory. Per-desk work (recognition, goals, plan) is unchanged; "
             "the saving is in the per-object retry loop.", fontsize=8.3, color=INK2)
    fig.savefig(FIG / "llm_calls_by_purpose.png", facecolor=SURFACE)
    plt.close(fig)

    # 2. program + reflection calls per desk by kind of object (Gemini colour 20+20, the largest pair)
    r = report["Gemini colour 20+20"]
    n = r["desks"]
    kinds = [k for k in ("pen", "paper", "mug", "book", "block")]
    nice = {"pen": "Pen → stand in holder", "paper": "Paper ball → wastebasket", "mug": "Mug → coaster",
            "book": "Book → tray", "block": "Toy block → box"}
    fig = plt.figure(figsize=(8, 4.8), dpi=200, facecolor=SURFACE)
    ax = fig.add_axes([.26, .12, .62, .62])
    style(ax)
    for i, k in enumerate(kinds):
        for j, (cond, col) in enumerate((("none", NEUTRAL), ("skill", SLOTS[0]))):
            v = r[cond]["by_kind"].get(k, {})
            calls = (v.get("programs", 0) + v.get("failed_attempts", 0)) / n
            yy = i + (j - .5) * .36
            ax.barh(yy, calls, color=col, height=.34, edgecolor=SURFACE, linewidth=1)
            first = f"first try {v.get('first_ok', 0)}/{v.get('subgoals', 0)}"
            ax.text(calls + .1, yy, f"{calls:.1f}   ({first})", va="center", fontsize=7.6, color=INK2)
    ax.set_yticks(range(len(kinds)))
    ax.set_yticklabels([nice[k] for k in kinds], fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 10.5)
    ax.set_xlabel("Program + reflection calls per desk spent on this kind of object", fontsize=8.5, color=INK2)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=NEUTRAL), plt.Rectangle((0, 0), 1, 1, color=SLOTS[0])],
               ["No memory", "Skill memory"], loc="upper right", ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(.97, .80))
    fig.text(.03, .93, "Most of the saving is the hard, recurring sub-task: standing a pen in its holder", fontsize=12,
             color=INK, weight="bold")
    fig.text(.03, .875, "Gemini, colour perception, 20 + 20 desks. Without memory the pen is re-solved on every desk "
             "and never succeeds first (0/28);", fontsize=8, color=INK2)
    fig.text(.03, .84, "the stored skill succeeds first in 13/19. Routine tasks (mug, paper ball) save about one "
             "program each; easy ones (book) save nothing.", fontsize=8, color=INK2)
    fig.savefig(FIG / "llm_calls_by_object.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    figures(main())

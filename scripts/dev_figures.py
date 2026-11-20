"""Slide figures and numbers from the development record (run scripts/dev_stats.py first). No simulator, no LLM.
Writes results/dev_stats/numbers.json and docs/figures/dev/*.png (English, 16:9).
Settings compared (the harness changed between them, so they are stages of development, not a controlled ablation):
  A  mid-level API, Qwen3-30B (local), colour perception          runs of 2026-10-17 with api_level "mid"
  B  act-level API, Gemini 3.8 Flash, colour perception            2026-11-04/05
  C  act-level API, DeepSeek, realistic desk with SAM2 perception  sdesk desks up to 2026-11-06 (the 2026-11-08 runs
     use the fixes made after the skill-vs-none comparison, which the report leaves out)
Usage: python3 scripts/dev_figures.py"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STATS, FIG = ROOT / "results" / "dev_stats", ROOT / "docs" / "figures" / "dev"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # validated order
NEUTRAL = "#b9b8b2"
SETTINGS = {"A": "Mid-level API\nQwen3-30B, colour perception",
            "B": "Act-level API\nGemini 3.8 Flash, colour perception",
            "C": "Act-level API\nDeepSeek, SAM2 real perception"}
# Failure causes, folded to seven for the chart (fixed order = fixed colours)
FOLD = {"goal not met": "Ran, but the check failed", "blocked by contact": "Blocked by contact",
        "pose out of reach": "Out of reach / limits", "argument out of range": "Out of reach / limits",
        "grasp failed": "Grasp failed", "object not seen": "Object not seen / lost",
        "lost from view after acting": "Object not seen / lost", "program bug": "Program bug / rejected",
        "rejected by sandbox": "Program bug / rejected", "program gave up": "Program bug / rejected",
        "wrong robot state": "Other", "other": "Other"}
CAUSES = ["Ran, but the check failed", "Blocked by contact", "Out of reach / limits", "Grasp failed",
          "Object not seen / lost", "Program bug / rejected", "Other"]
PAIRS = {  # skill memory vs no memory on the same desks (runs as listed in docs/REALDESK_TASK_zh.md)
    "Gemini\ncolour\n20 + 20 desks": (["actgemini_realdesk2_none_20261104T144951Z", "actgemini_realdesk3_none_20261104T155202Z",
                                            "actgemini_realdesk4_none10_20261104T164331Z"],
                                           ["actgemini_realdesk3_skill10_20261104T155202Z", "actgemini_realdesk4_skill10_20261104T164331Z"]),
    "Gemini\n+GraspNet/cuRobo\n10 + 10 desks": (["actgemini_gpu2_realdesk_none10_20261104T182458Z"],
                                                     ["actgemini_gpu3_realdesk_skill10_20261104T184856Z"]),
    "DeepSeek\nSAM2\n3 + 3 desks": (["ds_cmp_none_20261106T115457Z"], ["ds_cmp_skill_20261106T115457Z"]),
}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def setting(r):
    if r["date"] == "2026-10-17" and r["api_level"] == "mid":
        return "A"
    if r["api_level"] == "act" and "gemini-3.8" in r["model"]:
        return "B"
    if r["api_level"] == "act" and "deepseek" in r["model"] and r["family"] == "sdesk" and r["date"] <= "2026-11-06":
        return "C"
    return None


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)


def figure(w=8, h=4.5):
    fig = plt.figure(figsize=(w, h), dpi=200, facecolor=SURFACE)
    return fig


def hbar_stack(ax, rows, parts, colours, labels=True):
    """100% horizontal stacked bars: rows = [(label, Counter)], parts in fixed order."""
    for i, (label, c) in enumerate(rows):
        total, left = sum(c[p] for p in parts), 0.0
        for p, col in zip(parts, colours):
            w = 100 * c[p] / total if total else 0
            ax.barh(i, w, left=left, color=col, edgecolor=SURFACE, linewidth=1.5, height=.62)
            if labels and w >= 7:
                ax.text(left + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=8.5,
                        color="white" if col not in ("#eda100", "#e87ba4", NEUTRAL, "#1baf7a") else INK)
            left += w
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    eps = list(csv.DictReader(open(STATS / "episodes.csv")))
    att = list(csv.DictReader(open(STATS / "attempts.csv")))
    sub = list(csv.DictReader(open(STATS / "subgoals.csv")))
    out = {}

    # Scale of the record
    runs = sorted({p.name for p in (ROOT / "results").iterdir() if p.is_dir() and p.name not in ("dev_stats", "demo")})
    agent_runs = {e["run"] for e in eps}
    out["scale"] = {"result_directories": len(runs), "directories_with_agent_episodes": len(agent_runs),
                    "directories_without_episodes_gates_probes_previews": len(set(runs) - agent_runs),
                    "desk_episodes": len(eps), "subgoals": len(sub), "attempts": len(att),
                    "programs_run": sum(a["has_code"] == "True" for a in att),
                    "llm_calls_in_episodes": int(sum(num(e["llm_calls"]) for e in eps)),
                    "completion_tokens_in_episodes": int(sum(num(e["completion_tokens"]) for e in eps)),
                    "episode_cost_usd": round(sum(num(e["cost_usd"]) for e in eps), 2),
                    "desks_by_date": dict(sorted(Counter(e["date"] for e in eps).items())),
                    "desks_by_model": dict(Counter(e["model"] or "scripted / unrecorded" for e in eps).most_common())}

    # 1. Where attempts fail, per setting
    causes = defaultdict(Counter)
    totals = defaultdict(Counter)
    for a in att:
        s = setting(a)
        if s:
            totals[s][a["outcome"] == "success"] += 1
            if a["outcome"] != "success":
                causes[s][FOLD.get(a["outcome"], "Other")] += 1
    out["failure_causes"] = {s: {"attempts": sum(totals[s].values()), "successful_attempts": totals[s][True],
                                 "failed_attempts": totals[s][False], "causes": dict(causes[s].most_common())}
                             for s in SETTINGS}
    fig = figure(9, 4.4)
    ax = fig.add_axes([.31, .22, .66, .6])
    style(ax)
    rows = [(f"{SETTINGS[s]}\n({sum(causes[s].values())} failed attempts)", causes[s]) for s in SETTINGS]
    colours = SLOTS[:6] + [NEUTRAL]
    hbar_stack(ax, rows, CAUSES, colours)
    used = [(c, n) for c, n in zip(colours, CAUSES) if any(r[1][n] for r in rows)]
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in used], labels=[n for _, n in used], ncol=3, fontsize=8,
              frameon=False, loc="upper left", bbox_to_anchor=(-.44, -.1), labelcolor=INK2)
    fig.text(.02, .93, "Why attempts failed, at each stage of development", fontsize=13, color=INK, weight="bold")
    fig.text(.02, .88, "Share of failed attempts by cause (each attempt = one generated program run on the robot)",
             fontsize=9, color=INK2)
    fig.savefig(FIG / "failure_causes.png", facecolor=SURFACE)
    plt.close(fig)

    # 2. Retries: subgoals reached on the first attempt, on a retry, or not at all
    rec = defaultdict(Counter)
    for s in sub:
        k = setting(s)
        if not k:
            continue
        n = s["success_attempt"]
        rec[k]["first" if n == "1" else "retry" if n else "not"] += 1
    out["retries"] = {k: {"subgoals": sum(v.values()), "first_attempt": v["first"], "after_retry": v["retry"],
                          "not_reached": v["not"],
                          "first_attempt_rate": round(v["first"] / sum(v.values()), 3),
                          "final_rate": round((v["first"] + v["retry"]) / sum(v.values()), 3)} for k, v in rec.items()}
    fig = figure(9, 3.8)
    ax = fig.add_axes([.31, .24, .66, .54])
    style(ax)
    parts = ["first", "retry", "not"]
    names = ["Reached on the first attempt", "Reached after a retry (diagnosis + reflection)", "Not reached"]
    hbar_stack(ax, [(f"{SETTINGS[k]}\n({sum(rec[k].values())} subgoals)", rec[k]) for k in SETTINGS], parts,
               [SLOTS[0], SLOTS[2], NEUTRAL])
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (SLOTS[0], SLOTS[2], NEUTRAL)], labels=names,
              ncol=3, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(-.44, -.14), labelcolor=INK2)
    fig.text(.02, .92, "Retries recover a share of subgoals", fontsize=13, color=INK, weight="bold")
    fig.text(.02, .86, "Subgoal = one object to one place; up to 3 attempts, judged by the robot's own camera check",
             fontsize=9, color=INK2)
    fig.savefig(FIG / "retries.png", facecolor=SURFACE)
    plt.close(fig)

    # 3. The robot's camera check against the offline ground truth (last entry per object in each desk)
    conf = {"colour": Counter(), "sam2": Counter()}
    seen = set()
    for f in sorted((ROOT / "results").glob("**/episode.json")):
        digest = hashlib.sha1(f.read_bytes()).hexdigest()
        if digest in seen:  # a copy of an episode already counted
            continue
        seen.add(digest)
        e = json.load(open(f))
        ev = (e.get("private_eval") or {}).get("objects") or {}
        if not ev or not e.get("subgoals"):
            continue
        models = str((e.get("llm_usage") or {}).get("models"))
        mode = "sam2" if e.get("family") == "sdesk" else "colour"
        if mode == "sam2" and "20261108" in str(f):  # after the post-comparison fixes (see the docstring)
            continue
        if mode == "colour" and not any(m in models for m in ("qwen", "gemini")) and "deepseek" not in models:
            pass  # scripted or unrecorded runs are kept: the check is the harness's, not the model's
        meta, ident = e.get("task_meta") or {}, e.get("identification") or {}
        inv = {v: k for k, v in (meta.get("perceived_name") or {}).items()}
        cat = meta.get("category") or {}
        last = {s["object"]: s for s in e["subgoals"]}
        for o, s in last.items():
            priv = o if o in ev else inv.get(o) if inv.get(o) in ev else None
            if priv is None and o in ident and cat:
                cands = [p for p in meta.get("out_of_place", []) if cat.get(p) == ident[o].get("category")]
                priv = cands[0] if len(cands) == 1 else None
            if priv is None:
                conf[mode]["unmapped"] += 1
                continue
            g = ev[priv]
            conf[mode][(bool(s["success"]), bool(g.get("inside") and g.get("resting") and g.get("upright")))] += 1
    out["camera_check_vs_ground_truth"] = {}
    for mode, c in conf.items():
        tp, fp, fn, tn = c[(True, True)], c[(True, False)], c[(False, True)], c[(False, False)]
        out["camera_check_vs_ground_truth"][mode] = {
            "both_pass": tp, "camera_pass_truth_fail": fp, "camera_fail_truth_pass": fn, "both_fail": tn,
            "unmatched": c["unmapped"], "agreement": round((tp + tn) / (tp + fp + fn + tn), 3),
            "precision": round(tp / (tp + fp), 3), "recall": round(tp / (tp + fn), 3)}
    fig = figure(9, 3.6)
    ax = fig.add_axes([.31, .26, .66, .48])
    style(ax)
    keys = [(True, True), (False, False), (True, False), (False, True)]
    names = ["Both say placed", "Both say not placed", "Camera: placed, truth: not (false accept)",
             "Camera: not, truth: placed (false reject)"]
    rows = []
    for mode, label in (("colour", "Colour perception\n(desks of Oct 17 - Nov 5)"), ("sam2", "SAM2 real perception\n(realistic desks, Nov 6)")):
        c = conf[mode]
        n = sum(c[k] for k in keys)
        rows.append((f"{label}\n{n} placements, agree {100 * (c[keys[0]] + c[keys[1]]) / n:.0f}%",
                     Counter({k: c[k] for k in keys})))
    cols = [SLOTS[0], "#86b6ef", SLOTS[1], SLOTS[3]]
    hbar_stack(ax, rows, keys, cols)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in cols], labels=names, ncol=2, fontsize=8,
              frameon=False, loc="upper left", bbox_to_anchor=(-.44, -.16), labelcolor=INK2)
    fig.text(.02, .92, "Can the robot trust its own camera check?", fontsize=13, color=INK, weight="bold")
    fig.text(.02, .86, "Final camera verdict per placement vs the offline ground truth (never shown to the robot)",
             fontsize=9, color=INK2)
    fig.savefig(FIG / "self_check.png", facecolor=SURFACE)
    plt.close(fig)

    # 4. Skill memory vs no memory on the same desks: success, LLM calls and cost per desk
    pairs = {}
    for name, (none, skill) in PAIRS.items():
        pairs[name] = {}
        done = [{e["task"] for e in eps if e["run"] in rs and e["success_gt"] in ("True", "False")} for rs in (none, skill)]
        both = done[0] & done[1]  # a desk stopped by the spend cap has no ground truth: left out of both groups
        for lab, rs in (("none", none), ("skill", skill)):
            v = [e for e in eps if e["run"] in rs and e["task"] in both]
            n = len(v)
            pairs[name][lab] = {"desks": n, "whole_desk_success": sum(e["success_gt"] == "True" for e in v),
                                "subtasks": f'{int(sum(num(e["subtasks_gt"]) for e in v))}/{int(sum(num(e["subtasks_total"]) for e in v))}',
                                "llm_calls_per_desk": round(sum(num(e["llm_calls"]) for e in v) / n, 1),
                                "programs_per_desk": round(sum(num(e["programs"]) for e in v) / n, 1),
                                "direct_skill_reuses": int(sum(num(e["direct_reuse"]) for e in v)),
                                "cost_per_desk_usd": round(sum(num(e["cost_usd"]) for e in v) / n, 3),
                                "minutes_per_desk": round(sum(num(e["seconds"]) for e in v) / n / 60, 1)}
    out["skill_vs_none"] = pairs
    fig = figure(9, 4.2)
    panels = [("whole_desk_success", "Whole desk tidied (ground truth)", lambda d: 100 * d["whole_desk_success"] / d["desks"], "{:.0f}%"),
              ("llm_calls_per_desk", "LLM calls per desk", lambda d: d["llm_calls_per_desk"], "{:.0f}"),
              ("cost_per_desk_usd", "API cost per desk (USD)", lambda d: d["cost_per_desk_usd"], "${:.2f}")]
    names = list(PAIRS)
    for j, (_, title, get, fmt) in enumerate(panels):
        ax = fig.add_axes([.06 + j * .32, .27, .27, .53])
        style(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, linewidth=.8)
        for i, name in enumerate(names):
            for k, (lab, col) in enumerate((("none", NEUTRAL), ("skill", SLOTS[0]))):
                val = get(pairs[name][lab])
                ax.bar(i + (k - .5) * .36, val, width=.34, color=col, edgecolor=SURFACE, linewidth=1)
                ax.text(i + (k - .5) * .36, val, fmt.format(val), ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(names)), names, fontsize=7, color=INK2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        if j == 0:
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (NEUTRAL, SLOTS[0])],
               labels=["No memory", "Skill memory"], ncol=2, fontsize=9, frameon=False, loc="lower center",
               labelcolor=INK2)
    fig.text(.02, .93, "Skill memory on the same desks: fewer LLM calls, lower cost", fontsize=13, color=INK, weight="bold")
    fig.text(.02, .87, "Paired runs: realdesk 20 + 20 (Gemini), 10 + 10 (Gemini with GraspNet/cuRobo), sdesk 3 + 3 "
             "(DeepSeek, SAM2). Small samples, run in sequence", fontsize=8.5, color=INK2)
    fig.savefig(FIG / "skill_vs_none.png", facecolor=SURFACE)
    plt.close(fig)

    # 5. Direct reuse of a stored skill vs a freshly written program (act level; reuse is chosen when it applies,
    # so the two groups are not comparable desks)
    dr = defaultdict(Counter)
    for a in att:
        if a["api_level"] == "act" and a["has_code"] == "True":
            dr["direct reuse" if a["direct_reuse"] == "True" else "fresh program"][a["outcome"] == "success"] += 1
    out["direct_reuse_vs_fresh"] = {k: {"attempts": v[True] + v[False], "success": v[True],
                                        "rate": round(v[True] / (v[True] + v[False]), 3)} for k, v in dr.items()}

    # 6. Recognition on the realistic desk (DeepSeek, Set-of-Mark), and where goals came from
    ident = [e for e in eps if e["things"] not in ("", "None") and setting(e) == "C"]
    out["recognition_sdesk"] = {"desks": len(ident), "things": int(sum(num(e["things"]) for e in ident)),
                                "category_correct": int(sum(num(e["category_ok"]) for e in ident)),
                                "role_correct": int(sum(num(e["role_ok"]) for e in ident))}
    src = Counter()
    for e in eps:
        if setting(e) == "C":
            for k in ("instruction", "user_memory", "scene_grouping", "assumption"):
                src[k] += int(num(e[f"goal_src_{k}"]))
    out["goal_sources_sdesk"] = dict(src)
    out.update(sam2_desks(eps, pairs["DeepSeek\nSAM2\n3 + 3 desks"]))
    json.dump(out, open(STATS / "numbers.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def single_bars(path, labels, values, texts, title, subtitle, percent=False, colours=None, w=5.2, h=4.6):
    """A slide figure with one bar per label (one series: no legend), value text above each bar."""
    fig = figure(w, h)
    ax = fig.add_axes([.12, .17, .84, .58])
    style(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, linewidth=.8)
    for i, (v, t) in enumerate(zip(values, texts)):
        ax.bar(i, v, width=.55, color=(colours or [SLOTS[0]] * len(values))[i], edgecolor=SURFACE, linewidth=1)
        ax.text(i, v + (1.5 if percent else max(values) * .015), t, ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(range(len(labels)), labels, fontsize=9.5, color=INK)
    if percent:
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    else:
        ax.set_ylim(0, max(values) * 1.18)
    fig.text(.03, .92, title, fontsize=13, color=INK, weight="bold")
    fig.text(.03, .86, subtitle, fontsize=8.5, color=INK2)
    fig.savefig(FIG / path, facecolor=SURFACE)
    plt.close(fig)


def sam2_desks(eps, pair):
    """Realistic desks with SAM2 perception (setting C, finished desks only): recognition, destination inference,
    items placed and whole desks by ground truth, by kind of object; and the two-bar memory comparison."""
    rows = [e for e in eps if setting(e) == "C" and e["success_gt"] in ("True", "False")]
    later = [e for e in eps if e["family"] == "sdesk" and e["date"] > "2026-11-06" and e["success_gt"] in ("True", "False")]

    def tally(desks):
        t = {"desks": len(desks), "whole_desk": 0, "items": 0, "placed": 0, "destination_right": 0, "things": 0,
             "category_right": 0, "tidy": 0, "tidy_kept": 0}
        kinds = defaultdict(lambda: [0, 0])
        for d in desks:
            e = json.load(open(ROOT / "results" / d["path"] / "episode.json"))
            meta, ev, ident = e["task_meta"], e["private_eval"]["objects"], e.get("identification") or {}
            cat = meta["category"]

            def ok(o):
                return bool(ev[o].get("inside") and ev[o].get("resting") and ev[o].get("upright"))
            goals = {(str(ident.get(g["object"], {}).get("category")), str(ident.get(g["zone"], {}).get("category")))
                     for g in (e.get("task_spec") or {}).get("goals", [])}
            t["whole_desk"] += d["success_gt"] == "True"
            for o in meta["out_of_place"]:
                t["items"] += 1
                t["placed"] += ok(o)
                t["destination_right"] += (cat[o], cat[ev[o]["zone"]]) in goals
                kinds[cat[o]][0] += ok(o)
                kinds[cat[o]][1] += 1
            for o in meta["already_tidy"]:
                t["tidy"] += 1
                t["tidy_kept"] += ok(o)
            sc = e.get("identification_score") or {}
            t["things"] += sc.get("things", 0)
            t["category_right"] += sc.get("category_ok", 0)
        t["by_kind"] = {k: v for k, v in kinds.items()}
        return t

    main, after = tally(rows), tally(later)
    stages = [("Recognise\nthings", main["category_right"], main["things"]),
              ("Infer the right\ndestination", main["destination_right"], main["items"]),
              ("Place the item\n(ground truth)", main["placed"], main["items"]),
              ("Tidy the\nwhole desk", main["whole_desk"], main["desks"])]
    single_bars("sam2_pipeline.png", [s[0] for s in stages], [100 * a / b for _, a, b in stages],
                [f"{100 * a / b:.0f}%\n({a}/{b})" for _, a, b in stages],
                "Real perception: understanding is solid, execution is the gap",
                f"{main['desks']} realistic desks, SAM2 perception, DeepSeek; the request is only \"Tidy up my desk\"",
                percent=True, w=7, h=4.6)
    order = sorted(main["by_kind"], key=lambda k: -main["by_kind"][k][0] / main["by_kind"][k][1])
    names = {"crumpled paper ball": "Paper ball\n(into the bin)", "mug": "Mug\n(onto the coaster)",
             "pen": "Pen (stand up\nin the holder)", "notebook": "Notebook (edge\npinch, onto pile)"}
    single_bars("sam2_by_object.png", [names.get(k, k) for k in order],
                [100 * main["by_kind"][k][0] / main["by_kind"][k][1] for k in order],
                [f"{100 * main['by_kind'][k][0] / main['by_kind'][k][1]:.0f}%\n({main['by_kind'][k][0]}/{main['by_kind'][k][1]})"
                 for k in order],
                "Items placed, by kind of object", f"Ground truth on {main['desks']} realistic desks with SAM2 perception",
                percent=True, w=7, h=4.6)
    n, k = pair["none"], pair["skill"]
    single_bars("memory_llm_calls.png", ["No memory", "Skill memory"], [n["llm_calls_per_desk"], k["llm_calls_per_desk"]],
                [f'{n["llm_calls_per_desk"]:.0f}', f'{k["llm_calls_per_desk"]:.0f}'], "LLM calls per desk",
                f'Same {n["desks"]} desks; {100 * (1 - k["llm_calls_per_desk"] / n["llm_calls_per_desk"]):.0f}% fewer calls '
                "with skill memory", colours=[NEUTRAL, SLOTS[0]], w=4.6, h=4.6)
    for t in (main, after):
        t["by_kind"] = {kk: f"{v[0]}/{v[1]}" for kk, v in t["by_kind"].items()}
    return {"sam2_desks_to_nov6": main, "sam2_desks_nov8_after_fixes": after}


if __name__ == "__main__":
    main()

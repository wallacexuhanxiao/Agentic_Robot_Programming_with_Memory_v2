"""Aggregate pen-case memory runs: per run and per condition (mean, std, min-max) over the ten pen tasks of each
run (the teaching task is excluded), plus paired differences per layout set.

Usage: python scripts/aggregate_pen_memory.py <results dir> <runs.json> <out.md>
runs.json: [{"condition": "both", "layout_set": "A", "sequence": "teach", "dir": "penteach30b_both_.../both"}, ...]
"""
import json
import statistics
import sys
from pathlib import Path

METRICS = [("success_gt", "success"), ("executions_failed", "failed executions"), ("code_generations", "attempts"),
           ("llm_calls", "LLM calls"), ("completion_tokens", "tokens"), ("skill_calls", "direct skill reuse")]
ORDER = ["none", "skill", "user", "both"]


def run_metrics(path):
    rows = [r for r in json.loads((path / "summary.json").read_text())["tasks"]
            if r["task"].startswith("pen_") and not r["task"].startswith("pen_teach")]
    out = {name: sum(r[key] for r in rows) for key, name in METRICS}
    out["tasks"] = len(rows)
    first = 0
    for r in rows:
        ep = json.loads((path / r["task"] / "episode.json").read_text())
        a0 = ep["subgoals"][0]["attempts"][0] if ep.get("subgoals") else None
        first += bool(a0 and a0["check"].get("satisfied") and ep["private_eval"]["success"])
    out["solved at first attempt"] = first
    return out


def fmt(values):
    if len(values) == 1:
        return f"{values[0]}"
    sd = statistics.stdev(values)
    return f"{statistics.mean(values):.1f} ± {sd:.1f} ({min(values)}–{max(values)})"


def main(results, runs_file, out):
    results = Path(results)
    runs = json.loads(Path(runs_file).read_text())
    for r in runs:
        r["m"] = run_metrics(results / r["dir"])
    lines = ["# Pen-case memory experiment: replication", "",
             "Ten pen tasks per run (the teaching task is excluded). Mean ± sd (min–max) over runs.", ""]
    for seq in ("teach", "plain"):
        sel = [r for r in runs if r["sequence"] == seq]
        if not sel:
            continue
        conds = [c for c in ORDER if any(r["condition"] == c for r in sel)]
        keys = ["success", "solved at first attempt", "failed executions", "attempts", "LLM calls", "tokens",
                "direct skill reuse"]
        lines += [f"## Sequence: {'teaching task + 10 pen tasks' if seq == 'teach' else '10 pen tasks (no teaching task)'}", "",
                  "| metric | " + " | ".join(f"{c} (n={sum(r['condition'] == c for r in sel)})" for c in conds) + " |",
                  "|---" * (len(conds) + 1) + "|"]
        for k in keys:
            lines.append(f"| {k} | " + " | ".join(fmt([r["m"][k] for r in sel if r["condition"] == c]) for c in conds) + " |")
        lines.append("")
    lines += ["## Per run", "", "| layout set | sequence | condition | success | first attempt | failed | LLM calls | skill reuse | dir |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: (r["layout_set"], r["sequence"], ORDER.index(r["condition"]))):
        m = r["m"]
        lines.append(f"| {r['layout_set']} | {r['sequence']} | {r['condition']} | {m['success']}/{m['tasks']} | "
                     f"{m['solved at first attempt']} | {m['failed executions']} | {m['LLM calls']} | {m['direct skill reuse']} | "
                     f"`{r['dir']}` |")
    teach = [r for r in runs if r["sequence"] == "teach"]
    sets = sorted({r["layout_set"] for r in teach})
    lines += ["", "## Paired by layout set (teaching sequence): success of both − max(user, skill)", ""]
    for s in sets:
        by = {}
        for r in teach:
            if r["layout_set"] == s:
                by.setdefault(r["condition"], []).append(r["m"]["success"])
        if all(c in by for c in ("both", "user", "skill")):
            b, u, k = (statistics.mean(by[c]) for c in ("both", "user", "skill"))
            n = by.get("none")
            lines.append(f"- set {s}: none {fmt(n) if n else '-'}, skill {k:.1f}, user {u:.1f}, both {b:.1f} "
                         f"(both − max(user, skill) = {b - max(u, k):+.1f}; both − (user + skill) = {b - u - k:+.1f})")
    Path(out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:4])

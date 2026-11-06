"""Table of act-level runs (Gemini): per task set and condition, tasks solved (ground truth), subgoals verified by the
camera, code generations, failed executions, calls of stored skills, harness errors and billed API spend.
Usage: python scripts/aggregate_act.py RESULTS_DIR OUT.md RUN_DIR [RUN_DIR ...]
A RUN_DIR holds condition directories (none/, skill/) or set directories that hold them (pen_holder_7000/none/...)."""
import json
import sys
from pathlib import Path

results, out, runs = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
rows = []
for run in runs:
    base = results / run
    for summary in sorted(base.glob("**/summary.json")):
        cond = summary.parent.name
        if cond not in ("none", "skill", "user", "both"):
            continue
        data = json.loads(summary.read_text())
        tasks = data["tasks"]
        spend = summary.parent / "spend.json"
        init = summary.parent / "memory_init.txt"
        rows.append({
            "run": run, "set": str(summary.parent.parent.relative_to(base)) or ".", "condition": cond,
            "inherited_memory": init.exists(),
            "tasks": len(tasks), "solved": sum(t["success_gt"] for t in tasks),
            "subgoals": sum(t.get("subgoals", 0) for t in tasks),
            "verified": sum(t.get("subgoals_verified", 0) for t in tasks),
            "code_generations": sum(t.get("code_generations", 0) for t in tasks),
            "failed_executions": sum(t.get("executions_failed", 0) for t in tasks),
            "skill_calls": sum(t.get("skill_calls", 0) for t in tasks),
            "harness_errors": sum(1 for t in tasks if t.get("harness_error")),
            "subtasks": (sum(t["subtasks_gt"] for t in tasks), sum(t["subtasks_total"] for t in tasks))
            if all("subtasks_total" in t for t in tasks) else None,
            "tidy_kept": (sum(t["tidy_kept_gt"] for t in tasks), sum(t["tidy_total"] for t in tasks))
            if all("tidy_total" in t for t in tasks) else None,
            "billed_usd": json.loads(spend.read_text()).get("usd") if spend.exists() else None})
lines = ["| run | set | condition | inherited skills | solved | subtasks done (GT) | tidy items kept (GT) | subgoals verified | code generations | failed executions | skill calls | harness errors | billed USD |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    frac = lambda p: "" if p is None else f"{p[0]}/{p[1]}"
    lines.append(f"| {r['run']} | {r['set']} | {r['condition']} | {'yes' if r['inherited_memory'] else ''} | {r['solved']}/{r['tasks']} | "
                 f"{frac(r['subtasks'])} | {frac(r['tidy_kept'])} | "
                 f"{r['verified']}/{r['subgoals']} | {r['code_generations']} | {r['failed_executions']} | {r['skill_calls']} | "
                 f"{r['harness_errors']} | {'' if r['billed_usd'] is None else round(r['billed_usd'], 3)} |")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n")
(out.with_suffix(".json")).write_text(json.dumps(rows, indent=1))
print("\n".join(lines))

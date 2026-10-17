"""Compare conditions of one run: per-task table + held-out aggregates + applicability accuracy."""
import json
import sys
from pathlib import Path


def load(run_dir, condition):
    path = Path(run_dir) / condition / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def aggregate(rows):
    n = len(rows)
    if not n:
        return {}
    total = lambda k: sum(r[k] for r in rows)
    return {"tasks": n, "task_success_gt": f"{total('success_gt')}/{n}",
            "subgoals_verified": f"{total('subgoals_verified')}/{total('subgoals')}",
            "code_generations": total("code_generations"), "failed_executions": total("executions_failed"),
            "skill_calls": total("skill_calls"), "llm_calls": total("llm_calls"),
            "completion_tokens": total("completion_tokens"), "control_steps": total("control_steps"),
            "wall_seconds": round(total("seconds"))}


def main(run_dir):
    lines = [f"# Run {Path(run_dir).name}", ""]
    data = {c: load(run_dir, c) for c in ("none", "memory")}
    for suite in ("train", "heldout"):
        lines += [f"## {suite}", "", "| metric | no memory | with memory |", "|---|---|---|"]
        aggs = {c: aggregate([r for r in d["tasks"] if r["suite"] == suite]) if d else {} for c, d in data.items()}
        for key in ("tasks", "task_success_gt", "subgoals_verified", "code_generations", "failed_executions",
                    "skill_calls", "llm_calls", "completion_tokens", "control_steps", "wall_seconds"):
            lines.append(f"| {key} | {aggs['none'].get(key, '-')} | {aggs['memory'].get(key, '-')} |")
        lines.append("")
    lines += ["## Per task", "", "| condition | task | GT success | subgoals ok | code gens | failed execs | skill calls |",
              "|---|---|---|---|---|---|---|"]
    for c, d in data.items():
        for r in (d or {}).get("tasks", []):
            lines.append(f"| {c} | {r['task']} | {r['success_gt']} | {r['subgoals_verified']}/{r['subgoals']} | "
                         f"{r['code_generations']} | {r['executions_failed']} | {r['skill_calls']} |")
    log = (data.get("memory") or {}).get("applicability_log", [])
    if log:
        right = sum((e["verdict"] == "applicable") == e["outcome"] for e in log)
        lines += ["", f"## Applicability checks", "",
                  f"{right}/{len(log)} verdicts matched the outcome of the skill call "
                  "(applicable -> success, uncertain/not_applicable -> failure)."]
        for e in log:
            lines.append(f"- {e['task']}: {e['function']} verdict={e['verdict']} outcome={'ok' if e['outcome'] else 'fail'}")
    mem = Path(run_dir) / "memory" / "memory.json"
    if mem.exists():
        lines += ["", "## Skill memory at end of run", ""]
        for s in json.loads(mem.read_text())["skills"]:
            kinds = {}
            for h in s["history"]:
                kinds.setdefault(h["features"]["kind"], [0, 0])[0 if h["outcome"] else 1] += 1
            lines.append(f"- {s['function']} ({s['status']}), from {s['origin']['task']}: " +
                         ", ".join(f"{k} {w} ok/{l} fail" for k, (w, l) in kinds.items()))
    text = "\n".join(lines) + "\n"
    (Path(run_dir) / "REPORT.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])

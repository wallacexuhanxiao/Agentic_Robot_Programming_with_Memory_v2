"""Run a task sequence under one condition: 'memory' (persistent skill memory) or 'none'."""
import argparse
import json
import time
from pathlib import Path

from ..agent.llm import LLM
from ..agent.loop import run_episode
from ..env.tasks import suite
from ..memory.store import SkillMemory


def summarize(record):
    subgoals = record.get("subgoals", [])
    attempts = [a for s in subgoals for a in s["attempts"]]
    return {"task": record["task_id"], "family": record["family"],
            "success_gt": record["private_eval"]["success"],
            "subgoals": len(subgoals), "subgoals_verified": sum(s["success"] for s in subgoals),
            "code_generations": sum(1 for a in attempts),
            "executions_failed": sum(1 for a in attempts if not a["check"].get("satisfied")),
            "skill_calls": sum(len(a["used_skills"]) for a in attempts),
            "control_steps": sum(a["control_steps"] for a in attempts),
            "llm_calls": record["llm_usage"]["calls"], "llm_seconds": record["llm_usage"]["seconds"],
            "completion_tokens": record["llm_usage"]["completion_tokens"],
            "seconds": record["seconds"], "harness_error": record.get("harness_error")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", choices=("memory", "none"), required=True)
    p.add_argument("--suites", default="train,heldout")
    p.add_argument("--base-url", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-validate", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    memory = SkillMemory(args.out / "memory.json") if args.condition == "memory" else None
    log_file = (args.out / "run.log").open("a")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    rows = []
    for suite_name in args.suites.split(","):
        for task in suite(suite_name):
            log(f"[{args.condition}] {suite_name} {task.task_id}: {task.instruction}")
            llm = LLM(args.base_url, log_dir=args.out / task.task_id / "llm")
            record = run_episode(task, llm, memory, args.out / task.task_id, validate=not args.no_validate, log=log)
            row = {"suite": suite_name, **summarize(record)}
            rows.append(row)
            log(f"  -> {row}")
            (args.out / "summary.json").write_text(json.dumps(
                {"condition": args.condition, "tasks": rows,
                 "applicability_log": memory.applicability_log if memory else []}, indent=1))


if __name__ == "__main__":
    main()

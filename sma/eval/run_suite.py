"""Run a task sequence under one condition: 'memory' (persistent skill memory) or 'none'."""
import argparse
import os
import json
import time
from pathlib import Path

from ..agent.llm import LLM
from ..agent.loop import run_episode
from ..env.tasks import suite
from ..memory.store import SkillMemory
from ..memory.user import UserMemory


def summarize(record):
    subgoals = record.get("subgoals", [])
    spec = record.get("task_spec", {})
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
            "seconds": record["seconds"], "harness_error": record.get("harness_error"),
            "goal_sources": spec.get("sources"), "goals_dropped": len(spec.get("dropped", []))}


def main():
    p = argparse.ArgumentParser()
    # none | skill | user | both  ("memory" is kept as an alias of "skill")
    p.add_argument("--condition", choices=("none", "skill", "user", "both", "memory"), required=True)
    p.add_argument("--suites", default="train,heldout")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B", help="served name of the general model")
    p.add_argument("--coder-url", default=None, help="optional separate endpoint for code/extract calls")
    p.add_argument("--coder-model", default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-validate", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    memory = SkillMemory(args.out / "memory.json") if args.condition in ("skill", "both", "memory") else None
    user_memory = UserMemory(args.out / "user_memory.json") if args.condition in ("user", "both") else None
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
            llm = LLM(args.base_url, model=args.model, log_dir=args.out / task.task_id / "llm",
                      coder_url=args.coder_url, coder_model=args.coder_model)
            try:
                record = run_episode(task, llm, memory, args.out / task.task_id, user_memory=user_memory,
                                     validate=not args.no_validate, log=log)
                row = {"suite": suite_name, **summarize(record)}
            except Exception as exc:  # keep the sequence going; the failure is recorded, not hidden
                import traceback
                log(f"  HARNESS EXCEPTION in {task.task_id}: {type(exc).__name__}: {exc}")
                (args.out / task.task_id).mkdir(parents=True, exist_ok=True)
                (args.out / task.task_id / "harness_exception.txt").write_text(traceback.format_exc())
                row = {"suite": suite_name, "task": task.task_id, "family": task.family, "success_gt": False,
                       "subgoals": 0, "subgoals_verified": 0, "code_generations": 0, "executions_failed": 0,
                       "skill_calls": 0, "control_steps": 0, "llm_calls": len(llm.calls),
                       "llm_seconds": 0, "completion_tokens": 0, "seconds": 0,
                       "harness_error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            log(f"  -> {row}")
            (args.out / "summary.json").write_text(json.dumps(
                {"condition": args.condition, "api_level": os.environ.get("SMA_API_LEVEL", "task"),
                 "general_model": args.model, "coder_model": args.coder_model or args.model, "tasks": rows,
                 "applicability_log": memory.applicability_log if memory else []}, indent=1))


if __name__ == "__main__":
    main()

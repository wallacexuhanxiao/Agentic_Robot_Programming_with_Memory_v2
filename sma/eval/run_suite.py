"""Run a task sequence under one condition: 'memory' (persistent skill memory) or 'none'."""
import argparse
import os
import json
import time
from pathlib import Path

from ..agent.llm import LLM, SpendCapReached
from ..agent.loop import run_episode
from ..env.tasks import suite
from ..memory.store import SkillMemory
from ..memory.user import UserMemory


def summarize(record):
    subgoals = record.get("subgoals", [])
    spec = record.get("task_spec", {})
    attempts = [a for s in subgoals for a in s["attempts"]]
    placed = {o for o, d in record["private_eval"].get("objects", {}).items()
              if d.get("inside") and d.get("resting") and d.get("upright")}
    meta = record.get("task_meta") or {}
    extra = {}
    if meta.get("out_of_place") is not None:  # realdesk: subtasks done, and tidy objects still in place (ground truth)
        extra = {"subtasks_total": len(meta["out_of_place"]),
                 "subtasks_gt": sum(o in placed for o in meta["out_of_place"]),
                 "tidy_total": len(meta["already_tidy"]), "tidy_kept_gt": sum(o in placed for o in meta["already_tidy"])}
    return {"task": record["task_id"], "family": record["family"],
            "success_gt": record["private_eval"]["success"], **extra,
            "subgoals": len(subgoals), "subgoals_verified": sum(s["success"] for s in subgoals),
            "code_generations": sum(1 for a in attempts),
            "executions_failed": sum(1 for a in attempts if not a["check"].get("satisfied")),
            "skill_calls": sum(len(a["used_skills"]) for a in attempts),
            "helper_calls": sum(len(a.get("used_helpers", [])) for a in attempts),
            "replans": len(record.get("replans", [])),
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
    p.add_argument("--resume", action="store_true",
                   help="continue a sequence whose process died: keep finished tasks (summary.json) and the memories "
                        "on disk, rerun from the first unfinished task")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    init = os.environ.get("SMA_MEMORY_INIT")  # carry skills over from an earlier run (cross-task transfer)
    if init and args.condition in ("skill", "both", "memory") and not (args.out / "memory.json").exists():
        import shutil
        args.out.mkdir(parents=True, exist_ok=True)
        shutil.copy(init, args.out / "memory.json")
        (args.out / "memory_init.txt").write_text(f"skill memory initialised from {init}\n")
    memory = SkillMemory(args.out / "memory.json") if args.condition in ("skill", "both", "memory") else None
    user_memory = UserMemory(args.out / "user_memory.json") if args.condition in ("user", "both") else None
    log_file = (args.out / "run.log").open("a")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    rows, done = [], set()
    previous = args.out / "summary.json"
    if args.resume and previous.exists():
        old = json.loads(previous.read_text())
        rows = old["tasks"]
        done = {r["task"] for r in rows}
        if memory is not None:
            memory.applicability_log = old.get("applicability_log", [])
        log(f"[{args.condition}] resuming after a harness crash: {len(rows)} finished tasks kept")
    if os.environ.get("SMA_LLM_PROVIDER") in ("gemini", "deepseek"):  # one spending ledger per run (every task)
        os.environ.setdefault("SMA_SPEND_LEDGER", str(args.out / "spend.json"))
    warm = os.environ.get("SMA_MEMORY_WARMUP")  # desks the robot only looks at first: what they show goes into the
    # long-term memory (no LLM, nothing moves); named desks only (a colour-coded desk needs an identification first)
    if warm and user_memory is not None and not (args.out / "warmup.json").exists():
        from ..agent.grouping import infer_homes
        from ..agent.loop import make_env
        seen = []
        for t in suite(warm):
            env = make_env(t, video=False)
            try:
                scene = env.reset()
                seen.append({"task": t.task_id, "observed": user_memory.observe_habits(infer_homes(scene), scene, t.task_id)})
            finally:
                env.close()
        (args.out / "warmup.json").write_text(json.dumps(seen, indent=1))
        log(f"[{args.condition}] long-term memory warmed up on {warm}: {sum(len(x['observed']) for x in seen)} observations")
    stopped = None
    for suite_name in args.suites.split(","):
        if stopped:
            break
        for task in suite(suite_name):
            if task.task_id in done:
                continue
            if args.resume and (args.out / task.task_id).exists():  # partial output of the task that crashed
                import shutil
                shutil.move(str(args.out / task.task_id), str(args.out / f"{task.task_id}.crashed_{int(time.time())}"))
            import resource
            log(f"[{args.condition}] {suite_name} {task.task_id}: {task.instruction} "
                f"(process max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB)")
            llm = LLM(args.base_url, model=args.model, log_dir=args.out / task.task_id / "llm",
                      coder_url=args.coder_url, coder_model=args.coder_model)
            try:
                record = run_episode(task, llm, memory, args.out / task.task_id, user_memory=user_memory,
                                     validate=not args.no_validate, log=log)
                row = {"suite": suite_name, **summarize(record)}
            except SpendCapReached as exc:  # stop the run: no further task may spend money
                log(f"  SPENDING CAP: {exc}; the run stops before {task.task_id} finishes")
                stopped = str(exc)
                break
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
            row["after_resume"] = bool(done)  # ran in a process restarted after a harness crash
            rows.append(row)
            log(f"  -> {row}")
            (args.out / "summary.json").write_text(json.dumps(
                {"condition": args.condition, "api_level": os.environ.get("SMA_API_LEVEL", "task"),
                 "general_model": args.model, "coder_model": args.coder_model or args.model, "tasks": rows,
                 "api_spend": json.loads(Path(os.environ["SMA_SPEND_LEDGER"]).read_text())
                 if os.environ.get("SMA_SPEND_LEDGER") and Path(os.environ["SMA_SPEND_LEDGER"]).exists() else None,
                 "applicability_log": memory.applicability_log if memory else []}, indent=1))
    if stopped:
        summary_path = args.out / "summary.json"
        data = json.loads(summary_path.read_text()) if summary_path.exists() else {"condition": args.condition, "tasks": rows}
        data["stopped_by_spend_cap"] = stopped
        summary_path.write_text(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()

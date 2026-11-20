"""Development statistics from every downloaded run (no simulator, no LLM): one row per desk (episode) and one row per
attempt, from results/**/episode.json and the summary.json beside them. Writes results/dev_stats/episodes.csv,
attempts.csv and subgoals.csv. Usage: python3 scripts/dev_stats.py"""
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS, OUT = ROOT / "results", ROOT / "results" / "dev_stats"

# What failed, from the attempt's error (or from the camera check when the program ran without error)
MOTION = {"motion_blocked": "blocked by contact", "unreachable_pose": "pose out of reach",
          "curobo_plan_failed": "pose out of reach", "target_outside_workspace": "argument out of range",
          "carry_height_out_of_range": "argument out of range", "grasp_height_out_of_range": "argument out of range",
          "not_holding_anything": "grasp failed", "already_holding_something": "wrong robot state",
          "object_not_visible": "object not seen", "object_not_visible_before_attempt": "object not seen",
          "footprint_unknown": "object not seen"}


def failure(attempt, names):
    """One of: success, goal not met (ran without error, the camera check failed), lost from view after acting (ran
    without error, the camera could not find the object to check it), blocked by contact, pose out of reach, argument
    out of range, grasp failed, object not seen, program bug, program gave up, rejected by sandbox, wrong robot state,
    other."""
    err, ok = attempt.get("error"), (attempt.get("check") or {}).get("satisfied")
    if not err:
        return "success" if ok else "lost from view after acting" if (attempt.get("check") or {}).get("reason") == \
            "not_visible" else "goal not met"
    if isinstance(err, str):
        return "other"
    kind = err.get("type")
    if kind == "rejected":
        return "rejected by sandbox"
    if kind == "robot_error":
        return MOTION.get(err.get("reason"), "other")
    if kind == "program_error":
        exc, msg = err.get("exception"), str(err.get("message") or "")
        low = msg.lower()
        if exc in ("RuntimeError", "AssertionError") and ("grasp" in low or "grip" in low or "missed" in low):
            return "grasp failed"
        if "not visible" in low or (exc == "KeyError" and msg.strip("'\"") in names):
            return "object not seen"
        if exc == "RuntimeError":
            return "program gave up"
        return "program bug"
    return "other"


def date_of(path):
    m = re.search(r"(2026)(\d\d)(\d\d)T", str(path))
    if m:
        return f"{m[1]}-{m[2]}-{m[3]}"
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d")


def kind_of(name, e):
    """What the subgoal's object is: the model's category for a camera thing (sdesk), the palette colour's true
    category (cdesk), else the kind in the object's name (red_pen -> pen)."""
    ident = e.get("identification") or {}
    if name in ident:
        return str(ident[name].get("category", "?")).replace("other: ", "")
    true = (e.get("task_meta") or {}).get("true_category") or {}
    if name in true:
        return true[name]
    return name.rsplit("_", 1)[-1] if "_" in name else name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    episodes, attempts, subgoals, seen = [], [], [], set()
    for f in sorted(RESULTS.glob("**/episode.json")):
        if "dev_stats" in f.parts:
            continue
        digest = hashlib.sha1(f.read_bytes()).hexdigest()
        if digest in seen:  # the same episode copied into two directories (ds_full2_sdesk: 17000 twice)
            continue
        seen.add(digest)
        try:
            e = json.load(open(f))
        except (OSError, ValueError):
            continue
        rel = f.relative_to(RESULTS)
        run = rel.parts[0]
        summ, row = {}, {}
        s_path = f.parent.parent / "summary.json"
        if s_path.exists():
            try:
                summ = json.load(open(s_path))
            except ValueError:
                summ = {}
            if isinstance(summ, dict):
                row = next((t for t in summ.get("tasks", []) or [] if t.get("task") == e.get("task_id")), {})
            else:
                summ = {}
        usage = e.get("llm_usage") or {}
        meta, ev = e.get("task_meta") or {}, e.get("private_eval") or {}
        names = set((ev.get("objects") or {}).keys()) | set((e.get("identification") or {}).keys())
        idscore = e.get("identification_score") or {}
        spec = e.get("task_spec") or {}
        sources = {}
        for g in spec.get("goals", []) or []:
            sources[g.get("source")] = sources.get(g.get("source"), 0) + 1
        all_attempts = [a for s in e.get("subgoals") or [] for a in s.get("attempts") or []]
        ep = {
            "run": run, "path": str(rel.parent), "date": date_of(f), "task": e.get("task_id"), "family": e.get("family"),
            "condition": summ.get("condition") or (rel.parts[-3] if len(rel.parts) >= 3 else ""),
            "api_level": summ.get("api_level") or "", "model": summ.get("coder_model") or summ.get("general_model") or
            ",".join(usage.get("models") or []),
            "success_gt": row.get("success_gt", ev.get("success")),
            "subtasks_total": row.get("subtasks_total", meta.get("subtasks")), "subtasks_gt": row.get("subtasks_gt"),
            "tidy_total": row.get("tidy_total"), "tidy_kept_gt": row.get("tidy_kept_gt"),
            "subgoals": len(e.get("subgoals") or []), "subgoals_verified": sum(bool(s.get("success")) for s in
                                                                                e.get("subgoals") or []),
            "attempts": len(all_attempts), "programs": sum(a.get("code") is not None for a in all_attempts),
            "direct_reuse": sum(bool(a.get("direct_reuse")) for a in all_attempts),
            "used_skill": sum(bool(a.get("used_skills")) for a in all_attempts),
            "replans": len(e.get("replans") or []) if isinstance(e.get("replans"), list) else e.get("replans") or 0,
            "llm_calls": usage.get("calls", row.get("llm_calls")), "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens", row.get("completion_tokens")),
            "llm_seconds": usage.get("seconds", row.get("llm_seconds")), "cost_usd": usage.get("cost_usd"),
            "list_price_usd": usage.get("list_price_usd"), "seconds": e.get("seconds"),
            "control_steps": row.get("control_steps"), "harness_error": bool(e.get("harness_error") or
                                                                            row.get("harness_error")),
            "things": idscore.get("things"), "category_ok": idscore.get("category_ok"), "role_ok": idscore.get("role_ok"),
            "goals_inferred": len(spec.get("goals", []) or []), "goals_dropped": len(spec.get("dropped", []) or []),
            **{f"goal_src_{k}": sources.get(k, 0) for k in ("instruction", "user_memory", "scene_grouping", "assumption")},
        }
        episodes.append(ep)
        for i, s in enumerate(e.get("subgoals") or []):
            kind = kind_of(s.get("object", ""), e)
            sa = s.get("attempts") or []
            subgoals.append({"run": run, "task": ep["task"], "family": ep["family"], "date": ep["date"],
                             "condition": ep["condition"], "api_level": ep["api_level"], "model": ep["model"],
                             "index": i, "object": s.get("object"), "zone": s.get("zone"), "kind": kind,
                             "attempts": len(sa), "camera_success": bool(s.get("success")),
                             "first_attempt_success": bool(sa and (sa[0].get("check") or {}).get("satisfied")
                                                           and not sa[0].get("error")),
                             "success_attempt": next((a.get("n") for a in sa if (a.get("check") or {}).get("satisfied")
                                                      and not a.get("error")), None),
                             "extracted_skill": bool(s.get("extracted_skill"))})
            for a in sa:
                err = a.get("error") if isinstance(a.get("error"), dict) else {}
                attempts.append({
                    "run": run, "task": ep["task"], "family": ep["family"], "date": ep["date"],
                    "condition": ep["condition"], "api_level": ep["api_level"], "model": ep["model"], "subgoal": i,
                    "kind": kind, "n": a.get("n"), "direct_reuse": bool(a.get("direct_reuse")),
                    "used_skill": bool(a.get("used_skills")), "used_helper": bool(a.get("used_helpers")),
                    "has_code": a.get("code") is not None, "program_ok": a.get("program_ok"),
                    "check_ok": bool((a.get("check") or {}).get("satisfied")),
                    "outcome": failure(a, names), "error_type": err.get("type"),
                    "error_reason": err.get("reason") or err.get("exception"),
                    "error_message": str(err.get("message") or "")[:80],
                    "control_steps": a.get("control_steps"), "seconds": a.get("seconds")})
    for name, rows in (("episodes", episodes), ("attempts", attempts), ("subgoals", subgoals)):
        with open(OUT / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(name, len(rows))


if __name__ == "__main__":
    main()

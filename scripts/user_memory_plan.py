"""Does the long-term user memory help? Planning only (no manipulation), on the user-preference sessions
(sma.env.tasks.UPREF_STORY): one simulated user, eight sessions on new realistic desks with SAM2 perception, where the user
states a lasting preference against what the desk shows, makes a one-off request and later changes the preference.

Per session the camera looks once and the model identifies the things once; the same scene then goes to three
conditions, each keeping its own state across the sessions:
  none     the current message only (plus the scene grouping and common sense, as always)
  history  the current message plus every earlier message of this user, verbatim (the obvious alternative to a memory)
  memory   the current message plus the structured user memory (sma.memory.user: lasting statements kept with their
           quoted evidence, task-only requests dropped, a newer statement replacing an older one; observed habits)
The refine step decides where each thing goes. Each loose thing's decision is scored against where this user wants it
now (private), by kind of test (stated / rule / one_off / changed / untaught).
Usage (DeepSeek settings as for a run, SAM2 perception): user_memory_plan.py OUT_DIR SEQ [SEQ ...]"""
import json
import os
import sys
import time

from sma.agent.grouping import infer_homes
from sma.agent.identify import identify, private_truth, score_identity
from sma.agent.llm import LLM
from sma.agent.loop import make_env
from sma.agent.refine import refine
from sma.env.tasks import upref_sessions
from sma.memory.user import UserMemory

CONDITIONS = ("none", "history", "memory")


def history_text(message, earlier):
    if not earlier:
        return message
    return message + "\n\nEarlier messages from this user, oldest first: " + " ".join(
        f'({k + 1}) "{m}"' for k, m in enumerate(earlier))


def main():
    out = sys.argv[1]
    seqs = [int(a) for a in sys.argv[2:]] or [0]
    model = os.environ.get("SMA_GENERAL_NAME", "deepseek-flash")
    rows, sessions = [], []
    for seq in seqs:
        sdir = f"{out}/seq{seq}"
        os.makedirs(sdir, exist_ok=True)
        llms = {c: LLM("", model=model, log_dir=f"{sdir}/llm_{c}") for c in CONDITIONS}
        eye = LLM("", model=model, log_dir=f"{sdir}/llm_identify")
        memory = UserMemory(f"{sdir}/user_memory.json")
        earlier = []
        for task in upref_sessions(seq):
            t0 = time.time()
            env = make_env(task, video=False)
            try:
                env.reset()
                identity, views = identify(eye, env)
                scene = env.observe()
                truth_category, thing_of = private_truth(env, task, scene)
                id_score = score_identity(identity, truth_category)
                for v in views:
                    with open(f"{sdir}/{task.task_id}_{v['camera']}.png", "wb") as fh:
                        fh.write(v["png"])
            finally:
                env.close()
            grouping = infer_homes(scene)
            relevant = list(task.meta["user_truth"]) + [z.name for z in task.layout.zones]
            seen_right = {n: bool(thing_of.get(n) and (identity.get(thing_of[n]) or {}).get("category") ==
                                  task.meta["category"].get(n)) for n in relevant}
            session = {"seq": seq, "session": task.meta["session"], "task": task.task_id, "message": task.instruction,
                       "identification": {"category_ok": id_score["category_ok"], "things": id_score["things"]},
                       "relevant_seen_right": seen_right, "thing_of": thing_of,
                       "grouping": {o: h["zone"] for o, h in grouping["homes"].items()}, "conditions": {}}
            for cond in CONDITIONS:
                llm = llms[cond]
                instruction = history_text(task.instruction, earlier) if cond == "history" else task.instruction
                preferences, learned = [], None
                if cond == "memory":
                    learned = memory.learn(llm, task.instruction, scene, task.task_id)
                    preferences = memory.retrieve(scene)
                    memory.observe_habits(grouping, scene, task.task_id)
                before = llm.usage()["prompt_tokens"]
                spec = refine(llm, instruction, preferences, scene, grouping=grouping)
                session["conditions"][cond] = {"spec": spec, "preferences": preferences, "learned": learned,
                                               "refine_prompt_tokens": llm.usage()["prompt_tokens"] - before}
                goals = {g["object"]: g for g in spec["goals"]}
                for obj, want in task.meta["user_truth"].items():
                    thing, want_thing = thing_of.get(obj), thing_of.get(want) if want else None
                    g = goals.get(thing) if thing else None
                    chose = g["zone"] if g else None
                    perceivable = bool(thing) and (want is None or bool(want_thing))
                    correct = perceivable and (chose == want_thing if want else g is None)
                    rows.append({"seq": seq, "session": task.meta["session"], "task": task.task_id, "condition": cond,
                                 "object": obj, "kind": obj.split("_", 1)[1], "case": task.meta["cases"][obj],
                                 "want": want, "want_thing": want_thing, "thing": thing, "chose": chose,
                                 "source": g["source"] if g else None, "perceivable": perceivable, "correct": correct,
                                 "identified_right": seen_right.get(obj, False) and (want is None or seen_right.get(want, False))})
                    r = rows[-1]
                    print(f"seq{seq} s{r['session']} {cond:7s} {r['kind']:8s} {r['case']:8s} want {want} ({want_thing}) "
                          f"chose {chose} [{r['source']}] {'OK' if correct else 'WRONG'}", flush=True)
                # things already in place that the plan would move (a side effect of a misapplied preference)
                session["conditions"][cond]["moves_tidy_things"] = [
                    g["object"] for g in spec["goals"] if g["object"] not in {thing_of.get(o) for o in task.meta["user_truth"]}]
            session["seconds"] = round(time.time() - t0, 1)
            sessions.append(session)
            earlier.append(task.instruction)
            with open(f"{out}/sessions.json", "w") as fh:
                json.dump(sessions, fh, indent=1)
            with open(f"{out}/rows.json", "w") as fh:
                json.dump(rows, fh, indent=1)
        usage = {c: llms[c].usage() for c in CONDITIONS}
        usage["identify"] = eye.usage()
        with open(f"{sdir}/usage.json", "w") as fh:
            json.dump(usage, fh, indent=1)
    summary = {}
    for r in rows:
        for key in (f"{r['condition']}:all", f"{r['condition']}:{r['case']}"):
            d = summary.setdefault(key, {"correct": 0, "total": 0, "correct_identified": 0, "identified": 0})
            d["correct"] += r["correct"]
            d["total"] += 1
            if r["identified_right"]:
                d["correct_identified"] += r["correct"]
                d["identified"] += 1
    with open(f"{out}/summary.json", "w") as fh:
        json.dump({"summary": summary, "sessions": len(sessions)}, fh, indent=1)
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

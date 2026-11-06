"""Recognition trial on the colour-coded desk (no manipulation): for each desk, the model is shown the two labelled camera
images and the camera's measurements and says what each coloured thing is (sma.agent.identify); scored against the
private categories. Usage (Gemini settings as for a run): identify_trial.py OUT_DIR SUITE [SUITE ...]"""
import json
import os
import sys

from sma.agent import identify as identify_module
from sma.agent.identify import identify, private_truth, score_identity
from sma.agent.llm import LLM
from sma.agent.loop import make_env
from sma.env.tasks import suite

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
llm = LLM("", model=os.environ.get("SMA_GENERAL_NAME", "gemini-3.5-flash"), log_dir=f"{out}/llm")
rows, confusions = [], {}
for name in sys.argv[2:]:
    for task in suite(name):
        env = make_env(task, video=False)
        try:
            env.reset()
            try:
                identity, views = identify(llm, env)
            except RuntimeError as exc:  # e.g. the free tier refusing
                print(task.task_id, "error", str(exc)[:200], flush=True)
                continue
            for v in views:
                open(f"{out}/{task.task_id}_{v['camera']}.png", "wb").write(v["png"])
            score = score_identity(identity, private_truth(env, task)[0])
        finally:
            env.close()
        rows.append({"task": task.task_id, **score, "identity": identity, "attempts": identify_module.LAST["attempts"],
                     "answered": any(i["category"] != "other" or i["why"] for i in identity.values())})
        for r in score["rows"]:
            if not r["category_ok"]:
                confusions[f"{r['truth']} -> {r['said']}"] = confusions.get(f"{r['truth']} -> {r['said']}", 0) + 1
        print(task.task_id, f"category {score['category_ok']}/{score['things']}, role {score['role_ok']}/{score['things']}",
              [(r["thing"], r["truth"], r["said"]) for r in score["rows"] if not r["category_ok"]], flush=True)
answered = [r for r in rows if r["answered"]]
summary = {"desks": len(rows), "things": sum(r["things"] for r in rows),
           "category_ok": sum(r["category_ok"] for r in rows), "role_ok": sum(r["role_ok"] for r in rows),
           "calls_first_empty": sum(r["attempts"] > 1 for r in rows), "desks_unanswered": len(rows) - len(answered),
           "answered_things": sum(r["things"] for r in answered),
           "answered_category_ok": sum(r["category_ok"] for r in answered),
           "confusions": confusions, "usage": llm.usage()}
json.dump({"summary": summary, "rows": rows}, open(f"{out}/summary.json", "w"), indent=1)
print("SUMMARY", json.dumps(summary), flush=True)

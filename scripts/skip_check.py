"""Check (no LLM) that a subgoal whose object the camera does not see generates no code: run_subgoal with a model stub
that fails if called, for a name no thing has. Usage: skip_check.py OUT_DIR INDEX (suite: SMA_GATE_SUITE)"""
import json
import os
import sys

from sma.agent.identify import private_truth, role_of
from sma.agent.loop import make_env, run_subgoal
from sma.env.tasks import suite


class NoModel:
    def chat(self, *a, **k):
        raise AssertionError("the model was asked for code for an object the camera does not see")


out, index = sys.argv[1], int(sys.argv[2])
os.makedirs(out, exist_ok=True)
task = suite(os.environ.get("SMA_GATE_SUITE", "sdesk@17100"))[index]
env = make_env(task)
try:
    scene = env.reset()
    truth, seen_as = private_truth(env, task, scene)
    env.set_identity({t: {"category": c, "role": role_of(c)} for t, c in truth.items()})
    zone = next(iter(env.observe()["zones"]))
    ok, attempts = run_subgoal(env, NoModel(), task, "thing999", zone, None, print)
    result = {"task": task.task_id, "ok": ok, "attempts": len(attempts), "code": [a["code"] for a in attempts],
              "error": attempts[-1]["error"]}
finally:
    env.close()
print("SKIP_CHECK", json.dumps(result), flush=True)
json.dump(result, open(f"{out}/skip_{task.task_id}.json", "w"), indent=1)

"""Stored skills on many new desks, no LLM: does what the skill memory learned carry over to other layouts, desk tops,
clutter and combinations of things? The memory (a memory.json from a run) is frozen: nothing is recorded or saved.
Per desk the things are taken as what they are (the private match standing in for a correct identification: this
measures the skills, not recognition); for every thing out of place the memory's own retrieval and applicability check
(goal / state / physical with cuRobo) ranks the stored skills that have handled this kind of thing, and the top one is
executed as is through the sandbox, whatever its verdict, so that the verdicts can be compared with the outcomes. Scored by the robot's camera check and the
ground truth. Writes OUT_DIR/rows.json and summary.json; a video of the first desk of each suite.
Usage (SMA_API_LEVEL=act, SAM2 perception, cuRobo running): skill_transfer.py OUT_DIR MEMORY_JSON SUITE [SUITE ...]"""
import json
import os
import sys
import time
from collections import defaultdict

import imageio.v2 as imageio

from sma.agent.identify import private_truth, role_of
from sma.agent.loop import ATTEMPT_STEPS, make_env, used_helpers, used_skills
from sma.agent.sandbox import API_LEVEL, Sandbox
from sma.env.robot_env import RobotError
from sma.env.tasks import suite
from sma.memory.store import SkillMemory

assert API_LEVEL == "act", "run with SMA_API_LEVEL=act"
ORDER = ("paper", "can", "mug", "book", "eraser", "sticky", "tape", "marker", "pen", "notebook")


class Frozen(SkillMemory):
    def save(self):  # the memory under test is never changed
        pass


def main():
    out, mem_path, suites = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out, exist_ok=True)
    memory = Frozen(mem_path)
    rows, desks = [], []
    for name in suites:
        for i, task in enumerate(suite(name)):
            env = make_env(task, video=i == 0)
            started = time.time()
            desk = {"suite": name, "task": task.task_id, "desk": task.meta.get("desk"), "decor": task.meta.get("decor"),
                    "out_of_place": len(task.meta["out_of_place"])}
            try:
                scene = env.reset()
                truth, seen_as = private_truth(env, task, scene)
                env.set_identity({t: {"category": c, "role": role_of(c)} for t, c in truth.items()})
                kind = {o.name: o.kind for o in task.layout.objects}
                for obj in sorted(task.meta["out_of_place"], key=lambda o: ORDER.index(kind[o])):
                    row = {"suite": name, "task": task.task_id, "object": obj, "kind": kind[obj], "desk": desk["desk"]}
                    rows.append(row)
                    thing, zone = seen_as.get(obj), seen_as.get(task.goal[obj])
                    if not thing or not zone:
                        row.update(outcome="not perceived", gt=False)
                        continue
                    try:
                        env.go_home()
                    except RobotError:
                        pass
                    scene = env.observe()
                    if thing not in scene["objects"]:
                        row.update(outcome="not seen", gt=False)
                        continue
                    feasibility = env.physical_feasibility(thing, zone) if hasattr(env, "physical_feasibility") else None
                    scene = env.observe()
                    retrieved = memory.retrieve(scene, thing, zone, limit=len(memory.skills), feasibility=feasibility,
                                                task_id=task.task_id)
                    seen_kind = scene["objects"][thing].get("kind")
                    # only skills that have handled this kind of thing (one learned for paper balls is never a
                    # candidate for an eraser), in the memory's own ranking
                    retrieved = [(sk, v) for sk, v in retrieved
                                 if any((h.get("features") or {}).get("kind") == seen_kind for h in sk["history"])][:2]
                    if not retrieved:
                        row.update(outcome="no skill for this kind", gt=False)
                        continue
                    skill, verdict = retrieved[0]
                    sandbox = Sandbox(env, skills={s["function"]: s["code"] for s, _ in retrieved},
                                      helpers=memory.helper_code())
                    env.steps, env.step_budget = 0, ATTEMPT_STEPS
                    t0 = time.time()
                    ok, error = sandbox.run(f'skills.{skill["function"]}("{thing}", "{zone}")')
                    env.step_budget = env.steps + 800
                    check = env.check_on_zone(thing, zone)
                    gt = env.private_evaluate({obj: task.goal[obj]})["objects"][obj]
                    row.update(outcome="ran", skill=skill["function"], verdict=verdict["verdict"],
                               reasons=verdict.get("reasons"), program_ok=ok,
                               error=error and (error.get("reason") or error.get("message")),
                               camera=bool(check.get("satisfied")),
                               gt=bool(gt["inside"] and gt["resting"] and gt["upright"]),
                               used_skills=used_skills(sandbox.trace), used_helpers=used_helpers(sandbox.trace),
                               seconds=round(time.time() - t0, 1))
                    print(task.task_id, kind[obj], skill["function"], verdict["verdict"], "ok" if ok else row["error"],
                          "camera", row["camera"], "gt", row["gt"], flush=True)
                env.step_budget = env.steps + 2000
                final = env.private_evaluate(task.goal)
                desk["success"] = final["success"]
                # final ground truth of every thing out of place (one moved early may be knocked out later)
                for r in rows:
                    if r["task"] == task.task_id:
                        d = final["objects"].get(r["object"], {})
                        r["gt_end"] = bool(d.get("inside") and d.get("resting") and d.get("upright"))
                if env.frames:
                    imageio.mimwrite(f"{out}/{task.task_id}.mp4", env.frames, fps=20)
            except Exception as exc:  # keep going; the failure is recorded
                desk["error"] = f"{type(exc).__name__}: {exc}"
                print(task.task_id, "ERROR", desk["error"], flush=True)
            finally:
                env.close()
            desk["seconds"] = round(time.time() - started, 1)
            desks.append(desk)
            json.dump({"rows": rows, "desks": desks}, open(f"{out}/rows.json", "w"), indent=1, default=str)
    summary = {"by_kind": defaultdict(lambda: {"things": 0, "ran": 0, "gt": 0, "gt_end": 0}),
               "by_verdict": defaultdict(lambda: {"ran": 0, "gt": 0}),
               "by_desk_top": defaultdict(lambda: {"things": 0, "gt_end": 0}),
               "desks": len(desks), "whole_desks": sum(bool(d.get("success")) for d in desks)}
    for r in rows:
        k = summary["by_kind"][r["kind"]]
        k["things"] += 1
        k["ran"] += r["outcome"] == "ran"
        k["gt"] += bool(r.get("gt"))
        k["gt_end"] += bool(r.get("gt_end"))
        if r["outcome"] == "ran":
            v = summary["by_verdict"][r["verdict"]]
            v["ran"] += 1
            v["gt"] += bool(r["gt"])
        t = summary["by_desk_top"][r.get("desk")]
        t["things"] += 1
        t["gt_end"] += bool(r.get("gt_end"))
    summary["outcomes"] = defaultdict(int)
    for r in rows:
        summary["outcomes"][r["outcome"]] += 1
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

"""Plumbing test with a scripted stand-in LLM (NOT Qwen): plan, code, extract, validate, reuse."""
import json, re, sys
from pathlib import Path
from sma.agent.llm import ScriptedLLM
from sma.agent.loop import run_episode
from sma.env.tasks import make_task
from sma.memory.store import SkillMemory

FRESH = '''
scene = observe()
o = scene["objects"]["{obj}"]
pick("{obj}", o["height"] / 2, o["long_axis_yaw"] + math.pi / 2)
place("{zone}")
'''
SKILL = '''```python
def place_on_zone_v1(obj_name, zone_name):
    scene = observe()
    o = scene["objects"][obj_name]
    pick(obj_name, o["height"] / 2, o["long_axis_yaw"] + math.pi / 2)
    place(zone_name)
```
{"object_kinds": ["cube"], "requires_zone_free": true, "requires_gripper_empty": true, "notes": "top-down grasp"}'''

def plan(messages):
    text = messages[0]["content"]
    scene = json.loads(text.split("Current scene (from the camera):\n")[1].split("\n\nDecide")[0])
    zone_of = {"cube": "purple_zone", "cup": "cyan_zone", "book": "orange_zone"}
    on_a_zone = lambda o: any(abs(o["center"][0] - z["center"][0]) < z["half_size"][0] and
                              abs(o["center"][1] - z["center"][1]) < z["half_size"][1] for z in scene["zones"].values())
    steps = [{"object": n, "zone": zone_of[o["kind"]] if "Organize" in text else re.search(r"on the (\w+) zone", text).group(1) + "_zone"}
             for n, o in sorted(scene["objects"].items(), key=lambda kv: not on_a_zone(kv[1]))]
    return json.dumps({"steps": steps})

def code(messages):
    text = messages[0]["content"]
    obj, zone = re.search(r'move the object "(\w+)" so it rests on the zone "(\w+)"', text).groups()
    if "skills.place_on_zone_v1" in text and "verdict APPLICABLE" in text:
        return f'skills.place_on_zone_v1("{obj}", "{zone}")'
    return FRESH.format(obj=obj, zone=zone)

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
memory = SkillMemory(out / "memory.json")
for task in (make_task("place", 21, kinds=["cube"]), make_task("place", 22, kinds=["book"]), make_task("clear", 1100)):
    llm = ScriptedLLM({"plan": plan, "code": code, "extract": SKILL})
    rec = run_episode(task, llm, memory, out / task.task_id)
    print(task.task_id, "GT", rec["private_eval"]["success"], "llm", llm.usage()["by_purpose"],
          "skills used", [a["used_skills"] for s in rec["subgoals"] for a in s["attempts"]])
print(json.dumps([{k: s[k] for k in ("function", "status")} | {"history": [(h["source"], h["features"]["kind"], h["outcome"]) for h in s["history"]]} for s in memory.skills], indent=1))
print("applicability log", memory.applicability_log)

"""Plumbing test with a scripted stand-in LLM (NOT Qwen): plan, code, extract, validate, reuse."""
import json, re, sys
from pathlib import Path
from sma.agent.llm import ScriptedLLM
from sma.agent.loop import run_episode
from sma.env.tasks import make_task
from sma.memory.store import SkillMemory

FRESH = '''
scene = observe()
obj = scene["objects"]["{obj}"]
zone = scene["zones"]["{zone}"]
t = scene["table_z"]
grip = t + min(obj["height"] - 0.015, 0.045)
yaw = obj["long_axis_yaw"] + math.pi / 2
open_gripper()
move_to(obj["center"][0], obj["center"][1], obj["top_z"] + 0.1, yaw)
move_to(obj["center"][0], obj["center"][1], grip, yaw)
close_gripper()
move_to(obj["center"][0], obj["center"][1], t + 0.18, yaw)
move_to(zone["center"][0], zone["center"][1], t + 0.18, yaw)
move_to(zone["center"][0], zone["center"][1], grip + 0.006, yaw)
open_gripper()
move_to(zone["center"][0], zone["center"][1], t + 0.18, yaw)
'''
SKILL = '''```python
def place_on_zone_v1(obj_name, zone_name):
    scene = observe()
    obj = scene["objects"][obj_name]
    zone = scene["zones"][zone_name]
    t = scene["table_z"]
    grip = t + min(obj["height"] - 0.015, 0.045)
    yaw = obj["long_axis_yaw"] + math.pi / 2
    open_gripper()
    move_to(obj["center"][0], obj["center"][1], obj["top_z"] + 0.1, yaw)
    move_to(obj["center"][0], obj["center"][1], grip, yaw)
    close_gripper()
    move_to(obj["center"][0], obj["center"][1], t + 0.18, yaw)
    move_to(zone["center"][0], zone["center"][1], t + 0.18, yaw)
    move_to(zone["center"][0], zone["center"][1], grip + 0.006, yaw)
    open_gripper()
    move_to(zone["center"][0], zone["center"][1], t + 0.18, yaw)
```
{"object_kinds": ["cube"], "requires_zone_free": true, "requires_gripper_empty": true, "notes": "top-down grasp"}'''

def plan(messages):
    text = messages[0]["content"]
    scene = json.loads(text.split("Current scene (from the camera):\n")[1].split("\n\nDecide")[0])
    zone_of = {"cube": "purple_zone", "cup": "cyan_zone", "book": "orange_zone"}
    steps = [{"object": n, "zone": zone_of[o["kind"]] if "Organize" in text else re.search(r"on the (\w+) zone", text).group(1) + "_zone"}
             for n, o in scene["objects"].items()]
    return json.dumps({"steps": steps})

def code(messages):
    text = messages[0]["content"]
    obj, zone = re.search(r'move the object "(\w+)" so it rests on the zone "(\w+)"', text).groups()
    if "skills.place_on_zone_v1" in text and "verdict APPLICABLE" in text:
        return f'skills.place_on_zone_v1("{obj}", "{zone}")'
    return FRESH.format(obj=obj, zone=zone)

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
memory = SkillMemory(out / "memory.json")
for task in (make_task("place", 21, kinds=["cube"]), make_task("place", 22, kinds=["cube"])):
    llm = ScriptedLLM({"plan": plan, "code": code, "extract": SKILL})
    rec = run_episode(task, llm, memory, out / task.task_id)
    print(task.task_id, "GT", rec["private_eval"]["success"], "llm", llm.usage()["by_purpose"],
          "skills used", [a["used_skills"] for s in rec["subgoals"] for a in s["attempts"]])
print(json.dumps([{k: s[k] for k in ("function", "status")} | {"history": [(h["source"], h["features"]["kind"], h["outcome"]) for h in s["history"]]} for s in memory.skills], indent=1))
print("applicability log", memory.applicability_log)

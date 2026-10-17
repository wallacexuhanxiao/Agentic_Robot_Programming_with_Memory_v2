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
{"object_kinds": ["block"], "requires_zone_free": true, "requires_gripper_empty": true, "notes": "top-down grasp"}'''

def refine(messages):
    text = messages[0]["content"]
    request = re.search(r'User request: "(.*?)"', text).group(1)
    objects = json.loads(re.search(r"\(name, kind, and the zone each currently lies on\): (\[.*?\])\n", text).group(1))
    zone_of = {"block": "purple_box", "mug": "cyan_tray", "book": "orange_tray"}
    goals = []
    for o in sorted(objects, key=lambda o: o["on_zone"] is None):
        m = re.search(r"(?:on|in) the (\w+) (tray|box)", request)
        zone = f"{m.group(1)}_{m.group(2)}" if m and "Put the" in request else zone_of[o["kind"]]
        goals.append({"object": o["name"], "zone": zone, "source": "assumption", "evidence": "scripted"})
    return json.dumps({"goals": goals, "refined_instruction": request})

def plan(messages):
    text = messages[0]["content"]
    goals = json.loads(re.search(r"Goals: (\[.*?\])\n", text).group(1))
    return json.dumps({"steps": goals})

def code(messages):
    text = messages[0]["content"]
    obj, zone = re.search(r'move the object "(\w+)" so it rests on the zone "(\w+)"', text).groups()
    if "skills.place_on_zone_v1" in text and "verdict APPLICABLE" in text:
        return f'skills.place_on_zone_v1("{obj}", "{zone}")'
    return FRESH.format(obj=obj, zone=zone)

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
memory = SkillMemory(out / "memory.json")
for task in (make_task("place", 21, kinds=["block"]), make_task("place", 22, kinds=["book"]), make_task("place", 23, kinds=["book"]), make_task("place", 24, kinds=["mug"]), make_task("clear", 1100)):
    llm = ScriptedLLM({"refine": refine, "plan": plan, "code": code, "extract": SKILL})
    rec = run_episode(task, llm, memory, out / task.task_id)
    print(task.task_id, "GT", rec["private_eval"]["success"], "llm", llm.usage()["by_purpose"],
          "skills used", [a["used_skills"] for s in rec["subgoals"] for a in s["attempts"]])
print(json.dumps([{k: s[k] for k in ("function", "status")} | {"history": [(h["source"], h["features"]["kind"], h["outcome"]) for h in s["history"]]} for s in memory.skills], indent=1))
print("applicability log", memory.applicability_log)

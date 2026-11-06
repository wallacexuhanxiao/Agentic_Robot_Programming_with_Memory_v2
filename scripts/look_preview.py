"""Pictures of desks as the cameras see them, to look at the scene's appearance (no LLM): for the first N desks of each
suite, the top-down camera (1024 px) and the demo camera (1280 x 960), and the images a model is shown with outlined,
numbered things and close-ups (SAM2 masks on sdesk desks).
Usage (segmentation service running): look_preview.py OUT_DIR N SUITE [SUITE ...]"""
import os
import sys

import imageio.v2 as imageio

from sma.agent.loop import make_env
from sma.env.tasks import suite

out, n = sys.argv[1], int(sys.argv[2])
os.makedirs(out, exist_ok=True)
for name in sys.argv[3:]:
    for task in suite(name)[:n]:
        env = make_env(task, video=False)
        try:
            env.reset()
            tag = f"{out}/{name.split('@')[0]}_{task.seed}"
            imageio.imwrite(f"{tag}_topdown.png", env.render("topdown", 1024, 1024))
            imageio.imwrite(f"{tag}_demo.png", env.render("demo", 1280, 960))
            scene = env.observe()
            labels = {t: str(i + 1) for i, t in enumerate(sorted(scene["things"], key=lambda t: (len(t), t)))}
            for v in env.labelled_views(labels=labels, marks="som"):
                imageio.imwrite(f"{tag}_som_{v['camera']}.png", v["image"])
            print(task.task_id, len(scene["things"]), "things", flush=True)
        finally:
            env.close()

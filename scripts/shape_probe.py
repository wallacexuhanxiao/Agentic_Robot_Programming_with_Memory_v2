"""Shape features of the perceived objects at the start of a few layouts (no LLM, no motion), to choose a rule that
tells a paper ball from a toy block: realdesk layouts (paper balls, some partly hidden in the wastebasket) and desk
layouts (toy blocks). Simulator state is used only for the true kind of each object.
Usage: shape_probe.py OUT.json SUITE [SUITE ...]"""
import json
import sys

import numpy as np

from sma.agent.loop import make_env
from sma.env.perception import _fill_ratio
from sma.env.tasks import suite

rows = []
for name in sys.argv[2:]:
    for task in suite(name):
        truth = {o.color: o.kind for o in task.layout.objects}
        env = make_env(task, video=False)
        try:
            scene = env.reset()
            for obj in scene["objects"].values():
                mask = env._masks.get(obj["color"])
                if mask is None or truth.get(obj["color"]) not in ("paper", "block"):
                    continue
                z = env._masks["_xyz"][mask][:, 2] - scene["table_z"]
                top = np.quantile(z, .9)
                rows.append({"task": task.task_id, "color": obj["color"], "true_kind": truth[obj["color"]],
                             "perceived_kind": obj["kind"], "pixels": int(mask.sum()), "fill": round(_fill_ratio(mask), 3),
                             "height": obj["height"], "footprint": [obj["footprint_long"], obj["footprint_short"]],
                             **{f"top_share_{int(b * 1000)}mm": round(float(np.mean(z > top - b)), 3) for b in (.002, .003, .006)},
                             "max_minus_p90_mm": round(float(z.max() - top) * 1000, 2),
                             "std_upper_half_mm": round(float(np.std(z[z >= np.median(z)])) * 1000, 2)})
                print(json.dumps(rows[-1]), flush=True)
        finally:
            env.close()
json.dump(rows, open(sys.argv[1], "w"), indent=1)

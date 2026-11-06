"""What Contact-GraspNet returns for the realdesk objects (needs the GraspNet service, so a GPU job): for each object
at the start of a few layouts, how many grasps, how far each approach is tilted from straight down, their scores,
and how far the grip point is from the object's centre. The top-down candidates keep only grasps within 30 degrees
of vertical; this shows how many that leaves.
Usage: graspnet_probe.py OUT.json SUITE [SUITE ...]"""
import json
import os
import sys
import time

import numpy as np

from sma.agent.loop import make_env
from sma.env.tasks import suite

rows = []
for name in sys.argv[2:]:
    for task in suite(name):
        env = make_env(task, video=False)
        try:
            scene = env.reset()
            for obj in list(scene["objects"].values()):
                t0 = time.time()
                if os.environ.get("SMA_PROBE_PATH", "candidates") == "candidates":  # the path the program uses
                    cands = env.grasp_candidates(obj["name"])
                    gn = [c for c in cands if c["source"] == "contact-graspnet"]
                    rows.append({"task": task.task_id, "object": obj["name"], "note": cands[0].get("graspnet_note") if cands else None,
                                 "seconds": round(time.time() - t0, 1), "graspnet_candidates": len(gn),
                                 "tilts": [c["tilt_deg"] for c in gn], "scores": [c["score"] for c in gn],
                                 "hand_clear": [c.get("hand_clear") for c in gn],
                                 "fits": [c.get("fits_open_fingers") for c in gn]})
                    print(json.dumps(rows[-1]), flush=True)
                    continue
                points = env._masks["_xyz"][env._masks[obj["color"]]] if obj["color"] in env._masks else None
                grasps, note = env._graspnet_raw(points)
                down = env.down.as_matrix()[:, 2]
                info = []
                for m, score in grasps or []:
                    tilt = float(np.degrees(np.arccos(np.clip(m[:3, 2] @ down, -1, 1))))
                    off = float(np.hypot(*(m[:2, 3] - obj["center"][:2])))
                    info.append((round(tilt, 1), round(score, 3), round(off, 3), round(float(m[2, 3] - scene["table_z"]), 3)))
                info.sort()
                rows.append({"task": task.task_id, "object": obj["name"], "kind": obj["kind"], "note": note,
                             "seconds": round(time.time() - t0, 1), "points": None if points is None else len(points),
                             "grasps": len(info), "within_30deg": sum(t <= 30 for t, *_ in info),
                             "within_45deg": sum(t <= 45 for t, *_ in info),
                             "tilt_score_offset_height": info[:12]})
                print(json.dumps(rows[-1]), flush=True)
        finally:
            env.close()
json.dump(rows, open(sys.argv[1], "w"), indent=1)

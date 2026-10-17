"""Compare colour perception and Grounded-SAM2 perception against private object poses (evaluation only)."""
import json, os, sys
import numpy as np
import imageio.v2 as imageio
from sma.env.robot_env import TabletopEnv
from sma.env.tasks import suite
from sma.env.perception import perceive
from sma.env.perception_gsam import perceive_gsam, _request

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
rows = []
for task in suite("session") + suite("train"):
    env = TabletopEnv(task, video=False, image_size=512)
    env.reset()
    rgb, depth = env.render("topdown", 512, 512, depth=True)
    depth_m = env._metric_depth(depth)
    truth = env.private_state()
    true_zones = {z.name: z.xy for z in task.layout.zones}
    res = {"color": perceive(rgb, depth_m, env.K, env.world_from_camera),
           "gsam": perceive_gsam(rgb, depth_m, env.K, env.world_from_camera, os.environ["SMA_SEG_URL"])}
    row = {"task": task.task_id, "objects": sorted(truth), "zones": sorted(true_zones)}
    for name, scene in res.items():
        found = {n: round(1000 * float(np.linalg.norm(np.array(scene["objects"][n]["center"][:2]) - truth[n]["position"][:2])), 1)
                 for n in truth if n in scene["objects"]}
        zfound = {n: round(1000 * float(np.linalg.norm(np.array(scene["zones"][n]["center"][:2]) - np.array(true_zones[n]))), 1)
                  for n in true_zones if n in scene["zones"]}
        row[name] = {"objects_found": f"{len(found)}/{len(truth)}", "xy_err_mm": found,
                     "zones_found": f"{len(zfound)}/{len(true_zones)}", "zone_err_mm": zfound,
                     "extra_objects": sorted(set(scene["objects"]) - set(truth)),
                     "raw": scene.get("perception", {}).get("raw_detections")}
    rows.append(row)
    print(json.dumps(row), flush=True)
    # mask overlay for the first few layouts
    if len(rows) <= 6:
        dets = _request(os.environ["SMA_SEG_URL"], rgb)["detections"]
        over = rgb.copy().astype(float)
        rng = np.random.default_rng(0)
        for d in dets:
            c = rng.integers(60, 255, 3)
            over[d["mask"]] = .45 * over[d["mask"]] + .55 * c
            x0, y0, x1, y1 = map(int, d["box"])
            import cv2
            cv2.rectangle(over, (x0, y0), (x1, y1), c.tolist(), 1)
            cv2.putText(over, f"{d['label']} {d['score']:.2f}", (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, .35, c.tolist(), 1)
        imageio.imwrite(f"{out}/{task.task_id}_gsam.png", np.hstack([rgb, over.astype(np.uint8)]))
    env.close()
json.dump(rows, open(f"{out}/perception_compare.json", "w"), indent=1)
for k in ("color", "gsam"):
    o = sum(int(r[k]["objects_found"].split("/")[0]) for r in rows); ot = sum(int(r[k]["objects_found"].split("/")[1]) for r in rows)
    z = sum(int(r[k]["zones_found"].split("/")[0]) for r in rows); zt = sum(int(r[k]["zones_found"].split("/")[1]) for r in rows)
    errs = [e for r in rows for e in r[k]["xy_err_mm"].values()]
    print(f"SUMMARY {k}: objects named correctly {o}/{ot}, containers {z}/{zt}, median xy error {np.median(errs) if errs else None} mm, extra objects {sum(len(r[k]['extra_objects']) for r in rows)}")

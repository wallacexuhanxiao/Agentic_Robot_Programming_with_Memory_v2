"""Name tracking on the shared-colour desk, for diagnosis (no LLM): runs a gate script (SMA_TRACK_GATE: notebook_gate.py
by default, or sam_gate.py) for each desk index with the camera's perception wrapped, so that every look records the
tracks before matching (mean colour, centre, size), the held and just-released track, the search for a thing just let
go of, the names given and the identity, and saves the top-down image with each thing's outline and name.
Usage (segmentation service running): track_debug.py OUT_DIR INDEX [INDEX ...] (an index may repeat)"""
import json
import os
import runpy
import sys

import cv2
import numpy as np

from sma.env import perception_sam
from sma.env.robot_env import TabletopEnv

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
original = TabletopEnv._perceive_neutral
original_expected = perception_sam._expected_mask
expected_log = []


def traced_expected(kept, spot, height, valid, rgb, url, masks):
    """The search for a thing just let go of: where it was expected, the thing found there and the mask chosen."""
    v, u, size = spot
    owner = next((t for t in kept if t["mask"][v, u]), None)
    mask = original_expected(kept, spot, height, valid, rgb, url, masks)
    expected_log.append({"pixel": [int(u), int(v)], "expected_pixels": round(float(size), 1),
                         "owner_pixels": owner and owner["area"], "chosen_pixels": None if mask is None else int(mask.sum())})
    return mask


perception_sam._expected_mask = traced_expected


def brief(t):
    return {"lab": np.round(np.asarray(t["lab"], float), 1).tolist(), "center": [round(float(c), 3) for c in t["center"]],
            "pixels": int(t["visible_pixels"])}


def traced(self, rgb, depth_m):
    before = {n: brief(t) for n, t in self._tracks.items()}
    held, released = self._held_track, self._released
    expected_log.clear()
    scene = original(self, rgb, depth_m)
    look = len(self.track_log)
    self.track_log.append({"look": look, "steps": self.steps, "end_effector_xy": self._ee()[:2].round(3).tolist(),
                           "held": held, "released": released, "expected_search": list(expected_log),
                           "released_here": getattr(self, "here_log", []),
                           "tracks_before": before,
                           "seen": {n: brief(self._tracks[n]) for n in scene["things"]},
                           "identity": {n: v.get("category") for n, v in self.identity.items()}})
    image = np.ascontiguousarray(rgb[..., ::-1])
    for n in scene["things"]:
        mask = self._masks.get(n)
        if mask is None:
            continue
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 255), 1)
        ys, xs = np.nonzero(mask)
        label = f"{n} {self.identity.get(n, {}).get('category') or ''}".strip()
        cv2.putText(image, label, (int(xs.mean()) - 20, int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
    cv2.imwrite(f"{self.track_dir}/look{look}.png", image)
    return scene


original_track = TabletopEnv._track


def traced_track(self, things):
    """Notes the thing perception marked as seen where the hand just let go of something."""
    self.here_log = [{"center": [round(float(c), 3) for c in t["center"]], "pixels": int(t["visible_pixels"])}
                     for t in things if t.get("released_here")]
    return original_track(self, things)


TabletopEnv._perceive_neutral = traced
TabletopEnv._track = traced_track
gate = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("SMA_TRACK_GATE", "notebook_gate.py"))
desks = sys.argv[2:]
for pos, index in enumerate(desks):  # an index given again (a repeat) gets its own directories
    tag = f"desk{index}" + (f"_{pos}" if desks.count(index) > 1 else "")
    init = TabletopEnv.__init__

    def start(self, *a, _tag=tag, **k):
        init(self, *a, **k)
        self.track_log, self.track_dir = [], f"{out}/{_tag}"
        os.makedirs(self.track_dir, exist_ok=True)

    TabletopEnv.__init__ = start
    envs = []
    close = TabletopEnv.close

    def keep(self):
        envs.append(self)
        close(self)

    TabletopEnv.close = keep
    sys.argv = [gate, f"{out}/gate" + tag[len(f"desk{index}"):], index]
    try:
        runpy.run_path(gate, run_name="__main__")
    finally:
        TabletopEnv.__init__, TabletopEnv.close = init, close
    for env in envs:
        json.dump(env.track_log, open(f"{env.track_dir}/tracks.json", "w"), indent=1)
        print("desk", index, "looks", len(env.track_log), flush=True)

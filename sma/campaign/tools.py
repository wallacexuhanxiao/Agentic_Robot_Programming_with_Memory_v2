"""Worker tools for the ASPIRE-style campaign: look at a scene, run a candidate program on development seeds,
inspect a run. Every episode starts from a fresh simulator. The success signal given to the agent is the
camera (perception) check; simulator ground truth is recorded separately for scoring only."""
import base64
import io
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from ..agent.loop import make_env
from ..agent.sandbox import Sandbox
from ..env.robot_env import RobotError
from ..env.tasks import make_task

EPISODE_STEPS = 3000


def png_b64(rgb, size=None):
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    if size:
        img = img.resize(size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class TaskSpec:
    """One campaign task: a task family plus its seed partition."""

    def __init__(self, name, family, kinds, dev_seeds, unseen_seeds, heldout_seeds):
        self.name, self.family, self.kinds = name, family, kinds
        self.dev_seeds, self.unseen_seeds, self.heldout_seeds = dev_seeds, unseen_seeds, heldout_seeds

    def episode(self, seed):
        return make_task(self.family, seed, kinds=self.kinds)


def run_episode(task_spec, seed, code, out_dir, keyframes=4):
    """Execute one program on one fresh seed. Returns a public result (perception) and a private one (GT)."""
    task = task_spec.episode(seed)
    env = make_env(task)
    started = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        env.reset()
        start_view = env.render("demo", 480, 360)
        sandbox = Sandbox(env)
        env.steps, env.step_budget = 0, EPISODE_STEPS
        ok, error = sandbox.run(code)
        env.step_budget = env.steps + 1500
        checks = {obj: env.check_on_zone(obj, zone) for obj, zone in task.goal.items()}
        end_view = env.render("demo", 480, 360)
        private = env.private_evaluate(task.goal)
        frames = env.frames
    except RobotError as exc:
        ok, error, checks, private, frames = False, {"type": "harness", "reason": exc.reason}, {}, {"success": False}, env.frames
        start_view = end_view = None
    finally:
        env.close()
    if frames:
        imageio.mimwrite(out_dir / "video.mp4", frames, fps=20)
    picks = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, keyframes)] if frames else []
    for i, f in enumerate(([start_view] if start_view is not None else []) + picks + ([end_view] if end_view is not None else [])):
        imageio.imwrite(out_dir / f"key_{i}.png", f)
    trace = [{k: t.get(k) for k in ("call", "args", "kwargs", "result", "error", "line")} for t in sandbox.trace] \
        if "sandbox" in locals() else []
    public = {"seed": seed, "instruction": task.instruction, "program_ok": ok, "error": error,
              "camera_success": bool(checks) and all(c.get("satisfied") for c in checks.values()),
              "camera_checks": {o: {k: c.get(k) for k in ("satisfied", "inside_zone", "resting", "offset_from_zone_center",
                                                           "reason")} for o, c in checks.items()},
              "calls": len(trace), "seconds": round(time.time() - started, 1)}
    (out_dir / "public.json").write_text(json.dumps(public, indent=1, default=str))
    (out_dir / "trace.json").write_text(json.dumps(trace, indent=1, default=str))
    (out_dir / "private_gt.json").write_text(json.dumps(private, indent=1, default=str))
    return public, private


class WorkerTools:
    def __init__(self, task_spec, workdir):
        self.spec = task_spec
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.runs = []  # (version, program, per-seed public results)

    # ---- tool: snapshot -------------------------------------------------------------------
    def snapshot(self):
        """Fresh scene of the first development seed: task language, base-frame summary, two images."""
        task = self.spec.episode(self.spec.dev_seeds[0])
        env = make_env(task, video=False)
        try:
            env.reset()
            summary = env.observe_base() if hasattr(env, "observe_base") else env.observe()
            top = env.last_rgb
            demo = env.render("demo", 480, 360)
        finally:
            env.close()
        text = json.dumps({"instruction": task.instruction, "scene": summary}, default=str)
        return text, [png_b64(top), png_b64(demo)]

    # ---- tool: run_program ------------------------------------------------------------------
    def run_program(self, code, seeds=None):
        seeds = list(seeds or self.spec.dev_seeds)
        bad = [s for s in seeds if s not in self.spec.dev_seeds]
        if bad:
            return f"refused: seeds {bad} are not development seeds {self.spec.dev_seeds}", []
        version = len(self.runs) + 1
        vdir = self.workdir / f"v{version:02d}"
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "program.py").write_text(code)
        results = []
        for seed in seeds:
            public, _ = run_episode(self.spec, seed, code, vdir / f"seed_{seed}")
            results.append(public)
        self.runs.append({"version": version, "program": code, "results": results})
        score = sum(r["camera_success"] for r in results)
        lines = [f"version v{version:02d}: {score}/{len(results)} development seeds succeeded (camera check)"]
        for r in results:
            err = r["error"] and (r["error"].get("reason") or r["error"].get("message"))
            lines.append(f"- seed {r['seed']}: {'SUCCESS' if r['camera_success'] else 'FAIL'}"
                         f"{'  error=' + str(err)[:200] if err else ''}  checks={json.dumps(r['camera_checks'])[:400]}")
        return "\n".join(lines), []

    # ---- tool: inspect_run --------------------------------------------------------------------
    def inspect_run(self, version, seed):
        sdir = self.workdir / f"v{int(version):02d}" / f"seed_{int(seed)}"
        if not sdir.exists():
            return f"no run v{int(version):02d} seed {seed}", []
        trace = json.loads((sdir / "trace.json").read_text())
        public = json.loads((sdir / "public.json").read_text())
        rows = []
        for t in trace[:40]:
            res = t.get("error") or t.get("result")
            rows.append(f"line {t.get('line')}: {t['call']}({', '.join(map(repr, t.get('args') or []))[:120]})"
                        f" -> {json.dumps(res, default=str)[:220]}")
        keys = sorted(sdir.glob("key_*.png"))
        images = [png_b64(imageio.imread(k), (320, 240)) for k in keys[:: max(1, len(keys) // 4)][:4]]
        text = (f"run v{int(version):02d} seed {seed}: {json.dumps({k: public[k] for k in ('camera_success', 'error')}, default=str)}\n"
                "call trace:\n" + "\n".join(rows) + "\nimages: keyframes in time order (demo camera)")
        return text, images

    def best_version(self):
        if not self.runs:
            return None
        return max(self.runs, key=lambda r: (sum(x["camera_success"] for x in r["results"]), r["version"]))

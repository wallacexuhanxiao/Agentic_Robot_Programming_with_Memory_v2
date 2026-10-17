"""ASPIRE-style campaign coordinator with an open model.

For each task in curriculum order: worker (explore, write, debug on development seeds, propose skills) ->
one measurement on unseen seeds -> held-out evaluation (scored with simulator ground truth, never shown to the
model) -> verifier re-solves from the library + proposals only -> verified proposals are promoted to the library.
State is saved after every step so a campaign can resume across PBS jobs (the interactive queue caps a job at
1.5 h).

Usage: python -m sma.campaign.coordinator --root <campaign dir> --base-url <vLLM url> --model <name>
                                          [--no-library] [--deadline-min 80]
"""
import argparse
import json
import time
from pathlib import Path

from ..agent.llm import LLM
from .tools import TaskSpec, WorkerTools, run_episode
from .worker import run_worker

# Curriculum: simple single-object placements first, then a dependency (clear an occupied container first).
TASKS = [
    TaskSpec("place-block", "place", ["block"], [101, 102, 103], [126, 127], list(range(201, 209))),
    TaskSpec("place-mug", "place", ["mug"], [104, 105, 106], [128, 129], list(range(211, 219))),
    TaskSpec("place-book", "place", ["book"], [107, 108, 109], [130, 131], list(range(221, 229))),
    TaskSpec("sort-two", "sort", None, [110, 112, 114], [132, 134], list(range(232, 248, 2))),
]


class Campaign:
    def __init__(self, root, llm, use_library=True, log=print):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.llm, self.use_library, self.log = llm, use_library, log
        self.state_path = self.root / "campaign.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "use_library": use_library, "library": [], "tasks": {}}

    def save(self):
        self.state_path.write_text(json.dumps(self.state, indent=1, default=str))

    def evaluate(self, spec, program, seeds, label):
        rows = []
        for seed in seeds:
            public, private = run_episode(spec, seed, program, self.root / spec.name / label / f"seed_{seed}", keyframes=2)
            rows.append({"seed": seed, "camera_success": public["camera_success"], "gt_success": private.get("success"),
                         "error": public["error"]})
            self.log(f"    {label} seed {seed}: camera={public['camera_success']} gt={private.get('success')}")
        return rows

    def run(self, deadline):
        for spec in TASKS:
            st = self.state["tasks"].setdefault(spec.name, {"status": "pending"})
            if time.time() > deadline:
                self.log("deadline reached; state saved for resume")
                return False
            library = self.state["library"] if self.use_library else []
            if st["status"] == "pending":
                self.log(f"[worker] {spec.name} (library size {len(library)})")
                tools = WorkerTools(spec, self.root / spec.name / "worker")
                result = run_worker(self.llm, tools, library=library, log=self.log)
                st.update({"status": "worker-done", "worker": result, "library_size_at_start": len(library),
                           "llm_usage": self.llm.usage()})
                self.save()
            if st["status"] == "worker-done":
                program = st["worker"]["program"]
                if program:
                    st["unseen"] = self.evaluate(spec, program, spec.unseen_seeds, "unseen")
                    st["heldout"] = self.evaluate(spec, program, spec.heldout_seeds, "heldout")
                st["status"] = "evaluated"
                self.save()
            if st["status"] == "evaluated":
                proposals = st["worker"].get("proposals") or []
                if self.use_library and proposals and time.time() < deadline:
                    self.log(f"[verifier] {spec.name}: {len(proposals)} proposal(s)")
                    vtools = WorkerTools(TaskSpec(spec.name, spec.family, spec.kinds, spec.dev_seeds[:2], [], []),
                                         self.root / spec.name / "verifier")
                    verdict = run_worker(self.llm, vtools, library=self.state["library"], max_runs=2, max_turns=10,
                                         verifier_proposals=proposals, log=self.log)
                    verified = bool(verdict["dev_score"]) and verdict["dev_score"] == len(vtools.spec.dev_seeds)
                    st["verifier"] = {k: verdict[k] for k in ("dev_score", "runs_used", "turns", "chosen")}
                    st["verifier"]["verified"] = verified
                    if verified:
                        for p in proposals:
                            self.state["library"].append({**p, "source_task": spec.name})
                    self.log(f"    proposals {'VERIFIED and promoted' if verified else 'refuted (not promoted)'}")
                st["status"] = "done"
                self.save()
        return True


def summary(state):
    lines = ["| task | library at start | runs used | dev (camera) | unseen (camera) | held-out GT | verified skills |",
             "|---|---|---|---|---|---|---|"]
    for name, st in state["tasks"].items():
        w = st.get("worker", {})
        unseen = st.get("unseen", [])
        held = st.get("heldout", [])
        lines.append(f"| {name} | {st.get('library_size_at_start', '-')} | {w.get('runs_used', '-')} | {w.get('dev_score', '-')} | "
                     f"{sum(r['camera_success'] for r in unseen)}/{len(unseen)} | "
                     f"{sum(bool(r['gt_success']) for r in held)}/{len(held)} | "
                     f"{'yes' if st.get('verifier', {}).get('verified') else 'no'} |")
    lines.append(f"\nLibrary: {[s['name'] for s in state['library']]}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="general")
    p.add_argument("--no-library", action="store_true")
    p.add_argument("--deadline-min", type=float, default=80)
    args = p.parse_args()
    log_file = (args.root / "campaign.log")
    args.root.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        with log_file.open("a") as f:
            f.write(line + "\n")

    llm = LLM(args.base_url, model=args.model, log_dir=args.root / "llm", temperature=0.3)
    campaign = Campaign(args.root, llm, use_library=not args.no_library, log=log)
    finished = campaign.run(time.time() + 60 * args.deadline_min)
    (args.root / "SUMMARY.md").write_text(summary(campaign.state) + ("\n" if finished else "\n(campaign incomplete)\n"))
    log(summary(campaign.state))


if __name__ == "__main__":
    main()

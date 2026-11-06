"""Offline evaluation of a VLM success check (Gemini through the OpenAI-compatible endpoint) on saved camera views.
For every goal object of every episode, the last saved check view (the zoomed top-down camera image taken right
before the episode's private evaluation) is shown to the VLM with a yes/no question. Its answers are compared with
the simulator ground truth and with the rule-based camera check. No simulation runs here.

Usage: SMA_LLM_PROVIDER=gemini SMA_API_KEY_FILE=... python scripts/vlm_check_eval.py OUT_DIR RUN_DIR [RUN_DIR ...]
       [--model gemini-3.8-flash] [--max-usd 1.5] [--limit N] [--dry-run]
RUN_DIR is a condition directory holding task directories (e.g. outputs/<run>/skill)."""
import argparse
import base64
import json
import os
import sys
from pathlib import Path

ZONE_WORDS = {"green_case": "green pen case (a narrow box with low walls)", "cyan_tray": "cyan tray (a flat tray with a low rim)",
              "orange_tray": "orange tray (a flat tray with a low rim)", "purple_box": "purple toy box (a box with walls)",
              "magenta_holder": "magenta pen holder (a small cup with tall walls)"}
QUESTION_HOLDER = """This is a top-down camera image of part of a table. Is the {object} standing upright inside the {zone}:
its lower end inside the cup and its body pointing up out of the cup (leaning against the cup's wall is fine), rather
than lying flat across the top of the cup or on the table? The gripper must not be holding it.
Answer with one line of JSON: {{"inside": true or false, "reason": "<one short sentence>"}}"""
QUESTION = """This is a top-down camera image of part of a table. Is the {object} lying inside the {zone}: fully within
its walls or rim (not resting on top of a wall or rim, not hanging over the edge, not standing on another object) and
lying flat? The gripper must not be holding it.
Answer with one line of JSON: {{"inside": true or false, "reason": "<one short sentence>"}}"""

p = argparse.ArgumentParser()
p.add_argument("out", type=Path)
p.add_argument("runs", nargs="+", type=Path)
p.add_argument("--model", default="gemini-3.8-flash")
p.add_argument("--max-usd", default="1.5")
p.add_argument("--limit", type=int, default=0)
p.add_argument("--dry-run", action="store_true", help="list the items and stop (no API calls)")
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SMA_SPEND_LEDGER", str(args.out / "spend.json"))
os.environ["SMA_SPEND_CAP_USD"] = args.max_usd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sma.agent.llm import LLM, SpendCapReached  # noqa: E402

items = []
for run in args.runs:
    for ep in sorted(run.glob("*/episode.json")):
        views = ep.parent / "check_views" / "index.json"
        if not views.exists():
            continue
        record, index = json.loads(ep.read_text()), json.loads(views.read_text())
        gt = (record.get("private_eval") or {}).get("objects", {})
        for obj, truth in gt.items():
            last = [v for v in index if v["object"] == obj]
            if last:
                v = last[-1]
                items.append({"episode": str(ep.parent), "object": obj, "zone": v["zone"], "image": str(ep.parent / "check_views" / v["file"]),
                              "gt": bool(truth["inside"] and truth["resting"] and truth["upright"]), "rule_check": v["rule_check"]})
if args.limit:
    items = items[:args.limit]
print(f"{len(items)} judgements from {len(args.runs)} run directories", flush=True)
if args.dry_run:
    sys.exit(0)

llm = LLM("", model=args.model, log_dir=args.out / "llm")
results = []
for it in items:
    color, kind = it["object"].split("_", 1)
    image = base64.b64encode(Path(it["image"]).read_bytes()).decode()
    question = QUESTION_HOLDER if it["zone"].endswith("holder") else QUESTION
    content = [{"type": "text", "text": question.format(object=f"{color} {kind}", zone=ZONE_WORDS.get(it["zone"], it["zone"]))},
               {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}]
    try:
        text = llm.chat([{"role": "user", "content": content}], purpose="vlm_check", max_tokens=200)
    except SpendCapReached as exc:
        print("stopped:", exc, flush=True)
        break
    line = next((l for l in reversed(text.strip().splitlines()) if l.strip().startswith("{")), "{}")
    try:
        answer = json.loads(line)
    except json.JSONDecodeError:
        answer = {}
    it.update({"vlm": answer.get("inside"), "vlm_reason": answer.get("reason"), "raw": text[-300:]})
    results.append(it)
    print(it["object"], it["zone"], "GT", it["gt"], "rule", it["rule_check"], "VLM", it["vlm"], "|", it["vlm_reason"], flush=True)


def agreement(key):
    judged = [r for r in results if isinstance(r[key], bool)]
    return {"agree": sum(r[key] == r["gt"] for r in judged), "judged": len(judged),
            "false_positive": sum(r[key] and not r["gt"] for r in judged),
            "false_negative": sum((not r[key]) and r["gt"] for r in judged)}


summary = {"model": args.model, "items": len(results), "gt_success": sum(r["gt"] for r in results),
           "rule_check_vs_gt": agreement("rule_check"), "vlm_vs_gt": agreement("vlm"),
           "unparsed_vlm_answers": sum(not isinstance(r["vlm"], bool) for r in results), "llm_usage": llm.usage()}
(args.out / "results.json").write_text(json.dumps(results, indent=1))
(args.out / "summary.json").write_text(json.dumps(summary, indent=1))
print("SUMMARY", json.dumps(summary), flush=True)

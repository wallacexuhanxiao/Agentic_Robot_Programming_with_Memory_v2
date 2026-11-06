"""Small trial of image-reading Gemini models (no simulation). (1) list the models the key can use; (2) find out whether
the key's project is on the free tier (gemini-4-argon has no free tier); (3) ask each candidate model, for a balanced
sample of finished pen episodes, whether the pen lies inside the green pen case, using the last frame of the episode
video (demo camera, cropped to the table and enlarged), and compare with the simulator ground truth.
The demo camera is far from the table, so this is a rough test of image reading, not the final check view.
Usage: SMA_LLM_PROVIDER=gemini SMA_API_KEY_FILE=... python scripts/vlm_free_trial.py OUT_DIR"""
import base64
import glob
import io
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sma.agent.llm import LLM, SpendCapReached  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
key = Path(os.environ["SMA_API_KEY_FILE"]).read_text().strip()
# Candidates: free tier and image input according to the pricing page (USD per 1M tokens at paid-tier list prices).
MODELS = {"gemini-3.8-flash": (0.75, 3.75), "gemini-robotics-er-2-preview": (1.0, 5.0), "gemini-3.5-flash-lite": (0.30, 2.50)}

request = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                                 headers={"x-goog-api-key": key})
with urllib.request.urlopen(request, timeout=60) as reply:
    names = sorted(m["name"].split("/")[-1] for m in json.loads(reply.read()).get("models", []))
print("models visible to this key:", len(names), flush=True)
print("candidates present:", {m: m in names for m in MODELS}, "| gemini-4-argon listed:", "gemini-4-argon" in names, flush=True)

os.environ.update(SMA_SPEND_LEDGER=str(out / "tier_check_spend.json"), SMA_PRICE_IN="2.0", SMA_PRICE_OUT="10.0")
try:  # Argon is not offered on the free tier: an error naming the free tier means this key is on it
    reply = LLM("", model="gemini-4-argon").chat([{"role": "user", "content": "Reply with OK."}], purpose="tier", max_tokens=20)
    tier = f"paid (gemini-4-argon answered: {reply!r})"
except RuntimeError as exc:
    tier = f"free tier likely (gemini-4-argon refused: {str(exc)[:200]})"
print("TIER:", tier, flush=True)

eps = []
for f in sorted(glob.glob("outputs/penrep_*/*/pen_3*/episode.json")):
    e = json.load(open(f))
    if (Path(f).parent / "video.mp4").exists():
        eps.append((f, bool(e["private_eval"]["success"])))
rng = random.Random(0)
sample = rng.sample([e for e in eps if e[1]], 6) + rng.sample([e for e in eps if not e[1]], 6)
images = []
for f, gt in sample:
    reader = imageio.get_reader(str(Path(f).parent / "video.mp4"))
    frame = reader.get_data(reader.count_frames() - 1)
    crop = Image.fromarray(frame[115:315, 110:390]).resize((840, 600), Image.LANCZOS)  # the table top, 3x
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    name = Path(f).parent.parent.name + "_" + Path(f).parent.name
    crop.save(out / f"{name}.png")
    pen = next(iter(json.load(open(f))["private_eval"]["objects"]))
    images.append({"episode": name, "pen": pen, "gt": gt, "png": base64.b64encode(buf.getvalue()).decode()})

QUESTION = ("This image shows a white table seen from above at an angle. Is the {pen} lying inside the green pen case "
            "(a narrow green box with low walls), fully between its walls, rather than across it, on its walls or "
            "beside it? Answer with one line of JSON: {{\"inside\": true or false, \"reason\": \"<short>\"}}")
summary = {"tier": tier}
for model, (pin, pout) in MODELS.items():
    if model not in names:
        summary[model] = "not available to this key"
        continue
    os.environ.update(SMA_SPEND_LEDGER=str(out / f"spend_{model}.json"), SMA_PRICE_IN=str(pin), SMA_PRICE_OUT=str(pout),
                      SMA_SPEND_CAP_USD="0.3")
    llm = LLM("", model=model, log_dir=out / "llm" / model)
    rows = []
    for im in images:
        color = im["pen"].split("_")[0]
        content = [{"type": "text", "text": QUESTION.format(pen=f"{color} pen")},
                   {"type": "image_url", "image_url": {"url": "data:image/png;base64," + im["png"]}}]
        try:
            text = llm.chat([{"role": "user", "content": content}], purpose="vlm_trial", max_tokens=200)
        except SpendCapReached:
            break
        except RuntimeError as exc:
            if "reasoning" in str(exc).lower() and llm.effort:
                llm.effort = ""  # this model takes no thinking setting; ask again without it
                text = llm.chat([{"role": "user", "content": content}], purpose="vlm_trial", max_tokens=200)
            else:
                rows.append({"episode": im["episode"], "error": str(exc)[:200]})
                continue
        line = next((l for l in reversed(text.strip().splitlines()) if l.strip().startswith("{")), "{}")
        try:
            answer = json.loads(line).get("inside")
        except json.JSONDecodeError:
            answer = None
        rows.append({"episode": im["episode"], "gt": im["gt"], "vlm": answer, "answer": text.strip()[-200:]})
        print(model, im["episode"], "GT", im["gt"], "VLM", answer, "|", text.strip()[-120:].replace("\n", " "), flush=True)
    judged = [r for r in rows if isinstance(r.get("vlm"), bool)]
    usage = llm.usage()
    summary[model] = {"correct": sum(r["vlm"] == r["gt"] for r in judged), "judged": len(judged), "asked": len(images),
                      "false_positive": sum(r["vlm"] and not r["gt"] for r in judged),
                      "false_negative": sum((not r["vlm"]) and r["gt"] for r in judged),
                      "errors": [r["error"] for r in rows if "error" in r][:2],
                      "seconds_per_call": round(usage["seconds"] / max(1, usage["calls"]), 1),
                      "list_price_usd": usage["cost_usd"]}
    (out / f"rows_{model}.json").write_text(json.dumps(rows, indent=1))
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print("SUMMARY", json.dumps(summary), flush=True)

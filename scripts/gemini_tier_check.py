"""Which tier is the key in SMA_API_KEY_FILE on? Lists the models it can see, asks gemini-4-argon (no free tier) and
gemini-3.5-flash-lite (free tier) for one word each. Prints only model names and outcomes, never the key."""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sma.agent.llm import LLM  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
key = Path(os.environ["SMA_API_KEY_FILE"]).read_text().strip()
print("key file:", os.environ["SMA_API_KEY_FILE"], "| key length:", len(key), "| prefix shape:", key[:4].isalnum(), flush=True)
try:
    request = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                                     headers={"x-goog-api-key": key})
    with urllib.request.urlopen(request, timeout=60) as reply:
        names = sorted(m["name"].split("/")[-1] for m in json.loads(reply.read()).get("models", []))
        print("models visible:", len(names), "| 3.x flash:", [n for n in names if n.startswith("gemini-3")], flush=True)
except urllib.error.HTTPError as exc:
    print("model list failed: HTTP", exc.code, exc.read()[:300], flush=True)
CANDIDATES = [m for m in os.environ.get("SMA_TIER_MODELS", "gemini-4-argon:gemini-3.5-flash-lite").split(":") if m]
for model, price in ((m, ("2.0", "10.0")) for m in CANDIDATES):
    os.environ.update(SMA_SPEND_LEDGER=str(out / f"spend_{model}.json"), SMA_PRICE_IN=price[0], SMA_PRICE_OUT=price[1])
    try:
        text = LLM("", model=model).chat([{"role": "user", "content": "Reply with OK."}], purpose="tier", max_tokens=20)
        print(f"{model}: answered {text.strip()!r}", flush=True)
    except RuntimeError as exc:
        print(f"{model}: refused: {str(exc)[:300]}", flush=True)

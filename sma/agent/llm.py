"""OpenAI-compatible chat client for the vLLM-served Qwen3.5-9B, with full call logging."""
import json
import time
import urllib.request
from pathlib import Path


class LLM:
    def __init__(self, base_url, model="Qwen/Qwen3.5-9B", log_dir=None, temperature=0.2):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.log_dir = Path(log_dir) if log_dir else None
        self.calls = []

    def chat(self, messages, *, purpose, max_tokens=2500):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": self.temperature, "chat_template_kwargs": {"enable_thinking": False}}
        started = time.time()
        request = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            reply = json.loads(response.read())
        choice = reply["choices"][0]
        text = choice["message"].get("content") or ""
        record = {"index": len(self.calls), "purpose": purpose, "seconds": round(time.time() - started, 2),
                  "finish_reason": choice.get("finish_reason"), "usage": reply.get("usage"),
                  "messages": messages, "response": text}
        self.calls.append(record)
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / f"call_{record['index']:03d}_{purpose}.json").write_text(
                json.dumps(record, indent=1, ensure_ascii=False))
        return text

    def last_truncated(self):
        return bool(self.calls) and self.calls[-1]["finish_reason"] == "length"

    def usage(self):
        tokens = [c["usage"] or {} for c in self.calls]
        return {"calls": len(self.calls),
                "prompt_tokens": sum(t.get("prompt_tokens", 0) for t in tokens),
                "completion_tokens": sum(t.get("completion_tokens", 0) for t in tokens),
                "seconds": round(sum(c["seconds"] for c in self.calls), 1),
                "by_purpose": {p: sum(1 for c in self.calls if c["purpose"] == p)
                               for p in sorted({c["purpose"] for c in self.calls})}}


class ScriptedLLM(LLM):
    """Offline stand-in that replays canned responses per purpose (for plumbing tests only)."""

    def __init__(self, responses, log_dir=None):
        super().__init__("http://offline", log_dir=log_dir)
        self.responses = responses

    def chat(self, messages, *, purpose, max_tokens=2500):
        queue = self.responses.get(purpose)
        text = queue.pop(0) if isinstance(queue, list) else queue
        if callable(text):
            text = text(messages)
        self.calls.append({"index": len(self.calls), "purpose": purpose, "seconds": 0.0,
                           "finish_reason": "stop", "usage": {}, "messages": messages, "response": text})
        return text

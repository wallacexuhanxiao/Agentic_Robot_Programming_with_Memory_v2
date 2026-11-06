"""OpenAI-compatible chat client with full call logging: the vLLM-served Qwen model on Hopper, or a hosted API through
its OpenAI-compatible endpoint (SMA_LLM_PROVIDER=gemini or deepseek), with a spending ledger and a hard cap."""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


CODE_PURPOSES = ("code", "extract")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEEPSEEK_URL = "https://api.deepseek.com"  # deepseek-flash reads images (image_url parts, as for Gemini)


class SpendCapReached(RuntimeError):
    """The run's API spending reached SMA_SPEND_CAP_USD; the run stops instead of continuing to spend."""


class SpendLedger:
    """Running API cost of one run, kept in a JSON file so it survives the per-task clients and crash restarts.
    Prices are USD per 1M tokens; output includes thinking tokens (billed as output). "usd" counts only calls made
    with a billed (paid-tier) key; "list_price_usd" counts every call at list price. The cap applies to "usd".
    A second, project-wide ledger (SMA_GLOBAL_LEDGER) adds up billed spend across runs against SMA_GLOBAL_CAP_USD."""

    def __init__(self, path, cap_usd, price_in, price_out, global_path=None, global_cap=None):
        self.path, self.cap = Path(path), float(cap_usd)
        self.price_in, self.price_out = float(price_in), float(price_out)
        self.global_path = Path(global_path) if global_path else None
        self.global_cap = float(global_cap) if global_cap else None

    @staticmethod
    def _read(path):
        return json.loads(path.read_text()) if path.exists() else {}

    def spent(self):
        return self._read(self.path).get("usd", 0.0)

    def check(self):
        if self.spent() >= self.cap:
            raise SpendCapReached(f"API spending {self.spent():.4f} USD reached the run cap of {self.cap} USD")
        if self.global_path and self.global_cap is not None:
            total = self._read(self.global_path).get("usd", 0.0)
            if total >= self.global_cap:
                raise SpendCapReached(f"project API spending {total:.4f} USD reached the budget cap of {self.global_cap} USD")

    def _update(self, path, cost, billed, prompt_tokens, output_tokens, extra=None):
        import fcntl
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path) + ".lock", "w") as lock:  # several jobs may add to the project-wide ledger at once
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self._read(path) or {"usd": 0.0, "list_price_usd": 0.0, "calls": 0, "billed_calls": 0,
                                         "prompt_tokens": 0, "output_tokens": 0}
            state["usd"] = round(state["usd"] + (cost if billed else 0.0), 6)
            state["list_price_usd"] = round(state.get("list_price_usd", 0.0) + cost, 6)
            state["calls"] += 1
            state["billed_calls"] = state.get("billed_calls", 0) + int(billed)
            state["prompt_tokens"] += int(prompt_tokens)
            state["output_tokens"] += int(output_tokens)
            state.update(extra or {})
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=1))
            tmp.replace(path)

    def add(self, prompt_tokens, output_tokens, billed=True):
        cost = (prompt_tokens * self.price_in + output_tokens * self.price_out) / 1e6
        self._update(self.path, cost, billed, prompt_tokens, output_tokens,
                     {"cap_usd": self.cap, "price_per_million": {"input": self.price_in, "output": self.price_out}})
        if self.global_path and billed:
            self._update(self.global_path, cost, billed, prompt_tokens, output_tokens, {"cap_usd": self.global_cap})
        return cost


class LLM:
    """Routes code-writing purposes to an optional coder endpoint; everything else to the general model."""

    def __init__(self, base_url, model="Qwen/Qwen3.5-9B", log_dir=None, temperature=0.2,
                 coder_url=None, coder_model=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.coder_url = coder_url.rstrip("/") if coder_url else None
        self.coder_model = coder_model
        self.temperature = temperature
        self.log_dir = Path(log_dir) if log_dir else None
        self.calls = []
        # Gemini: the key is read from a file (never logged); spending goes to the run's ledger and is capped.
        provider = os.environ.get("SMA_LLM_PROVIDER")
        self.gemini, self.deepseek = provider == "gemini", provider == "deepseek"
        self.remote = self.gemini or self.deepseek
        self.ledger = None
        if self.remote:
            self.base_url = os.environ.get("SMA_LLM_URL", GEMINI_URL if self.gemini else DEEPSEEK_URL).rstrip("/")
            self._key = Path(os.environ["SMA_API_KEY_FILE"]).read_text().strip()
            # A free-tier key costs nothing; SMA_API_KEY_FILE_FALLBACK (a paid key) takes over if the free tier keeps
            # refusing (rate limits, overload, timeouts) after all retries.
            self.billed = os.environ.get("SMA_API_KEY_TIER", "paid") != "free"
            self.fallback_file = os.environ.get("SMA_API_KEY_FILE_FALLBACK")
            self.effort = os.environ.get("SMA_REASONING_EFFORT", "low")
            self.ledger = SpendLedger(os.environ["SMA_SPEND_LEDGER"], os.environ.get("SMA_SPEND_CAP_USD", "1.5"),
                                      os.environ.get("SMA_PRICE_IN", "0.75"), os.environ.get("SMA_PRICE_OUT", "3.75"),
                                      os.environ.get("SMA_GLOBAL_LEDGER"), os.environ.get("SMA_GLOBAL_CAP_USD", "9.5"))

    def chat(self, messages, *, purpose, max_tokens=2500, thinking=False, think_room=8000, no_thinking=False):
        """think_room: tokens allowed for a hosted model's thinking on top of max_tokens; no_thinking: ask a hosted
        DeepSeek model not to think at all (e.g. a retry after a call that thought until the limit and said nothing)."""
        use_coder = self.coder_url is not None and purpose in CODE_PURPOSES
        url, model = (self.coder_url, self.coder_model) if use_coder else (self.base_url, self.model)
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": self.temperature}
        headers = {"Content-Type": "application/json"}
        if self.remote:
            self.ledger.check()
            # Both APIs think by default (deepseek-flash at effort "high") and count the thinking against max_tokens:
            # the same low effort for both, and room so the answer is not cut off (deepseek-flash spent all 1500 tokens
            # of an identification call thinking and returned no answer).
            body["max_tokens"] = max_tokens + think_room
            if self.deepseek and (no_thinking or os.environ.get("SMA_DEEPSEEK_THINKING") == "disabled"):  # none
                body["thinking"] = {"type": "disabled"}
            elif self.effort not in ("", "none"):  # some models take no thinking setting
                body["reasoning_effort"] = self.effort
            headers["Authorization"] = f"Bearer {self._key}"
        elif not use_coder:
            body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        started = time.time()
        try:
            reply = self._post(url + "/chat/completions", body, headers)
        except RuntimeError as exc:
            # Free tier still refusing after every retry: switch this run to the paid key (if one is given).
            if not (self.gemini and not self.billed and self.fallback_file) or "HTTP 4" in str(exc) and "429" not in str(exc):
                raise
            self._key, self.billed = Path(self.fallback_file).read_text().strip(), True
            os.environ["SMA_API_KEY_FILE"], os.environ["SMA_API_KEY_TIER"] = self.fallback_file, "paid"  # later tasks too
            print(f"[llm] free tier unavailable ({str(exc)[:120]}); switching to the paid key", flush=True)
            self.ledger.check()
            headers["Authorization"] = f"Bearer {self._key}"
            reply = self._post(url + "/chat/completions", body, headers)
        choice = reply["choices"][0]
        text = choice["message"].get("content") or ""
        record = {"index": len(self.calls), "purpose": purpose, "model": model,
                  "seconds": round(time.time() - started, 2),
                  "finish_reason": choice.get("finish_reason"), "usage": reply.get("usage"),
                  "messages": messages, "response": text}
        if self.remote:
            usage = reply.get("usage") or {}
            prompt = usage.get("prompt_tokens", 0)
            output = max(usage.get("total_tokens", 0) - prompt, usage.get("completion_tokens", 0))  # incl. thinking
            cost = self.ledger.add(prompt, output, billed=self.billed)
            record["cost_usd"], record["billed"] = round(cost if self.billed else 0.0, 6), self.billed
            record["list_price_usd"] = round(cost, 6)
        # Cut off because the thinking outgrew its room (or left no answer at all): ask the same again without
        # thinking instead of handing back a cut answer (deepseek-flash at effort low still thought 10500 tokens and
        # wrote no program in 9 of 22 code calls on sdesk, 2026-11-06).
        reasoning = ((reply.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        retry = (self.remote and self.deepseek and not no_thinking and choice.get("finish_reason") == "length"
                 and (not text.strip() or reasoning > think_room))
        record["retried_without_thinking"] = retry
        self.calls.append(record)
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / f"call_{record['index']:03d}_{purpose}.json").write_text(
                json.dumps(record, indent=1, ensure_ascii=False))
        if retry:
            return self.chat(messages, purpose=purpose, max_tokens=max_tokens, thinking=thinking, think_room=think_room,
                             no_thinking=True)
        return text

    def _post(self, url, body, headers, tries=6):
        """POST with backoff on rate limits, transient server errors, timeouts and dropped connections (the free tier
        limits requests per minute; a remote call can also hang). The local vLLM server keeps the long timeout."""
        timeout = 120 if self.remote else 600
        for attempt in range(tries):
            request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                    raise RuntimeError(f"LLM HTTP {exc.code}: {exc.read()[:300]!r}") from None
            except (TimeoutError, urllib.error.URLError, ConnectionError) as exc:
                if attempt == tries - 1:
                    raise RuntimeError(f"LLM request failed after {tries} tries: {type(exc).__name__}: {exc}") from None
            time.sleep(min(60, 5 * 2 ** attempt))

    def last_truncated(self):
        return bool(self.calls) and self.calls[-1]["finish_reason"] == "length"

    def usage(self):
        tokens = [c["usage"] or {} for c in self.calls]
        return {"calls": len(self.calls),
                "prompt_tokens": sum(t.get("prompt_tokens", 0) for t in tokens),
                "completion_tokens": sum(t.get("completion_tokens", 0) for t in tokens),
                "seconds": round(sum(c["seconds"] for c in self.calls), 1),
                "by_purpose": {p: sum(1 for c in self.calls if c["purpose"] == p)
                               for p in sorted({c["purpose"] for c in self.calls})},
                "cost_usd": round(sum(c.get("cost_usd", 0.0) for c in self.calls), 6),
                "list_price_usd": round(sum(c.get("list_price_usd", 0.0) for c in self.calls), 6),
                "models": sorted({c.get("model") or "" for c in self.calls})}


class ScriptedLLM(LLM):
    """Offline stand-in that replays canned responses per purpose (for plumbing tests only)."""

    def __init__(self, responses, log_dir=None):
        super().__init__("http://offline", log_dir=log_dir)
        self.responses = responses

    def chat(self, messages, *, purpose, max_tokens=2500, thinking=False, think_room=8000, no_thinking=False):
        queue = self.responses.get(purpose, "")
        text = queue.pop(0) if isinstance(queue, list) else queue
        if callable(text):
            text = text(messages)
        self.calls.append({"index": len(self.calls), "purpose": purpose, "seconds": 0.0,
                           "finish_reason": "stop", "usage": {}, "messages": messages, "response": text})
        return text

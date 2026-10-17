"""ASPIRE-style worker and verifier for an open model: a multi-turn tool loop (JSON action per turn, code in a
fenced block) that explores a scene, writes a task-level program, debugs it on development seeds, and proposes
skills. Condensed from ASPIRE's iterative-debugging subagent and verifier prompts."""
import json
import re

from ..agent import prompts

WORKER_SYSTEM = """You are a fix-loop worker for a code-as-policy robot benchmark (modelled on NVIDIA's ASPIRE).
Your job: for ONE task, write a Python program that controls a Franka Panda through the API below, make it work on
the development seeds, and report what you learned. Each seed is a different random table layout of the same task.

Program rules (results are invalid otherwise):
- No imports except the module math (already available). Set a top-level variable `result`.
- Every task-dependent choice (which object, which container, grasp and target poses, sizes) must be derived at
  runtime from get_task_context().language and perception. Never copy observed coordinates, colours, sizes or
  seed-specific behaviour into constants. Generic relative offsets, tolerances and step limits are fine.
- The instruction names objects as "<colour> <kind>" and containers as "<colour> tray" / "<colour> box";
  perception names them "<colour>_<kind>" and "<colour>_tray" / "<colour>_box".

{api}

{library}

How to work (one action per reply):
1. snapshot once to see the task language, the scene summary and two images.
2. Write an initial program and run_program it on the development seeds.
3. Debug by failure mode, not by seed: inspect_run failing seeds (call trace + keyframes), form a hypothesis,
   change one thing, run again on ALL development seeds. Keep the version that succeeds on the most seeds.
4. When satisfied (or out of runs), write_findings (what failed, what fixed it), optionally propose_skill for a
   reusable pattern (trigger -> code -> evidence), then submit the version you choose.
Budget: at most {max_runs} run_program calls and {max_turns} replies in total.

Reply format: first ONE JSON object on its own line, then (only for run_program / propose_skill) the program in
ONE ```python block IN THE SAME REPLY. A run_program reply without the program is wasted.
  {{"thought": "<short reasoning>", "tool": "snapshot"}}
  {{"thought": "...", "tool": "run_program"}}   followed by ```python <the full program> ```
  {{"thought": "...", "tool": "inspect_run", "args": {{"version": 2, "seed": 101}}}}
  {{"thought": "...", "tool": "write_findings", "args": {{"text": "<what failed, what fixed it>"}}}}
  {{"thought": "...", "tool": "propose_skill", "args": {{"name": "<kebab-name>", "trigger": "<when to use>",
     "evidence": "<which runs show it works>"}}}}   followed by ```python <the skill code or pattern> ```
  {{"thought": "...", "tool": "submit", "args": {{"version": 2}}}}"""

VERIFIER_EXTRA = """
You are a VERIFIER. Another worker solved this task and proposed the skills shown under "Proposed skills". You do
not see its program. Solve the task using only the API, the library and the proposals. You have at most
{max_runs} run_program calls. Submit the version you believe works."""


def parse_action(text):
    match = re.search(r"\{.*?\}(?=\s*(```|$))", text, re.S) or re.search(r"\{.*\}", text, re.S)
    action = {}
    if match:
        try:
            action = json.loads(match.group(0))
        except json.JSONDecodeError:
            depth, start = 0, text.find("{")
            for i, ch in enumerate(text[start:], start):  # first balanced object
                depth += ch == "{"
                depth -= ch == "}"
                if depth == 0:
                    try:
                        action = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        action = {}
                    break
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return action, (blocks[-1].strip() if blocks else None)


def _content(text, images):
    if not images:
        return text
    return [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b}}
                                                for b in images]


def library_text(library):
    if not library:
        return "Skill library: empty (nothing verified yet)."
    parts = ["Skill library (verified skills from earlier tasks; use them when their trigger matches):"]
    for s in library:
        parts.append(f"### {s['name']}\nTrigger: {s['trigger']}\nEvidence: {s['evidence']}\n```python\n{s['code']}\n```")
    return "\n".join(parts)


def run_worker(llm, tools, *, library, max_runs=6, max_turns=24, verifier_proposals=None, log=print):
    system = WORKER_SYSTEM.format(api=prompts.API_DOC, library=library_text(library), max_runs=max_runs,
                                  max_turns=max_turns)
    if verifier_proposals is not None:
        system += VERIFIER_EXTRA.format(max_runs=max_runs) + "\n\nProposed skills:\n" + library_text(verifier_proposals)
    assignment = (f"Task: {tools.spec.name}. Development seeds: {tools.spec.dev_seeds}. "
                  "Start with snapshot.")
    history = []          # (assistant text, observation content) pairs
    ledger = []           # one line per run, always shown
    findings, proposals, submitted = None, [], None
    runs = 0
    for turn in range(max_turns):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": assignment + ("\nRun ledger so far:\n" + "\n".join(ledger) if ledger else "")}]
        for i, (said, seen) in enumerate(history[-5:]):  # keep the last exchanges; images only in the newest
            messages.append({"role": "assistant", "content": said})
            last = i == len(history[-5:]) - 1
            messages.append({"role": "user", "content": seen if last or isinstance(seen, str) else seen[0]["text"]})
        reply = llm.chat(messages, purpose="worker", max_tokens=4000)
        action, code = parse_action(reply)
        tool = action.get("tool")
        args = action.get("args") or {}
        code = code or args.get("code") or action.get("code")
        if tool in ("run_program", "propose_skill") and not code:
            # Two-step for open models: they often stop after the JSON. Ask for the code alone.
            follow = llm.chat(messages + [{"role": "assistant", "content": reply},
                                          {"role": "user", "content": "Now write the complete program for this action "
                                           "in ONE ```python block, with nothing else."}],
                              purpose="worker_code", max_tokens=3000)
            _, code = parse_action(follow)
            if code is None and "def " in follow or (code is None and "=" in follow):
                code = follow.strip().strip("`")
            reply = reply + "\n```python\n" + (code or "") + "\n```"
        log(f"    turn {turn + 1}: {tool} {json.dumps(args)[:120]}")
        images = []
        if tool == "snapshot":
            text, images = tools.snapshot()
        elif tool == "run_program":
            if runs >= max_runs:
                text = f"refused: run budget of {max_runs} is used up; write_findings and submit."
            elif not code:
                text = "error: put the full program in a ```python block after the JSON."
            else:
                runs += 1
                text, images = tools.run_program(code, args.get("seeds"))
                ledger.append(text.splitlines()[0])
        elif tool == "inspect_run":
            try:
                text, images = tools.inspect_run(args.get("version"), args.get("seed"))
            except (TypeError, ValueError):
                text = "error: inspect_run needs integer version and seed"
        elif tool == "write_findings":
            findings = args.get("text")
            text = "findings recorded"
        elif tool == "propose_skill":
            if code:
                proposals.append({"name": args.get("name", f"skill-{len(proposals) + 1}"), "trigger": args.get("trigger", ""),
                                  "evidence": args.get("evidence", ""), "code": code})
                text = "proposal recorded"
            else:
                text = "error: put the skill code in a ```python block after the JSON."
        elif tool == "submit":
            submitted = args.get("version")
            break
        else:
            text = ("error: could not read an action. Reply with ONE JSON object like "
                    '{"thought": "...", "tool": "snapshot"} and, for run_program, a ```python block.')
        history.append((reply, _content(f"Tool result ({tool}):\n{text}", images)))
    chosen = next((r for r in tools.runs if r["version"] == submitted), None) or tools.best_version()
    return {"submitted_version": submitted, "chosen": chosen and chosen["version"], "program": chosen and chosen["program"],
            "dev_score": chosen and sum(x["camera_success"] for x in chosen["results"]),
            "runs_used": runs, "turns": turn + 1, "findings": findings, "proposals": proposals, "ledger": ledger}

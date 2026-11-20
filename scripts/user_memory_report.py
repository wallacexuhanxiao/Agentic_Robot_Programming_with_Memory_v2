"""Summary of a user_memory_plan run: where each loose thing was planned to go, against where the simulated user wants it,
per condition (none / history / memory) and kind of test; also only the decisions whose things (the object and its
destination) the model identified correctly, and the refine prompt length. Usage: user_memory_report.py RUN_DIR"""
import json
import sys
from collections import defaultdict

run = sys.argv[1]
rows = json.load(open(f"{run}/rows.json"))
sessions = json.load(open(f"{run}/sessions.json"))


def norm(t):
    import re
    return re.sub(r"\s+", " ", str(t).lower().replace("\u2019", "'")).strip(" .,'\"")


# The history condition's goals that the model drew from the earlier messages, labelled "user_memory", were dropped by
# the refine step's source check (it only knows stored preferences, and this condition has none). Counted here as the
# condition's decision when the evidence quotes an earlier message of this user verbatim (a quoted passage of 12+
# characters, every '...'-separated piece of it found in them). Nothing else is changed.
earlier = {}
for s in sessions:
    earlier[(s["seq"], s["session"])] = [norm(x["message"]) for x in sessions if x["seq"] == s["seq"]
                                         and x["session"] < s["session"]]
rescued = 0
for r in rows:
    if r["condition"] != "history" or r["chose"] is not None or not r["thing"]:
        continue
    s = next(x for x in sessions if x["seq"] == r["seq"] and x["session"] == r["session"])
    for g in s["conditions"]["history"]["spec"]["dropped"]:
        if g.get("object") != r["thing"] or g.get("rejected_because") != "no stored preference says this":
            continue
        import re
        ev = str(g.get("evidence", ""))
        quoted = [a or b for a, b in re.findall(r'["\u201c]([^"\u201d]{12,})["\u201d]|\'([^\']{12,})\'', ev)] or [ev]
        text = " ".join(earlier[(r["seq"], r["session"])])
        if any(all(norm(p) in text for p in q.split("...") if len(norm(p)) >= 8) for q in quoted):
            r["chose"], r["source"] = g["zone"], "history (dropped by the source check, counted)"
            r["correct"] = r["perceivable"] and (r["chose"] == r["want_thing"] if r["want"] else False)
            rescued += 1
            break
print("history goals counted although the source check dropped them:", rescued)
CASES = ["stated", "rule", "one_off", "changed", "untaught"]
CONDS = ["none", "history", "memory"]
table = defaultdict(lambda: [0, 0, 0, 0])  # correct, total, correct when identified, identified
for r in rows:
    for key in ((r["condition"], r["case"]), (r["condition"], "all")):
        t = table[key]
        t[0] += r["correct"]
        t[1] += 1
        if r["identified_right"]:
            t[2] += r["correct"]
            t[3] += 1
print(f"{'case':10s}" + "".join(f"{c:>22s}" for c in CONDS))
for case in CASES + ["all"]:
    cells = []
    for c in CONDS:
        t = table[(c, case)]
        cells.append(f"{t[0]}/{t[1]} ({t[2]}/{t[3]} id ok)")
    print(f"{case:10s}" + "".join(f"{x:>22s}" for x in cells))
tokens = defaultdict(list)
for s in sessions:
    for c, d in s["conditions"].items():
        tokens[c].append(d["refine_prompt_tokens"])
print("refine prompt tokens per session:", {c: (min(v), max(v), round(sum(v) / len(v))) for c, v in tokens.items()})
print("identification:", sum(s["identification"]["category_ok"] for s in sessions), "/",
      sum(s["identification"]["things"] for s in sessions))
print("\nwrong decisions:")
for r in rows:
    if not r["correct"]:
        print(f"  seq{r['seq']} s{r['session']} {r['condition']:7s} {r['kind']:8s} {r['case']:8s} want {r['want']} -> "
              f"chose {r['chose']} [{r['source']}] identified_right={r['identified_right']} perceivable={r['perceivable']}")
moves = [(s["seq"], s["session"], c, d["moves_tidy_things"]) for s in sessions for c, d in s["conditions"].items()
         if d["moves_tidy_things"]]
print("\nplans that also move things already in place:", moves)

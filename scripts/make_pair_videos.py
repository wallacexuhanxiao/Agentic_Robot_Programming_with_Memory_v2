"""Side-by-side comparison videos: same task, no memory (left) vs memory (right). Usage: python scripts/make_pair_videos.py results"""
import json, sys, textwrap
import cv2, numpy as np, imageio.v2 as imageio
R = sys.argv[1]
W, H, TOP = 480, 368, 120
WHITE, GREY, GREEN, RED, CYAN, YELLOW = (235, 235, 235), (160, 160, 160), (90, 220, 110), (255, 95, 95), (120, 220, 235), (255, 220, 60)

def put(img, text, xy, color=WHITE, scale=.42):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

def load(path):
    cap = cv2.VideoCapture(path); frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(cv2.resize(f, (W, H)), cv2.COLOR_BGR2RGB))
    return frames

def header(lines):
    img = np.full((TOP, W, 3), 22, np.uint8)
    for i, (t, c) in enumerate(lines):
        put(img, t[:70], (8, 20 + 19 * i), c, .42)
    return img

def describe(d, task, label):
    ep = json.load(open(f"{R}/{d}/{task}/episode.json"))
    ok = ep["private_eval"]["success"]
    atts = [a for s in ep["subgoals"] for a in s["attempts"]]
    reuse = sum(a.get("direct_reuse", False) for a in atts)
    goals = ep.get("task_spec", {}).get("goals", [])
    lines = [(label, CYAN), (f"ground truth: {'SUCCESS' if ok else 'FAIL'}  |  {len(atts)} execution attempt(s)", GREEN if ok else RED)]
    if reuse:
        lines.append((f"{reuse} attempt(s) ran a learned skill from memory (no new code)", YELLOW))
    if goals and len({g['source'] for g in goals}) and any(g["source"] != "instruction" for g in goals):
        txt = ", ".join(f"{g['object']}->{g['zone'].replace('_', ' ')}" for g in goals)
        src = "/".join(sorted({g["source"] for g in goals}))
        for k, chunk in enumerate(textwrap.wrap(f"plan ({src}): {txt}", 62)[:3]):
            lines.append(("  " * (k > 0) + chunk, WHITE))
    return lines

def pair(left, right, task, title_lines):
    lf, rf = load(f"{R}/{left[0]}/{task}/video.mp4"), load(f"{R}/{right[0]}/{task}/video.mp4")
    n = max(len(lf), len(rf))
    lf += [lf[-1]] * (n - len(lf)); rf += [rf[-1]] * (n - len(rf))
    hl, hr = header(describe(left[0], task, left[1])), header(describe(right[0], task, right[1]))
    out = []
    card = np.full((TOP + H, 2 * W, 3), 25, np.uint8)
    for i, (t, c) in enumerate(title_lines):
        put(card, t, (24, 60 + 30 * i), c, .62 if i == 0 else .5)
    out += [card] * 50
    for a, b in zip(lf, rf):
        out.append(np.hstack([np.vstack([hl, a]), np.vstack([hr, b])]))
    out += [out[-1]] * 30
    return out

def main(spec, dest):
    frames = []
    for p in spec:
        frames += pair(p["left"], p["right"], p["task"], p["title"])
    imageio.mimwrite(dest, frames, fps=20, macro_block_size=8)
    print(dest, len(frames), "frames", round(len(frames) / 20, 1), "s")

skill = [
 {"task": f"pen_{n}", "left": (f"penrep_{s}_none_20261017T172651Z/none", "NO MEMORY"), "right": (f"penrep_{s}_skill_20261017T172651Z/skill", "SKILL MEMORY"),
  "title": [(f"Skill memory on an alignment-critical placement: layout set {s}, task pen_{n}", CYAN),
            ('"Put the pen in the green pen case." The pen fits only if turned to the case axis.', WHITE),
            ("Left: no memory, the model writes new code each attempt.", WHITE),
            ("Right: skill memory; a skill extracted from an earlier verified success is reused.", WHITE),
            ("Same layout, same model (Qwen3-30B-A3B), same mid-level API.", GREY)]}
 for s, n in (("A", 2001), ("B", 3004), ("C", 4003))]
user = [
 {"task": t, "left": (f"{run}/none", "NO MEMORY"), "right": (f"{run}/user", "USER MEMORY"),
  "title": [(f"User memory on a vague tidy-up request ({lvl} API): {t}", CYAN),
            (f'Instruction: "{json.load(open(f"{R}/{run}/none/{t}/episode.json"))["instruction"]}"', WHITE),
            ("Earlier the user said once: blocks -> purple box, mugs -> cyan tray, books -> orange tray.", WHITE),
            ("Left: no memory, destinations are guessed (assumption).", WHITE),
            ("Right: user memory supplies each destination with the user's own words as evidence.", WHITE)]}
 for run, lvl, t in (("mid30b_20261017T110309Z", "mid-level", "tidy_1005"), ("final_20261017T084229Z", "task-level", "tidy_1001"))]
main(skill, f"{R}/demo/skill_memory_pairs.mp4")
main(user, f"{R}/demo/user_memory_pairs.mp4")

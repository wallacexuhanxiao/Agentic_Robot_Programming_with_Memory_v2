"""Demo clip for one episode, made locally from its record (no simulator): a title card with what the robot saw (the
things SAM2 separated, outlined, and the categories the model gave them) and the goals it inferred; the recorded video
with a caption per attempt (the subgoal, the attempt, a fresh program or a reused skill, the camera check, held a moment
on screen); a closing card with the camera's verdicts and the offline ground-truth score, which the robot never sees.
The episode's own video.mp4 is left as it is (the raw recording).
Usage: make_demo_video.py EPISODE_DIR OUT.mp4 [en|zh]   (default en)"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
W, H, FPS = 960, 720, 20
TEXT = {
    "en": {"header": "Realistic desk · SAM2 + depth perception · LLM-written programs · no ground truth for the robot",
           "title": "A desk it has never seen: the robot looks, recognizes and decides by itself",
           "instruction": "Instruction: ", "left": "Left: the things SAM2 separated (numbered)",
           "recognized": "Objects and receptacles the model recognized:", "goals": "Goals it inferred:", "join": ", ",
           "goal": "{obj} → {zone} ({source})", "step": "Step {i}: {obj} → {zone} · attempt {n}",
           "reuse": "Reusing stored skill {fn} directly (no LLM call)", "fresh": "Program written by the LLM",
           "calls": ", calling stored skill {fn}", "pass": "Camera check: passed", "fail": "Camera check: failed",
           "why": " ({why})", "result": "Result", "camera": "The robot's own camera checks: {v} of {n} placements confirmed",
           "gt": "Offline ground-truth score (evaluation only, never shown to the robot):", "row": "  {cat}: {state}",
           "placed": "placed", "not_placed": "not placed",
           "kept": "  Items already in place: {k} of {m} still where they were",
           "source": {"instruction": "instruction", "user_memory": "long-term memory",
                      "scene_grouping": "scene grouping", "assumption": "model's assumption"},
           "reason": {"not_visible": "object not visible", "object_not_visible": "object not visible",
                      "motion_blocked": "motion blocked", "program_error": "program error",
                      "unreachable_pose": "pose out of reach", "rejected": "program rejected",
                      "carry_height_out_of_range": "carry height out of range",
                      "target_outside_workspace": "target outside the workspace"},
           "names": {}},
    "zh": {"header": "真实外观桌面 · SAM2 + 深度感知 · DeepSeek 写程序 · 机器人看不到仿真真值",
           "title": "一张没见过的桌子：机器人自己看、自己认、自己决定放哪",
           "instruction": "指令：", "left": "左图：SAM2 分出的东西（编号）",
           "recognized": "模型认出的物品和容器：", "goals": "推断出的整理目标：", "join": "、",
           "goal": "{obj} → {zone}（{source}）", "step": "第 {i} 步：{obj} → {zone} · 第 {n} 次尝试",
           "reuse": "直接复用技能 {fn}（不调用模型）", "fresh": "模型新写的程序", "calls": "，调用已存技能 {fn}",
           "pass": "相机检查：通过", "fail": "相机检查：未通过", "why": "（{why}）", "result": "结果",
           "camera": "机器人自己的相机检查：{n} 个放置中确认了 {v} 个","gt": "离线真值评分（只用于评估，机器人看不到）：",
           "row": "  {cat}：{state}", "placed": "放好了", "not_placed": "没放好",
           "kept": "  原本就放好的 {m} 件：{k} 件仍在原位",
           "source": {"instruction": "指令", "user_memory": "长期记忆", "scene_grouping": "场景分组", "assumption": "模型假设"},
           "reason": {"not_visible": "相机找不到物体", "object_not_visible": "相机找不到物体", "motion_blocked": "运动被挡住",
                      "program_error": "程序出错", "unreachable_pose": "够不到", "rejected": "程序被拒绝",
                      "carry_height_out_of_range": "搬运高度超出范围", "target_outside_workspace": "目标超出工作空间"},
           "names": {"pen": "笔", "pen holder": "笔筒", "crumpled paper ball": "纸团", "wastebasket": "废纸篓", "mug": "杯子",
                     "coaster": "杯垫", "notebook": "笔记本", "pile of books": "书堆", "book": "书", "paper": "纸团",
                     "pencil": "铅笔", "tray": "托盘", "laptop": "笔记本电脑", "smartphone": "手机", "desk lamp": "台灯"}}}


def draw_text(img, lines, xy, size=26, fill=(255, 255, 255), gap=10, right=W - 24):
    """lines: [str or (str, (r, g, b))] drawn top-down from xy on a PIL image. A line too wide to end by `right` is
    drawn up to 15% smaller, and wrapped (at a space, else anywhere) if that is not enough."""
    d = ImageDraw.Draw(img)
    x, y = xy
    for line in lines:
        text, colour = (line, fill) if isinstance(line, str) else line
        s = size
        while s > .85 * size and d.textlength(text, font=ImageFont.truetype(FONT, s)) > right - x:
            s -= 1
        f = ImageFont.truetype(FONT, s)
        while text:
            cut = len(text)
            while cut > 1 and d.textlength(text[:cut], font=f) > right - x:
                cut = text.rfind(" ", 0, cut - 1) if " " in text[:cut - 1] else cut - 1
            cut = max(cut, len(text) - len(text.lstrip()) + 1)  # at least one visible character per line
            d.text((x, y + (size - s) / 2), text[:cut], font=f, fill=colour)
            text = text[cut:].lstrip()
            if text:
                text = "   " + text  # a wrapped line is indented
            y += size + gap
    return y


def caption(frame, lines, header):
    """The frame (RGB array) with the header on top and a translucent band of caption lines at the bottom."""
    img = Image.fromarray(frame).convert("RGBA")
    band = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(band)
    d.rectangle([0, 0, W, 40], fill=(0, 0, 0, 140))
    if lines:
        d.rectangle([0, H - 30 - 44 * len(lines), W, H], fill=(0, 0, 0, 165))
    img = Image.alpha_composite(img, band).convert("RGB")
    draw_text(img, [header], (16, 8), size=20, fill=(225, 225, 225))
    if lines:
        draw_text(img, lines, (24, H - 14 - 44 * len(lines)), size=28, gap=16)
    return np.asarray(img)


def main(ep_dir, out, lang="en"):
    t = TEXT[lang]
    ep_dir = Path(ep_dir)
    e = json.load(open(ep_dir / "episode.json"))
    ident = e["identification"]

    def name(thing, tag=False):
        """The model's category for a thing (translated where known); with its number in the outlined image if tag."""
        cat = (ident.get(thing) or {}).get("category", thing).replace("other: ", "")
        num = (ident.get(thing) or {}).get("tag")
        return t["names"].get(cat, cat) + (f" ({num})" if tag and num else "")

    video = cv2.VideoCapture(str(ep_dir / "video.mp4"))
    frames = []
    while True:
        ok, f = video.read()
        if not ok:
            break
        frames.append(cv2.resize(f[..., ::-1], (W, H), interpolation=cv2.INTER_AREA if f.shape[1] > W else cv2.INTER_CUBIC))
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
    put = lambda rgb: writer.write(np.ascontiguousarray(rgb[..., ::-1]))

    # Title card: what the robot saw and what it decided to do
    card = Image.new("RGB", (W, H), (24, 26, 30))
    seen = Image.open(ep_dir / "identify_topdown.png").convert("RGB")
    seen.thumbnail((500, 500))
    card.paste(seen, (24, 150))
    draw_text(card, [t["title"]], (24, 30), size=28)
    draw_text(card, [(t["instruction"] + e["instruction"], (190, 200, 215))], (24, 80), size=18)
    draw_text(card, [(t["left"], (190, 200, 215))], (24, 660), size=18, right=524)
    goals = [t["goal"].format(obj=name(g["object"], True), zone=name(g["zone"], True),
                              source=t["source"].get(g["source"], g["source"])) for g in e["task_spec"]["goals"]]
    cats = sorted({name(th) for th, v in ident.items() if v.get("role") in ("object", "receptacle")})
    per = 4 if lang == "zh" else 3
    y = draw_text(card, [t["recognized"]], (548, 150), size=22)
    y = draw_text(card, [t["join"].join(cats[i:i + per]) for i in range(0, len(cats), per)], (548, y + 4), size=19,
                  fill=(200, 210, 225))
    y = draw_text(card, [t["goals"]], (548, y + 24), size=22)
    draw_text(card, goals, (548, y + 4), size=20, fill=(150, 220, 170))
    for _ in range(FPS * 6):
        put(np.asarray(card))

    # The episode with a caption per attempt
    events = []  # (first frame, last program frame, next start, caption lines, check lines)
    entries = e["subgoals"]
    starts = [(i, a) for i, s in enumerate(entries) for a in s["attempts"] if a.get("trace")]
    for k, (i, a) in enumerate(starts):
        s = entries[i]
        first, last = a["trace"][0]["frame"], a["trace"][-1]["frame"]
        nxt = starts[k + 1][1]["trace"][0]["frame"] if k + 1 < len(starts) else len(frames)
        how = (t["reuse"].format(fn=a["used_skills"][0]) if a.get("direct_reuse") else
               t["fresh"] + (t["calls"].format(fn=a["used_skills"][0]) if a.get("used_skills") else ""))
        head = t["step"].format(i=i + 1, obj=name(s["object"]), zone=name(s["zone"]), n=a["n"])
        err = a.get("error") or {}
        why = t["reason"].get(a["check"].get("reason") or err.get("reason") or err.get("type"), "")
        verdict = (t["pass"], (150, 230, 160)) if a["check"].get("satisfied") else \
            (t["fail"] + (t["why"].format(why=why) if why else ""), (255, 170, 150))
        events.append((first, last, nxt, [head, how], [head, verdict]))
    pos = 0
    for first, last, nxt, lines, check in events:
        for f in frames[pos:first]:
            put(caption(f, [], t["header"]))
        for f in frames[first:last + 1]:
            put(caption(f, lines, t["header"]))
        tail = frames[last + 1:nxt] or [frames[min(last, len(frames) - 1)]]
        for f in tail:
            put(caption(f, check, t["header"]))
        for _ in range(FPS * 2):  # held so the verdict can be read
            put(caption(tail[-1], check, t["header"]))
        pos = nxt
    for f in frames[pos:]:
        put(caption(f, [], t["header"]))

    # Closing card: the camera's verdicts, then the offline ground truth (never shown to the robot)
    meta, gt = e["task_meta"], (e.get("private_eval") or {}).get("objects", {})
    ok = lambda o: gt.get(o, {}).get("inside") and gt.get(o, {}).get("resting") and gt.get(o, {}).get("upright")
    end = Image.fromarray((np.asarray(frames[-1]) * .35).astype(np.uint8))
    y = draw_text(end, [t["result"]], (48, 60), size=34)
    placements = {}  # a placement retried after a replan counts once, confirmed if any of its tries passed
    for s in entries:
        placements[(s["object"], s["zone"])] = placements.get((s["object"], s["zone"])) or bool(s["success"])
    y = draw_text(end, [t["camera"].format(v=sum(placements.values()), n=len(placements))], (48, y + 16), size=26)
    y = draw_text(end, [(t["gt"], (200, 210, 225))], (48, y + 30), size=24)
    rows = [(t["row"].format(cat=t["names"].get(meta["category"][o], meta["category"][o]),
                             state=t["placed"] if ok(o) else t["not_placed"]),
             (150, 230, 160) if ok(o) else (255, 170, 150)) for o in meta["out_of_place"]]
    y = draw_text(end, rows, (48, y + 8), size=26)
    draw_text(end, [t["kept"].format(k=sum(bool(ok(o)) for o in meta["already_tidy"]), m=len(meta["already_tidy"]))],
              (48, y + 8), size=26)
    for _ in range(FPS * 6):
        put(np.asarray(end))
    writer.release()
    print(out, len(frames), "recorded frames,", len(events), "attempts captioned")


if __name__ == "__main__":
    main(*sys.argv[1:4])

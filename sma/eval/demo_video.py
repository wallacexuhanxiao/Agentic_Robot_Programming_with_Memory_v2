"""Concatenate episode videos with title cards (task, condition, ground-truth result).

Usage: python -m sma.eval.demo_video <run_dir> <out.mp4> <condition>/<task> [<condition>/<task> ...]
"""
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def card(text_lines, size=(360, 480), seconds=3.0, fps=20):
    frame = np.full((size[0], size[1], 3), 25, np.uint8)
    for i, line in enumerate(text_lines):
        cv2.putText(frame, line, (8, 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, .42, (235, 235, 235), 1, cv2.LINE_AA)
    return [frame] * int(seconds * fps)


def legend_card(size, seconds=8.0, fps=20):
    """Opening card that explains the scene and the on-screen annotations."""
    h, w = size
    frame = np.full((h, w, 3), 25, np.uint8)
    rows = [
        ("LEGEND", (120, 220, 235)),
        ("Containers (the simulated user's long-term preference):", (235, 235, 235)),
        ("  purple-floored toy box with grey walls  ->  toy blocks", (205, 140, 255)),
        ("  cyan tray  ->  mugs          orange tray  ->  pocket notebooks (books)", (120, 220, 235)),
        ("Objects are named <colour>_<kind>, e.g. red_block, yellow_mug, blue_book.", (235, 235, 235)),
        ("Left: robot (MuJoCo/Robosuite). Top bar = task id | memory condition.", (235, 235, 235)),
        ("Right: what the agent is doing, in sync with the robot:", (235, 235, 235)),
        ("  User instruction, current subgoal, attempt number and code source", (235, 235, 235)),
        ("  (code written by the LLM now, or a learned skill reused from memory)", (90, 220, 110)),
        ("  Program being executed; the highlighted yellow line is running now", (255, 220, 60)),
        ("  > green line = robot API call in progress", (90, 220, 110)),
        ("  bottom = camera check after each attempt / final ground truth", (235, 235, 235)),
        ("Ground truth comes from the simulator and is never shown to the model.", (150, 150, 150)),
    ]
    for i, (text, color) in enumerate(rows):
        cv2.putText(frame, text, (24, 40 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, .5 if i else .7, color, 1, cv2.LINE_AA)
    return [frame] * int(seconds * fps)


def label(frame, text):
    frame = frame.copy()
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(frame, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


LABELS = {"none": "no memory", "skill": "skill memory", "user": "user memory", "both": "user + skill memory",
          "memory": "skill memory", "plumbing": "scripted program, no LLM"}


def main(run_dir, out, items, legend=False):
    frames = []
    for item in items:
        condition, task = item.split("/")
        episode = json.loads((Path(run_dir) / condition / task / "episode.json").read_text())
        video = Path(run_dir) / condition / task / "video.mp4"
        gt = episode["private_eval"]["success"]
        gens = sum(len(s["attempts"]) for s in episode.get("subgoals", []))
        skills = sorted({f for s in episode.get("subgoals", []) for a in s["attempts"] for f in a["used_skills"]})
        goals = [f"{s['object']}->{s['zone']}{' ok' if s['success'] else ' X'}"
                 for s in episode.get("subgoals", [])]
        lines = [f"{task}  [{LABELS.get(condition, condition)}]",
                 episode["instruction"][:45], episode["instruction"][45:90],
                 f"code generations: {gens}", f"skills called: {', '.join(skills) or 'none'}",
                 "plan: " + ", ".join(goals)[:46],
                 f"ground truth: {'SUCCESS' if gt else 'FAIL'}"]
        clip = imageio.mimread(video, memtest=False)
        if legend and not frames:
            frames += legend_card(clip[0].shape[:2])
        frames += card(lines, size=clip[0].shape[:2])  # cards match the clip size (encoders pad to x16)
        tag = f"{task} | {LABELS.get(condition, condition)} | GT {'ok' if gt else 'fail'}"
        frames += clip if legend else [label(f, tag) for f in clip]  # replay clips carry their own labels
    imageio.mimwrite(out, frames, fps=20)
    print(out, len(frames), "frames")


if __name__ == "__main__":
    args = sys.argv[1:]
    legend = "--legend" in args
    args = [a for a in args if a != "--legend"]
    main(args[0], args[1], args[2:], legend=legend)

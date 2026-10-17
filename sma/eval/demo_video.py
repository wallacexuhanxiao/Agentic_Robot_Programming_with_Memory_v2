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


def label(frame, text):
    frame = frame.copy()
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(frame, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


LABELS = {"none": "no memory", "skill": "skill memory", "user": "user memory", "both": "user + skill memory",
          "memory": "skill memory", "plumbing": "scripted program, no LLM"}


def main(run_dir, out, items):
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
        frames += card(lines, size=clip[0].shape[:2])  # cards match the clip size (encoders pad to x16)
        tag = f"{task} | {LABELS.get(condition, condition)} | GT {'ok' if gt else 'fail'}"
        frames += [label(f, tag) for f in clip]
    imageio.mimwrite(out, frames, fps=20)
    print(out, len(frames), "frames")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])

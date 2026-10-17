"""Open-vocabulary segmentation service: Grounding DINO (text -> boxes) + SAM2 (boxes -> masks).

Runs in the provider environment (torch + transformers) on the job's GPU. The simulator process sends a
camera image and text prompts and receives one mask per detection. Weights come from the shared cache:
  SMA_GDINO = .../models--IDEA-Research--grounding-dino-tiny/snapshots/<rev>
  SMA_SAM2  = .../models--facebook--sam2-hiera-large/snapshots/<rev>

POST /segment  {"image_png_b64": str, "prompts": [str], "box_threshold": float, "text_threshold": float}
-> {"detections": [{"label": str, "score": float, "box": [x0,y0,x1,y1], "mask_png_b64": str}], "seconds": float}
"""
import base64
import io
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
from PIL import Image
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GDINO_PROC = AutoProcessor.from_pretrained(os.environ["SMA_GDINO"])
GDINO = AutoModelForZeroShotObjectDetection.from_pretrained(os.environ["SMA_GDINO"]).to(DEVICE).eval()
SAM_PROC = Sam2Processor.from_pretrained(os.environ["SMA_SAM2"])
SAM = Sam2Model.from_pretrained(os.environ["SMA_SAM2"]).to(DEVICE).eval()


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])
    return inter / max(area(a) + area(b) - inter, 1e-6)


@torch.no_grad()
def segment(image, prompts, box_threshold=.3, text_threshold=.25):
    text = ". ".join(p.lower().strip(". ") for p in prompts) + "."
    inputs = GDINO_PROC(images=image, text=text, return_tensors="pt").to(DEVICE)
    outputs = GDINO(**inputs)
    try:
        res = GDINO_PROC.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[image.size[::-1]])[0]
    except TypeError:  # older argument name
        res = GDINO_PROC.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[image.size[::-1]])[0]
    labels = res.get("text_labels", res.get("labels"))
    dets = sorted(({"label": str(l), "score": float(s), "box": [float(v) for v in b.tolist()]}
                   for l, s, b in zip(labels, res["scores"], res["boxes"])), key=lambda d: -d["score"])
    kept = []
    for d in dets:  # per-label non-maximum suppression
        if all(_iou(d["box"], k["box"]) < .6 or d["label"] != k["label"] for k in kept):
            kept.append(d)
    if not kept:
        return []
    sam_in = SAM_PROC(images=image, input_boxes=[[k["box"] for k in kept]], return_tensors="pt").to(DEVICE)
    sam_out = SAM(**sam_in, multimask_output=False)
    masks = SAM_PROC.post_process_masks(sam_out.pred_masks.cpu(), sam_in["original_sizes"])[0]
    masks = masks[:, 0].numpy() > 0 if masks.ndim == 4 else masks.numpy() > 0
    for k, m in zip(kept, masks):
        buf = io.BytesIO()
        Image.fromarray((m * 255).astype(np.uint8)).save(buf, format="PNG")
        k["mask_png_b64"] = base64.b64encode(buf.getvalue()).decode()
    return kept


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        started = time.time()
        image = Image.open(io.BytesIO(base64.b64decode(body["image_png_b64"]))).convert("RGB")
        try:
            dets = segment(image, body["prompts"], body.get("box_threshold", .3), body.get("text_threshold", .25))
            reply = {"detections": dets, "seconds": round(time.time() - started, 3)}
            code = 200
        except Exception as exc:  # report, do not crash the service
            reply, code = {"error": f"{type(exc).__name__}: {exc}"}, 500
        data = json.dumps(reply).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    port = int(sys.argv[1])
    print(f"segmentation service on {port}, device {DEVICE}", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()

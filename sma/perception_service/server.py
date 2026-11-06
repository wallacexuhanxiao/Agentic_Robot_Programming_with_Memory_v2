"""Open-vocabulary segmentation service: Grounding DINO (text -> boxes) + SAM2 (boxes -> masks).

Runs in the provider environment (torch + transformers) on the job's GPU. The simulator process sends a
camera image and text prompts and receives one mask per detection. Weights come from the shared cache:
  SMA_GDINO = .../models--IDEA-Research--grounding-dino-tiny/snapshots/<rev>
  SMA_SAM2  = .../models--facebook--sam2-hiera-large/snapshots/<rev>

POST /segment  {"image_png_b64": str, "prompts": [str], "box_threshold": float, "text_threshold": float}
-> {"detections": [{"label": str, "score": float, "box": [x0,y0,x1,y1], "mask_png_b64": str}], "seconds": float}
POST /auto     {"image_png_b64": str, "points_per_side": int, "iou_threshold": float, "points": [[x, y]] (optional)}
-> {"masks": [{"score": float, "point": [x, y], "area": int, "mask_png_b64": str}], "seconds": float}
   Class-agnostic: SAM2 prompted with a grid of single points (no text, no categories), every mask with a predicted
   IoU above the threshold, near-duplicates removed. With "points": those prompts instead of the grid.
POST /classify {"labels": [str], "items": [{"image_png_b64": str, "box": [x0,y0,x1,y1] | null}]}
-> {"scores": [{label: float}], "seconds": float}
   Grounding DINO as a classifier over a vocabulary (needs SMA_GDINO): per image, every label's score for the thing in
   the box (see classify()).
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
GDINO_PROC = GDINO = None  # text-prompted detection only when its weights are given (/auto needs SAM2 alone)
if os.environ.get("SMA_GDINO"):
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


def _png(mask):
    buf = io.BytesIO()
    Image.fromarray((mask * 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def auto_masks(image, points_per_side=16, iou_threshold=.8, batch=64, points=None):
    """SAM2 'segment everything': one image encoding, then a grid of single positive points (or the given points) in
    batches; all three mask granularities per point that score above iou_threshold; duplicates (mask IoU > 0.9)
    dropped."""
    W, H = image.size
    enc = SAM_PROC(images=image, return_tensors="pt").to(DEVICE)
    embeddings = SAM.get_image_embeddings(enc["pixel_values"])
    points = [[float(x), float(y)] for x, y in points] if points else \
        [[(i + .5) * W / points_per_side, (j + .5) * H / points_per_side]
         for j in range(points_per_side) for i in range(points_per_side)]
    found = []
    for k in range(0, len(points), batch):
        chunk = points[k:k + batch]
        prompt = SAM_PROC(images=image, input_points=[[[q] for q in chunk]], input_labels=[[[1] for _ in chunk]],
                          return_tensors="pt").to(DEVICE)
        out = SAM(input_points=prompt["input_points"], input_labels=prompt["input_labels"],
                  image_embeddings=embeddings, multimask_output=True)
        masks = SAM_PROC.post_process_masks(out.pred_masks.cpu(), prompt["original_sizes"])[0]  # (N, 3, H, W)
        scores = out.iou_scores.cpu()[0]
        for n, q in enumerate(chunk):
            for m in range(scores.shape[1]):
                if float(scores[n, m]) >= iou_threshold:
                    mask = masks[n, m].numpy() > 0
                    if mask.sum() >= 20:
                        found.append((float(scores[n, m]), q, mask))
    found.sort(key=lambda f: -f[0])
    kept = []
    for score, q, mask in found:
        area = mask.sum()
        if all((mask & k[2]).sum() / (area + k[2].sum() - (mask & k[2]).sum()) <= .9 for k in kept):
            kept.append((score, q, mask))
    return [{"score": round(s, 3), "point": [round(q[0], 1), round(q[1], 1)], "area": int(m.sum()), "mask_png_b64": _png(m)}
            for s, q, m in kept]


@torch.no_grad()
def classify(image, labels, box=None):
    """All labels in one prompt, so that they compete. A label's score: the highest, over the predicted boxes (those
    overlapping `box` with IoU > 0.3, when given and any do), of the mean probability of the label's tokens."""
    text = " . ".join(labels) + " ."
    inputs = GDINO_PROC(images=image, text=text, return_tensors="pt").to(DEVICE)
    out = GDINO(**inputs)
    probs = out.logits.sigmoid()[0]  # (queries, text positions), aligned with the prompt's tokens
    offsets = GDINO_PROC.tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    keep = torch.ones(probs.shape[0], dtype=torch.bool, device=probs.device)
    if box is not None:
        W, H = image.size
        cx, cy, w, h = out.pred_boxes[0].unbind(-1)
        boxes = torch.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], -1).tolist()
        near = torch.tensor([_iou(b, box) > .3 for b in boxes], device=probs.device)
        if near.any():
            keep = near
    scores, pos = {}, 0
    for label in labels:
        start = text.index(label, pos)
        pos = end = start + len(label)
        tokens = [i for i, (a, b) in enumerate(offsets) if b > a and a >= start and b <= end]
        scores[label] = round(float(probs[keep][:, tokens].mean(-1).max()), 4)
    return scores


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
        try:
            route = self.path.rstrip("/")
            if route == "/classify":
                scores = [classify(Image.open(io.BytesIO(base64.b64decode(it["image_png_b64"]))).convert("RGB"),
                                   body["labels"], it.get("box")) for it in body["items"]]
                reply = {"scores": scores}
            else:
                image = Image.open(io.BytesIO(base64.b64decode(body["image_png_b64"]))).convert("RGB")
                if route == "/auto":
                    reply = {"masks": auto_masks(image, body.get("points_per_side", 16), body.get("iou_threshold", .8),
                                                 points=body.get("points"))}
                else:
                    reply = {"detections": segment(image, body["prompts"], body.get("box_threshold", .3),
                                                   body.get("text_threshold", .25))}
            reply["seconds"] = round(time.time() - started, 3)
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

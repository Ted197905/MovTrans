#!/usr/bin/env python3
"""
On-screen text: burned-in subtitles, title cards, place/time captions (signs, logos and UI are ignored).

  ocr.py --video proxy.mp4 --out screen.json --work DIR/ocr --ollama http://127.0.0.1:11434 --model VLMODEL

1. 1 fps frames (640 px wide) -> EasyOCR text *detection* only (fast, no recognition).
2. Boxes that sit at the same place in most frames are static overlays (logo, watermark): dropped.
3. Consecutive frames with the same boxes and the same pixels inside the main box form one interval.
4. One full-size frame per interval goes to the vision model, which reads the text, classifies it and
   translates it into Korean.

Writes screen.json: [{"start", "end", "type": "subtitle"|"caption", "text", "ko"}].
"""
import argparse
import base64
import json
import os
import re
import shutil
import subprocess

import numpy as np

from subs import extract_json, log, ollama_chat, progress

PROMPT = (
    "Read every piece of text visible in this video frame, exactly as written (keep the original language). "
    "Classify each item:\n"
    "- subtitle: a burned-in dialogue subtitle, usually centered at the bottom\n"
    "- caption: text the video maker overlaid on the picture for the viewer, not dialogue: title card, chapter "
    "title, place/date/time, narration or explanation text, a message shown as a graphic\n"
    "- sign: any text physically present in the scene (signage, notices, posters, papers, screens, products, "
    "clothing) even when it is readable and relevant\n"
    "- logo: broadcaster/studio logo, watermark, channel name, rating badge\n"
    "- ui: timers, counters, camera info\n"
    "Return only a JSON array: [{\"text\": \"...\", \"type\": \"subtitle|caption|sign|logo|ui\", "
    "\"ko\": \"natural Korean subtitle translation of the text (for subtitle and caption only, else empty)\"}]. "
    "If there is no text, return []. Do not refuse."
)


URLISH = re.compile(r"(https?://|www\.|\.(com|net|cc|tv|xyz|jp|kr|io)\b|地址|网址|@)", re.I)


def probe_dims(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,duration",
                        "-of", "json", video], capture_output=True, text=True)
    st = json.loads(r.stdout)["streams"][0]
    return int(st["width"]), int(st["height"])


def iou(a, b):
    x1, x2, y1, y2 = max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / ua if ua > 0 else 0.0


def main_box(boxes):
    return max(boxes, key=lambda b: (b[1] - b[0]) * (b[3] - b[2]))


def same_boxes(a, b):
    """Same main text box (detection splits a line into 1-2 boxes from frame to frame; compare the biggest)."""
    return iou(main_box(a), main_box(b)) > 0.5


def crop_sig(img, box):
    import cv2
    x1, x2, y1, y2 = [int(v) for v in box]
    c = img[max(0, y1):y2, max(0, x1):x2]
    if c.size == 0:
        return None
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (160, 32)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-intervals", type=int, default=400)
    ap.add_argument("--min-height", type=float, default=0.025, help="text height as a fraction of the frame height")
    a = ap.parse_args()
    import cv2
    import torch

    os.makedirs(a.work, exist_ok=True)
    frames_dir = os.path.join(a.work, "frames")
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir)
    W, H = probe_dims(a.video)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", a.video, "-vf", "fps=1,scale=640:-2", "-q:v", "5",
                    os.path.join(frames_dir, "%06d.jpg")], check=True)
    files = sorted(os.listdir(frames_dir))
    n = len(files)
    log("%d frames" % n)
    progress(5)

    import easyocr
    reader = easyocr.Reader(["ja", "en"], gpu=torch.cuda.is_available(), recognizer=False,
                            model_storage_directory=os.path.join(os.environ.get("HOME", "/tmp"), ".cache", "easyocr"), verbose=False)
    per_frame = []  # list of (boxes, sig)
    img0 = cv2.imread(os.path.join(frames_dir, files[0]))
    h, w = img0.shape[:2]
    min_h = a.min_height * h
    for i, f in enumerate(files):
        img = cv2.imread(os.path.join(frames_dir, f))
        boxes = []
        try:
            hl, fl = reader.detect(img, min_size=10, text_threshold=0.6, low_text=0.4, link_threshold=0.4, mag_ratio=1.0)
            for x1, x2, y1, y2 in hl[0]:
                if y2 - y1 >= min_h and x2 - x1 >= 2 * (y2 - y1) * 0.5:
                    boxes.append([float(x1), float(x2), float(y1), float(y2)])
        except Exception as e:
            log("detect failed on %s: %s" % (f, e))
        per_frame.append(boxes)
        if i % 50 == 0:
            progress(5 + 45 * i / n)
    del reader; torch.cuda.empty_cache()

    # static overlays (logo, watermark, rating badge): a box position that recurs in > 15 % of the frames
    clusters = []  # [box, count]
    for boxes in per_frame:
        for b in boxes:
            for c in clusters:
                if iou(c[0], b) > 0.4:
                    c[1] += 1
                    break
            else:
                clusters.append([b, 1])
    static = [c[0] for c in clusters if c[1] > 0.15 * n]
    if static:
        log("static overlays ignored: %d (%s)" % (len(static), ", ".join("%d,%d-%d,%d x%d" % (c[0][0], c[0][2], c[0][1], c[0][3], c[1]) for c in clusters if c[1] > 0.15 * n)))
    for boxes in per_frame:
        boxes[:] = [b for b in boxes if not any(iou(b, s) > 0.4 for s in static)]

    # intervals of unchanged text
    intervals, cur, prev_sig = [], None, None
    for i, boxes in enumerate(per_frame):
        if not boxes:
            if cur:
                intervals.append(cur); cur = None
            prev_sig = None
            continue
        mb = main_box(boxes)
        sig = crop_sig(cv2.imread(os.path.join(frames_dir, files[i])), mb)
        changed = True
        if cur and same_boxes(cur["boxes"], boxes):
            small = (mb[1] - mb[0]) * (mb[3] - mb[2]) < 0.015 * w * h  # a logo-sized box: same place = same text
            if small or (sig is not None and prev_sig is not None and sig.shape == prev_sig.shape and float(np.mean(np.abs(sig - prev_sig))) <= 30.0):
                changed = False
        if cur and not changed:
            cur["end"] = i + 1
        else:
            if cur:
                intervals.append(cur)
            cur = {"start": i, "end": i + 1, "boxes": boxes, "area": (mb[1] - mb[0]) * (mb[3] - mb[2])}
        prev_sig = sig
    if cur:
        intervals.append(cur)
    log("%d text interval(s): %s" % (len(intervals), " ".join("%d-%d(%d,%d)" % (iv["start"], iv["end"], iv["boxes"][0][0], iv["boxes"][0][2]) for iv in intervals[:40])))
    if len(intervals) > a.max_intervals:
        intervals = sorted(intervals, key=lambda x: -(x["end"] - x["start"]) * x["area"])[:a.max_intervals]
        intervals.sort(key=lambda x: x["start"])
    progress(55)

    out = []
    logo_boxes = []  # positions the model already classified as logo/sign/ui: skip identical ones
    for k, iv in enumerate(intervals):
        if iv["boxes"] and all(any(iou(b, lb) > 0.5 for lb in logo_boxes) for b in iv["boxes"]):
            continue
        t = (iv["start"] + iv["end"]) / 2.0
        frame = os.path.join(a.work, "read.jpg")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "%.2f" % t, "-i", a.video, "-frames:v", "1",
                        "-vf", "scale='min(1280,iw)':-2", "-q:v", "3", frame], check=False)
        if not os.path.isfile(frame):
            continue
        try:
            reply = ollama_chat(a.ollama, a.model, [{"role": "user", "content": PROMPT}],
                                images=[base64.b64encode(open(frame, "rb").read()).decode("ascii")])
            items = extract_json(reply)
        except Exception as e:
            log("read failed at %ds: %s" % (t, str(e)[:120]))
            items = []
        kept = [it for it in items if isinstance(it, dict) and it.get("type") in ("subtitle", "caption") and (it.get("text") or "").strip()
                and not URLISH.search(it["text"])]
        if items and not kept:
            logo_boxes += [b for b in iv["boxes"] if not any(iou(b, lb) > 0.5 for lb in logo_boxes)]
        for it in kept:
            out.append({"start": float(iv["start"]), "end": float(iv["end"]), "type": it["type"],
                        "text": it["text"].strip(), "ko": (it.get("ko") or "").strip()})
            log("%s %d-%ds: %s -> %s" % (it["type"], iv["start"], iv["end"], it["text"][:40], (it.get("ko") or "")[:40]))
        progress(55 + 44 * (k + 1) / max(1, len(intervals)))
    shutil.rmtree(frames_dir, ignore_errors=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log("%d on-screen text cue(s)" % len(out))
    progress(100)


if __name__ == "__main__":
    main()

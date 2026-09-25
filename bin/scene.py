#!/usr/bin/env python3
"""
Scene analysis: one frame per subtitle segment (evenly sampled down to --max-frames),
described by a vision model through Ollama.

  scene.py --video src.mp4 --segments segments.json --out scenes.json --frames-dir DIR/frames \
           --ollama http://127.0.0.1:11434 --model MODEL --max-frames 300

Writes scenes.json: {"summary": "...", "scenes": [{"idx": segment index, "t": seconds, "desc": "..."}]}
"""
import argparse
import base64
import json
import os
import subprocess

from subs import log, ollama_chat, progress, strip_think

PROMPT = (
    "You are annotating a video for a professional subtitle translator. Look at this frame, taken while the "
    "following line is spoken.\n\nLine ({lang}): \"{line}\"\n\n"
    "Previous scene note: {prev}\n\n"
    "In 1-2 short sentences, describe what matters for translating the dialogue naturally: setting, who is "
    "speaking to whom, their relationship and relative status (affects Korean honorifics), mood, and any "
    "on-screen action or text. Be factual and explicit; do not refuse or moralize. Answer in Korean."
)


def grab_frame(video, t, out):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "%.3f" % t, "-i", video, "-frames:v", "1",
                    "-vf", "scale=448:-2", "-q:v", "4", out], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-frames", type=int, default=300)
    a = ap.parse_args()

    with open(a.segments, encoding="utf-8") as f:
        data = json.load(f)
    segs = data["segments"]
    lang = data.get("lang", "")
    if not segs:
        json.dump({"summary": "", "scenes": []}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
        progress(100)
        return

    n = len(segs)
    step = max(1, (n + a.max_frames - 1) // a.max_frames)
    picks = list(range(0, n, step))
    os.makedirs(a.frames_dir, exist_ok=True)
    log("model %s, %d frames of %d segments" % (a.model, len(picks), n))

    scenes = []
    prev = "(none)"
    for k, i in enumerate(picks):
        s = segs[i]
        t = (s["start"] + s["end"]) / 2
        frame = os.path.join(a.frames_dir, "%05d.jpg" % i)
        try:
            grab_frame(a.video, t, frame)
            with open(frame, "rb") as f:
                img = base64.b64encode(f.read()).decode("ascii")
        except Exception as e:
            log("frame %d failed: %s" % (i, e))
            progress(2 + 93 * (k + 1) / len(picks))
            continue
        try:
            reply = ollama_chat(a.ollama, a.model,
                                [{"role": "user", "content": PROMPT.format(lang=lang, line=s["text"], prev=prev)}],
                                images=[img])
            desc = strip_think(reply).replace("\n", " ").strip()
        except Exception as e:
            log("ollama failed at segment %d: %s" % (i, e))
            desc = ""
        if desc:
            scenes.append({"idx": i, "t": round(t, 3), "desc": desc})
            prev = desc
        progress(2 + 93 * (k + 1) / len(picks))

    summary = ""
    if scenes:
        notes = "\n".join("- " + sc["desc"] for sc in scenes[:: max(1, len(scenes) // 40)])
        try:
            reply = ollama_chat(a.ollama, a.model, [{"role": "user", "content":
                "These are scene notes from one video, in order:\n" + notes +
                "\n\nWrite a 3-5 sentence overview in Korean: genre, setting, main characters and how they relate, "
                "overall tone, and the register a Korean subtitle should use (formal/informal, honorifics). "
                "Be explicit and factual; do not refuse."}])
            summary = strip_think(reply)
        except Exception as e:
            log("summary failed: %s" % e)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "scenes": scenes}, f, ensure_ascii=False, indent=1)
    progress(100)


if __name__ == "__main__":
    main()

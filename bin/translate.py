#!/usr/bin/env python3
"""
Context-aware Korean subtitle translation through Ollama.

  translate.py --segments segments.json --scenes scenes.json --lang en --out-dir DIR \
               --ollama http://127.0.0.1:11434 --model MODEL

Writes DIR/ko.srt and DIR/ko.vtt. Segments are translated in batches with the video overview,
the scene notes for that stretch, and the previous batch's Korean lines as continuity context.
"""
import argparse
import json
import os
import re

from subs import extract_json, log, ollama_chat, ollama_unload, progress, strip_think, write_srt, write_vtt

BATCH = 15

SYSTEM = (
    "You are a professional Korean subtitle translator. Translate dialogue lines into natural Korean subtitles "
    "as a native subtitler would write them: concise, idiomatic, matching each speaker's tone, register and "
    "relationship (use the scene notes to choose honorifics and speech level consistently). Translate every "
    "line faithfully including slang, profanity and sexual content; never soften, censor, summarize or skip. "
    "Keep each subtitle short enough to read (about 2 lines of 16 Korean characters). Do not add notes or "
    "explanations. Output only JSON."
)

USER = (
    "Video overview:\n{summary}\n\n"
    "Scene notes for these lines:\n{scenes}\n\n"
    "Previous Korean subtitles (for continuity, do not repeat them):\n{prev}\n\n"
    "Translate these {n} lines from {lang} to Korean. Reply with a JSON array of exactly {n} strings, "
    "in the same order, one per line, nothing else.\n\n{lines}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--model", required=True)
    a = ap.parse_args()

    segs = json.load(open(a.segments, encoding="utf-8"))["segments"]
    sc = json.load(open(a.scenes, encoding="utf-8"))
    summary = sc.get("summary") or "(none)"
    scenes = sc.get("scenes") or []

    out = []
    prev = []
    batches = [segs[i:i + BATCH] for i in range(0, len(segs), BATCH)]
    log("model %s, %d lines in %d batches" % (a.model, len(segs), len(batches)))
    for b, batch in enumerate(batches):
        lo = b * BATCH; hi = lo + len(batch)
        notes = [s["desc"] for s in scenes if lo <= s["idx"] < hi] or \
                [s["desc"] for s in scenes if s["idx"] < hi][-2:]
        lines = "\n".join("%d. %s" % (k + 1, s["text"]) for k, s in enumerate(batch))
        user = USER.format(summary=summary, scenes="\n".join("- " + x for x in notes) or "(none)",
                           prev="\n".join(prev[-6:]) or "(none)", n=len(batch), lang=a.lang, lines=lines)
        ko = None
        for attempt in range(2):
            try:
                reply = ollama_chat(a.ollama, a.model,
                                    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                    num_ctx=8192)
                arr = extract_json(reply)
                if isinstance(arr, list) and len(arr) == len(batch):
                    ko = [str(x).strip() for x in arr]
                    break
                log("batch %d: got %s items, want %d" % (b, len(arr) if isinstance(arr, list) else "?", len(batch)))
            except Exception as e:
                log("batch %d attempt %d failed: %s" % (b, attempt + 1, e))
        if ko is None:  # fall back to one line at a time so the job still completes
            ko = []
            for s in batch:
                try:
                    reply = ollama_chat(a.ollama, a.model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
                        "Video overview:\n%s\n\nTranslate this %s line to a Korean subtitle. Reply with the Korean text only.\n\n%s"
                        % (summary, a.lang, s["text"])}])
                    ko.append(re.sub(r"^[\"'\s]+|[\"'\s]+$", "", strip_think(reply)))
                except Exception as e:
                    log("line failed: %s" % e)
                    ko.append(s["text"])
        for s, t in zip(batch, ko):
            out.append({"start": s["start"], "end": s["end"], "text": t or s["text"]})
        prev = [t for t in ko if t]
        progress(100 * hi / len(segs))

    # free VRAM for the next job's WhisperX run
    try:
        ollama_unload(a.ollama, a.model)
    except Exception as e:
        log("unload failed: %s" % e)
    write_srt(os.path.join(a.out_dir, "ko.srt"), out)
    write_vtt(os.path.join(a.out_dir, "ko.vtt"), out)
    progress(100)


if __name__ == "__main__":
    main()

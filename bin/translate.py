#!/usr/bin/env python3
"""
Context-aware Korean subtitle translation through Ollama.

  translate.py --segments segments.json --scenes scenes.json --lang en --out-dir DIR \
               --ollama http://127.0.0.1:11434 --model MODEL --rating rated|unrated

Writes DIR/ko.srt and DIR/ko.vtt. Segments are translated in batches with the video overview,
the scene notes for that stretch, and the previous batch's Korean lines as continuity context.
"""
import argparse
import json
import os
import re
import time

from subs import extract_json, log, ollama_chat, ollama_unload, progress, strip_think, write_srt, write_vtt

BATCH = 15

SYSTEM = (
    "You are a professional Korean subtitle translator. Translate dialogue lines into natural Korean subtitles "
    "as a native subtitler would write them: concise, idiomatic, matching each speaker's tone, register and "
    "relationship. Each source line starts with a speaker tag in brackets: F = female voice, M = male voice, "
    "the number tells speakers of the same sex apart, ? = unknown. Use the tags, the cast list and the scene "
    "notes to give each speaker a consistent voice and speech level (banmal/jondaetmal, how they address each "
    "other); different speakers must not sound like one narrator. Never put the tags in the output. {rating} "
    "Never summarize or skip a line. Keep each subtitle short enough to read (about 2 lines of 16 Korean characters). Do not add notes or "
    "explanations. Output only JSON."
)

RATING = {
    # theatrical release: meaning and heat kept, wording at Korean cinema-subtitle level
    "rated": "Content level: RATED, like a US theatrical release. Keep the meaning, insults and innuendo, but "
             "phrase profanity and sexual content the way Korean theatrical release subtitles do: natural and "
             "non-graphic, without explicit slang for sexual acts or body parts.",
    # adult video: nothing toned down
    "unrated": "Content level: UNRATED, like an adult video. Translate slang, profanity and explicit sexual content "
               "literally with direct, explicit Korean wording; never soften, censor or euphemize.",
}

USER = (
    "Video overview:\n{summary}\n\n"
    "Cast (speaker tag: number of lines):\n{cast}\n\n"
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
    ap.add_argument("--rating", choices=sorted(RATING), default="rated")
    a = ap.parse_args()
    system = SYSTEM.format(rating=RATING[a.rating])

    data = json.load(open(a.segments, encoding="utf-8"))
    segs = data["segments"]
    cast = "\n".join("- [%s]: %d lines" % (k, v.get("lines", 0)) for k, v in (data.get("cast") or {}).items()) or "(unknown)"
    tag = lambda s: ("[%s] " % s["spk"]) if s.get("spk") else ""
    sc = json.load(open(a.scenes, encoding="utf-8"))
    summary = sc.get("summary") or "(none)"
    scenes = sc.get("scenes") or []

    out = []
    prev = []
    batches = [segs[i:i + BATCH] for i in range(0, len(segs), BATCH)]
    log("model %s, rating %s, %d lines in %d batches" % (a.model, a.rating, len(segs), len(batches)))
    for b, batch in enumerate(batches):
        lo = b * BATCH; hi = lo + len(batch)
        notes = [s["desc"] for s in scenes if lo <= s["idx"] < hi] or \
                [s["desc"] for s in scenes if s["idx"] < hi][-2:]
        lines = "\n".join("%d. %s%s" % (k + 1, tag(s), s["text"]) for k, s in enumerate(batch))
        user = USER.format(summary=summary, cast=cast, scenes="\n".join("- " + x for x in notes) or "(none)",
                           prev="\n".join(prev[-6:]) or "(none)", n=len(batch), lang=a.lang, lines=lines)
        ko = None
        for attempt in range(3):
            if attempt:
                time.sleep(5)  # an Ollama 500 usually means the runner crashed and is reloading
            try:
                reply = ollama_chat(a.ollama, a.model,
                                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
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
                    reply = ollama_chat(a.ollama, a.model, [{"role": "system", "content": system}, {"role": "user", "content":
                        "Video overview:\n%s\n\nTranslate this %s line to a Korean subtitle. Reply with the Korean text only.\n\n%s%s"
                        % (summary, a.lang, tag(s), s["text"])}])
                    ko.append(re.sub(r"^[\"'\s]+|[\"'\s]+$", "", strip_think(reply)))
                except Exception as e:
                    log("line failed: %s" % e)
                    ko.append(s["text"])
        for s, t in zip(batch, ko):
            out.append({"start": s["start"], "end": s["end"], "text": t or s["text"]})
        prev = [tag(s) + t for s, t in zip(batch, ko) if t]
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

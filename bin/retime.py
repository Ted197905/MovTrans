#!/usr/bin/env python3
"""
Re-time the cues of an already processed video without re-running the pipeline.

  retime.py --dir writable/media/ID [--lang ja] [--verify] [--no-align]

Reads DIR/audio.wav and DIR/segments.json, drops lines Whisper made up on silence, force-aligns each cue to the
audio (timing.align_cues; apply once - a second run on already aligned cues drifts), rewrites segments.json,
orig.srt/vtt, and moves the matching cues of ko.srt/vtt and ko.sdh.srt/vtt (matched by their old start/end;
screen-text and sound cues keep their times).
--verify: hallucination check (verify.py) - each cue is decoded again by large-v3 and kotoba-whisper on its own
span and dropped unless a VAD hears speech there or one of them writes the same words.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

from subs import log, write_srt, write_vtt
from timing import SR, align_cues, finish_cues, peak_db
from transcribe import is_hallucination
from verify import decode_spans, keep as agree, similarity, vad_max, vad_probs


def read_cues(path):
    """SRT or VTT -> [{start, end, text, setting}] (setting = VTT cue settings such as 'line:8% align:center')."""
    vtt = path.endswith(".vtt")
    cues, blk = [], []
    for line in open(path, encoding="utf-8").read().split("\n") + [""]:
        if line.strip():
            blk.append(line)
            continue
        for k, l in enumerate(blk):
            m = re.match(r"(\d\d):(\d\d):(\d\d)[.,](\d{3}) --> (\d\d):(\d\d):(\d\d)[.,](\d{3})(.*)", l)
            if m:
                v = [int(x) for x in m.groups()[:8]]
                cues.append({"start": v[0] * 3600 + v[1] * 60 + v[2] + v[3] / 1000, "end": v[4] * 3600 + v[5] * 60 + v[6] + v[7] / 1000,
                             "text": "\n".join(blk[k + 1:]), "setting": m.group(9).strip()})
                break
        blk = []
    return cues


def remap(path, old_new, dropped):
    if not os.path.isfile(path):
        return
    out = []
    for c in read_cues(path):
        key = (round(c["start"], 3), round(c["end"], 3))
        if key in dropped:
            continue
        if key in old_new:
            c["start"], c["end"] = old_new[key]
        if c["setting"].startswith("line:8%"):
            c["pos"] = "top"
        out.append(c)
    out.sort(key=lambda c: c["start"])
    (write_vtt if path.endswith(".vtt") else write_srt)(path, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--lang", default="")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--models", default="large-v3,kotoba-tech/kotoba-whisper-v2.0-faster")
    a = ap.parse_args()

    import torch
    from faster_whisper import decode_audio
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = json.load(open(os.path.join(a.dir, "segments.json"), encoding="utf-8"))
    lang = a.lang or data.get("lang") or "ja"
    audio = decode_audio(os.path.join(a.dir, "audio.wav"), sampling_rate=SR).astype(np.float32)
    cues, dropped = [], set()
    for s in data["segments"]:
        if is_hallucination(s["text"]) or peak_db(audio, s["start"], s["end"]) < -55:
            dropped.add((round(s["start"], 3), round(s["end"], 3)))
            log("drop %.1f-%.1f: %s" % (s["start"], s["end"], s["text"][:40]))
            continue
        cues.append(dict(s, old=(round(s["start"], 3), round(s["end"], 3))))
    if a.verify:
        from faster_whisper import WhisperModel
        probs = vad_probs(audio)
        spans = [(c["start"], c["end"]) for c in cues]
        others = []
        for name in a.models.split(","):
            m = WhisperModel(name, device=device, compute_type="float16" if device == "cuda" else "int8")
            others.append(decode_spans(m, audio, spans, lang))
            del m
        kept = []
        for i, c in enumerate(cues):
            v = vad_max(probs, c["start"], c["end"])
            best = max((o[i] for o in others), key=lambda t: similarity(c["text"], t))
            if agree(c["text"], best, v):
                kept.append(c)
            else:
                dropped.add(c["old"])
                log("unconfirmed %.1f-%.1f (vad %.2f): %s | %s" % (c["start"], c["end"], v, c["text"][:30], " / ".join(o[i][:20] for o in others)))
        log("verify: %d of %d cues confirmed" % (len(kept), len(cues)))
        cues = kept
    moved = align_cues(audio, cues, lang, device) if not a.no_align else 0
    for c in (finish_cues(cues) if not a.no_align else finish_cues(cues, lead=0, tail=0)):
        dropped.add(c["old"])
        log("drop duplicate %.1f-%.1f: %s" % (c["old"][0], c["old"][1], c["text"][:40]))
    old_new = {c["old"]: (c["start"], c["end"]) for c in cues}
    shifts = [c["start"] - c["old"][0] for c in cues]
    log("%d cues, %d dropped, %d aligned; start moved by median %.2fs, p90 %.2fs, max %.2fs" % (
        len(cues), len(dropped), moved, float(np.median(shifts)), float(np.percentile(shifts, 90)), max(shifts)))
    for c in cues:
        del c["old"]
    data["segments"] = cues
    with open(os.path.join(a.dir, "segments.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    write_srt(os.path.join(a.dir, "orig.srt"), cues)
    write_vtt(os.path.join(a.dir, "orig.vtt"), cues)
    for name in ("ko.srt", "ko.vtt", "ko.sdh.srt", "ko.sdh.vtt"):
        remap(os.path.join(a.dir, name), old_new, dropped)


if __name__ == "__main__":
    main()

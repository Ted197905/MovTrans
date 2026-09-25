#!/usr/bin/env python3
"""
WhisperX transcription with word-level alignment.

  transcribe.py --audio audio.wav --lang en --model large-v3 --compute float16 --out-dir DIR

Writes DIR/segments.json ([{start,end,text}]), DIR/orig.srt, DIR/orig.vtt.
Progress goes to stderr as "progress <0..100>".
"""
import argparse
import gc
import json
import os

from subs import log, progress, write_srt, write_vtt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--compute", default="float16")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = a.compute if device == "cuda" else "int8"
    log("device %s, model %s (%s)" % (device, a.model, compute))
    progress(2)

    audio = whisperx.load_audio(a.audio)
    model = whisperx.load_model(a.model, device, compute_type=compute, language=a.lang)
    progress(10)
    result = model.transcribe(audio, batch_size=a.batch, language=a.lang)
    log("transcribed %d segments" % len(result["segments"]))
    progress(60)
    del model; gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    try:
        align_model, meta = whisperx.load_align_model(language_code=a.lang, device=device)
        result = whisperx.align(result["segments"], align_model, meta, audio, device, return_char_alignments=False)
        del align_model; gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:  # no alignment model for this language: keep whisper timestamps
        log("alignment skipped: %s" % e)
    progress(90)

    segs = []
    for s in result["segments"]:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        segs.append({"start": round(float(s["start"]), 3), "end": round(float(s["end"]), 3), "text": text})

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "segments.json"), "w", encoding="utf-8") as f:
        json.dump({"lang": a.lang, "segments": segs}, f, ensure_ascii=False, indent=1)
    write_srt(os.path.join(a.out_dir, "orig.srt"), segs)
    write_vtt(os.path.join(a.out_dir, "orig.vtt"), segs)
    progress(100)


if __name__ == "__main__":
    main()

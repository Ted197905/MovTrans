#!/usr/bin/env python3
"""
Coverage audit: which speech-like regions (silero VAD, lenient) have no subtitle cue?

  audit.py --audio audio.wav --segments segments.json [--out audit.json] [--min 1.5]

Prints a summary and the uncovered regions (start, end, seconds). Used to find where dialogue is lost.
"""
import argparse
import json

import numpy as np

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--out")
    ap.add_argument("--min", type=float, default=1.5)
    ap.add_argument("--threshold", type=float, default=0.3)
    a = ap.parse_args()
    import torch  # noqa: F401  (loads CUDA libs before ctranslate2/onnx)
    from faster_whisper import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = decode_audio(a.audio, sampling_rate=SR).astype(np.float32)
    total = len(audio) / SR
    speech = [(t["start"] / SR, t["end"] / SR) for t in
              get_speech_timestamps(audio, VadOptions(threshold=a.threshold, min_silence_duration_ms=500, speech_pad_ms=100))]
    segs = json.load(open(a.segments, encoding="utf-8"))["segments"]
    cues = sorted((s["start"], s["end"]) for s in segs)

    def covered(s, e):
        for cs, ce in cues:
            if ce < s - 0.5:
                continue
            if cs > e + 0.5:
                break
            return True
        return False

    unc = []
    for s, e in speech:
        if e - s >= a.min and not covered(s, e):
            unc.append({"start": round(s, 2), "end": round(e, 2), "sec": round(e - s, 1)})
    speech_sec = sum(e - s for s, e in speech)
    unc_sec = sum(u["sec"] for u in unc)
    out = {"duration": round(total), "speech_sec": round(speech_sec), "cues": len(cues), "uncovered_regions": len(unc),
           "uncovered_sec": round(unc_sec), "regions": unc}
    print("duration %ds, VAD speech %ds, cues %d, uncovered regions %d (%ds, %.0f%% of speech)" % (
        total, speech_sec, len(cues), len(unc), unc_sec, 100 * unc_sec / max(1, speech_sec)))
    for u in unc:
        print("  %7.1f - %7.1f  %5.1fs" % (u["start"], u["end"], u["sec"]))
    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

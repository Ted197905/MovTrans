#!/usr/bin/env python3
"""
Speaker diarization with NVIDIA streaming Sortformer (runs in the NeMo venv: pyenv/nemo/bin/python).

  diarize_sortformer.py --audio audio.wav --out turns.json [--chunk 3600]

Writes [[start, end, "speaker_N"], ...]. Offline preset (high latency = best accuracy), up to 4 speakers per
1 h piece; longer files are diarized in overlapping pieces whose labels are matched on the overlap.
"""
import argparse
import json
import logging
import os
import sys


CHUNK = 3600.0    # a 4 h file's mel/STFT on the GPU fails under WSL ("Failed to create GPU mapping"): pieces of 1 h
OVERLAP = 120.0   # consecutive pieces share 2 min; the labels are matched on that stretch


def run(m, path, offset=0.0):
    segs = m.diarize(audio=[path], batch_size=1)
    turns = []
    for line in (segs[0] if segs else []):
        parts = str(line).split()
        if len(parts) >= 3:
            turns.append([float(parts[0]) + offset, float(parts[1]) + offset, parts[2]])
    turns.sort()
    return turns


def overlap_matrix(a_turns, b_turns, lo, hi):
    """Seconds each label of a co-occurs with each label of b inside [lo, hi]."""
    m = {}
    for s1, e1, l1 in a_turns:
        if e1 <= lo or s1 >= hi:
            continue
        for s2, e2, l2 in b_turns:
            ov = min(e1, e2, hi) - max(s1, s2, lo)
            if ov > 0:
                m[(l1, l2)] = m.get((l1, l2), 0.0) + ov
    return m


def pieces(m, path, info, dur, chunk):
    import tempfile
    starts = [0.0]
    while starts[-1] + chunk < dur:
        starts.append(starts[-1] + chunk - OVERLAP)
    chunks = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, cs in enumerate(starts):
            ce = min(dur, cs + chunk)
            piece = os.path.join(tmp, "piece%d.wav" % k)
            data, sr = sf.read(path, start=int(cs * info.samplerate), frames=int((ce - cs) * info.samplerate), dtype="int16")
            sf.write(piece, data, sr)
            t = run(m, piece, cs)
            sys.stderr.write("sortformer piece %d (%.0f-%.0f s): %d turns, %d speakers\n" % (k, cs, ce, len(t), len({x[2] for x in t})))
            chunks.append((cs, ce, t))
            os.unlink(piece)
    return stitch(chunks)


def stitch(chunks):
    """chunks: [(start, end, turns)] in order. Labels of each piece are renamed to the global speaker they
    co-occur with in the overlap with the previous piece; new voices get new global names."""
    out, next_id = [], 0
    for i, (cs, ce, turns) in enumerate(chunks):
        names = {}
        if i:
            lo, hi = cs, chunks[i - 1][1]
            mat = overlap_matrix(out, turns, lo, hi)
            for (g, l), _ in sorted(mat.items(), key=lambda kv: -kv[1]):
                if l not in names and g not in names.values():
                    names[l] = g
            mid = cs + (hi - lo) / 2
            out = [t for t in out if t[0] < mid]
        for s, e, l in turns:
            if l not in names:
                names[l] = "speaker_%d" % next_id
                next_id += 1
            if not i or s >= mid:
                out.append([s, e, names[l]])
    out.sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="nvidia/diar_streaming_sortformer_4spk-v2")
    ap.add_argument("--chunk", type=float, default=CHUNK, help="seconds per piece for long files (the caller retries with smaller pieces)")
    a = ap.parse_args()
    logging.disable(logging.WARNING)
    os.environ.setdefault("NEMO_LOGGING_LEVEL", "ERROR")
    global sf
    import soundfile as sf
    from nemo.collections.asr.models import SortformerEncLabelModel
    m = SortformerEncLabelModel.from_pretrained(a.model, map_location="cuda" if _cuda() else "cpu")
    m.eval()
    sm = m.sortformer_modules  # offline / high-latency preset from the model card
    sm.chunk_len = 340; sm.chunk_right_context = 40; sm.fifo_len = 40
    sm.spkcache_update_period = 300; sm.spkcache_len = 188
    info = sf.info(a.audio)
    dur = info.frames / info.samplerate
    turns = run(m, a.audio) if dur <= a.chunk + OVERLAP else pieces(m, a.audio, info, dur, a.chunk)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(turns, f)
    sys.stderr.write("sortformer: %d turns, %d speakers\n" % (len(turns), len({t[2] for t in turns})))


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()

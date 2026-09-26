#!/usr/bin/env python3
"""
Speaker diarization with NVIDIA streaming Sortformer (runs in the NeMo venv: pyenv/nemo/bin/python).

  diarize_sortformer.py --audio audio.wav --out turns.json

Writes [[start, end, "speaker_N"], ...]. Offline preset (high latency = best accuracy), up to 4 speakers.
"""
import argparse
import json
import logging
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="nvidia/diar_streaming_sortformer_4spk-v2")
    a = ap.parse_args()
    logging.disable(logging.WARNING)
    os.environ.setdefault("NEMO_LOGGING_LEVEL", "ERROR")
    from nemo.collections.asr.models import SortformerEncLabelModel
    m = SortformerEncLabelModel.from_pretrained(a.model, map_location="cuda" if _cuda() else "cpu")
    m.eval()
    sm = m.sortformer_modules  # offline / high-latency preset from the model card
    sm.chunk_len = 340; sm.chunk_right_context = 40; sm.fifo_len = 40
    sm.spkcache_update_period = 300; sm.spkcache_len = 188
    segs = m.diarize(audio=[a.audio], batch_size=1)
    turns = []
    for line in (segs[0] if segs else []):
        parts = str(line).split()
        if len(parts) >= 3:
            turns.append([float(parts[0]), float(parts[1]), parts[2]])
    turns.sort()
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

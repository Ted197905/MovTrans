"""Cue timing: Whisper's word timestamps are only rough (the first word of a segment absorbs the pause before it,
so a cue starts when the previous line ended and the next line "appears early"). The cue text is therefore
force-aligned to the audio with wav2vec2 (whisperx) and the cue starts/ends at the aligned characters."""
import re

import numpy as np

from subs import log

SR = 16000
LEAD = 0.05   # a cue may appear this long before the first sound of the line
TAIL = 0.20   # and stay this long after the last one
MIN_DUR = 0.85  # Netflix-style readable minimum (5/6 s)
GAP = 0.05    # the next cue never starts before the previous one is gone


def fix_word_spans(words, lang):
    """First/last word of a Whisper segment: a 1-3 char word that spans seconds is a placeholder timestamp
    (the segment's leading/trailing silence); shrink it to the pace of the rest of the segment."""
    if len(words) < 2:
        return words
    cjk = lang in ("ja", "zh", "th")

    def chars(w):
        return max(1, len(re.sub(r"[\s。、．，,.!！?？…]", "", w["word"])) * (3 if re.search(r"\d", w["word"]) else 1))

    inner = words[1:-1] if len(words) > 2 else words[1:]
    span = sum(w["end"] - w["start"] for w in inner)
    rate = span / max(1, sum(chars(w) for w in inner)) if span > 0 else (0.15 if cjk else 0.08)
    rate = min(max(rate, 0.08), 0.4)
    for w, side in ((words[0], "start"), (words[-1], "end")):
        est = min(max(rate * chars(w), 0.12), 1.2)
        if w["end"] - w["start"] > est + 0.4:
            if side == "start":
                w["start"] = w["end"] - est
            else:
                w["end"] = w["start"] + est
    return words


def peak_db(audio, start, end):
    """Loudest 100 ms of a span, in dBFS: text on a span below about -55 dB was made up (nothing audible)."""
    x = audio[max(0, int(start * SR)):int(end * SR)]
    n = len(x) // 1600 * 1600
    if n < 1600:
        return -100.0
    fr = x[:n].reshape(-1, 1600)
    return float(20 * np.log10(np.sqrt((fr ** 2).mean(1)).max() + 1e-9))


def load_aligner(lang, device):
    import whisperx
    return whisperx.load_align_model(language_code=lang, device=device)


STRETCH = 0.45  # a character "spoken" longer than this is the aligner idling on silence/noise next to the line


def span_from_chars(chars):
    """Onset/offset of a line from its aligned characters. On silence the wav2vec2 model keeps emitting the
    neighbouring character, so the first/last character of a line stretches out to the window edge; the real
    sound of a stretched character sits at its transition (end of a leading one, start of a trailing one).
    Returns (start, end, n) or None."""
    cs = [c for c in chars if isinstance(c.get("start"), (int, float)) and c["end"] - c["start"] >= 0.03]
    if not cs:
        return None
    # characters scattered over separate bursts (a moan matched here, the line there): keep the biggest cluster
    clusters, cur = [], [cs[0]]
    for prev, c in zip(cs, cs[1:]):
        if c["start"] - min(prev["end"], prev["start"] + STRETCH) > 1.0:
            clusters.append(cur); cur = []
        cur.append(c)
    clusters.append(cur)
    cs = max(clusters, key=len)
    lead = cs[0]
    start = lead["start"] if lead["end"] - lead["start"] <= STRETCH else lead["end"] - 0.2
    tail = cs[-1]
    end = tail["end"] if tail["end"] - tail["start"] <= STRETCH else tail["start"] + 0.2
    return max(start, cs[0]["start"]), max(end, start + 0.1), len(cs)  # n counts the cluster only


def align_cues(audio, cues, lang, device, aligner=None, pad_start=0.1, pad_end=0.35):
    """Force-align every cue's text within its own window (+pads); the cue then starts at its first aligned
    character and ends at its last. A cue whose alignment is implausible keeps its Whisper times.
    Returns the number of cues re-timed."""
    import whisperx
    model, meta = aligner or load_aligner(lang, device)
    total = len(audio) / SR
    moved = 0
    for c in cues:
        lo = max(0.0, c["start"] - pad_start)
        hi = min(total, c["end"] + pad_end)
        if hi - lo < 0.2:
            continue
        try:
            res = whisperx.align([{"start": lo, "end": hi, "text": c["text"]}], model, meta, audio, device,
                                 return_char_alignments=True)
        except Exception as e:
            log("align failed at %.1f: %s" % (c["start"], str(e).split("\n")[0][:120]))
            continue
        span = span_from_chars([ch for s in res["segments"] for ch in s.get("chars", [])])
        n = len(re.sub(r"\s", "", c["text"]))
        if not span:  # nothing recognisable in the window (moans, music): at least do not let the text linger
            c["end"] = round(min(c["end"], c["start"] + 1.0 + 0.25 * n), 3)
            continue
        # a line the model did not recognise gets its characters squeezed into a few frames; the position is still
        # right (measured against a VAD), the length is not: give it a speakable duration
        c["start"], c["end"] = round(span[0], 3), round(max(span[1], span[0] + 0.06 * n), 3)
        moved += 1
    return moved


def finish_cues(cues, lead=LEAD, tail=TAIL, min_dur=MIN_DUR):
    """Lead-in, tail, readable minimum, and no cue overlapping the next one. Two cues aligned to the same moment
    are the same line heard twice (main pass + fill pass): the shorter text goes. Returns the removed cues."""
    removed, out = [], []
    for c in sorted(cues, key=lambda c: c["start"]):
        if out and c["start"] < out[-1]["start"] + 0.3:
            ta, tb = re.sub(r"\W", "", out[-1]["text"]), re.sub(r"\W", "", c["text"])
            if ta and ta in tb:
                removed.append(out.pop())
            elif tb and tb in ta:
                removed.append(c)
                continue
        out.append(c)
    cues[:] = out
    for c in cues:
        c["start"] = max(0.0, c["start"] - lead)
        c["end"] = max(c["end"] + tail, c["start"] + min_dur)
    for a, b in zip(cues, cues[1:]):
        if a["end"] > b["start"] - GAP:
            a["end"] = max(a["start"] + 0.3, b["start"] - GAP)
        if b["start"] < a["end"] + GAP:  # both start at nearly the same moment: the second waits
            b["start"] = a["end"] + GAP
            b["end"] = max(b["end"], b["start"] + 0.5)
    for c in cues:
        c["start"], c["end"] = round(c["start"], 3), round(c["end"], 3)
    return removed

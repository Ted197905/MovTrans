#!/usr/bin/env python3
"""
Transcription with word timestamps, hallucination filtering, speaker labels and subtitle-sized cues.

  transcribe.py --audio audio.wav --lang ja --model large-v3 --compute float16 --out-dir DIR [--hf-token TOKEN]

Pipeline: faster-whisper (word timestamps, no VAD: silero merges moaning/music into huge chunks and dialogue is lost) -> drop hallucinated / non-speech segments
-> speaker labels (pyannote diarization when --hf-token is given, else per-cue pitch -> F/M)
-> regroup words into cues (max 6 s / ~28 CJK chars, split at pauses and speaker changes).

Writes DIR/segments.json {"lang", "cast": {tag: {"lines", "f0"}}, "segments": [{start,end,text,spk}]},
DIR/orig.srt, DIR/orig.vtt. Progress goes to stderr as "progress <0..100>".
"""
import argparse
import gc
import json
import os
import re

import numpy as np

from subs import log, progress, write_srt, write_vtt

SR = 16000
NO_SPACE_LANGS = {"ja", "zh", "th"}

# Whole-segment phrases Whisper emits on silence, music or moaning (credits/outro training data).
HALLUCINATIONS = [
    "ご視聴ありがとうございました", "ご視聴ありがとうございます", "チャンネル登録", "高評価", "最後までご視聴",
    "字幕", "おやすみなさい", "また次の動画で", "お疲れ様でした",
    "thank you for watching", "thanks for watching", "subtitles by", "subscribe", "like and subscribe",
    "please subscribe", "see you in the next video", "copyright", "amara.org",
    "시청해 주셔서 감사합니다", "구독", "좋아요", "谢谢观看", "感谢观看", "请订阅", "字幕由",
    "gracias por ver", "merci d'avoir regardé", "danke fürs zuschauen", "obrigado por assistir",
]


MOAN_JA = re.compile(r"[あぁいぃうぅえぇおぉんっーはひふへほ〜～]+")
MOAN_LATIN = re.compile(r"(?:[aeiou]+h*|h[aeiou]+|m+|hm+|uh+|ah+|oh+|mm+|mhm)")


def is_hallucination(text):
    t = re.sub(r"[\s。、．，,.!！?？…・「」\"'()（）\-]", "", text).lower()
    if not t or MOAN_JA.fullmatch(t) or MOAN_LATIN.fullmatch(t):  # moans / interjections are not dialogue
        return True
    if re.fullmatch(r"(.)\1{2,}", t) or re.fullmatch(r"(..)\1{2,}", t):  # ああああ / はぁはぁはぁ
        return True
    if len(set(t)) <= 2 and len(t) >= 6:
        return True
    for h in HALLUCINATIONS:
        h2 = re.sub(r"[\s']", "", h).lower()
        if t == h2 or (h2 in t and len(t) <= len(h2) + 4):
            return True
    return False


def yin_f0(x, fmin=70, fmax=400, thr=0.15):
    """YIN (difference function + cumulative mean normalisation) per 40 ms frame, 20 ms hop.
    Returns (times, f0) of the voiced frames; times are seconds from the start of x."""
    n = int(0.04 * SR); tmax = SR // fmin; tmin = SR // fmax; hop = int(0.02 * SR); L = n + tmax
    empty = (np.array([]), np.array([]))
    if len(x) < L:
        return empty
    starts = np.arange(0, len(x) - L + 1, hop)
    frames = np.stack([x[s:s + L] for s in starts]).astype(np.float64)
    loud = np.sqrt((frames[:, :n] ** 2).mean(1)) > 0.01
    frames, starts = frames[loud], starts[loud]
    if not len(frames):
        return empty
    nfft = 1 << (2 * L - 1).bit_length()
    A = np.fft.rfft(frames[:, :n], nfft, axis=1); B = np.fft.rfft(frames, nfft, axis=1)
    corr = np.fft.irfft(np.conj(A) * B, nfft, axis=1)[:, : tmax + 1]          # sum_j x[j] x[j+tau]
    cs = np.concatenate([np.zeros((len(frames), 1)), np.cumsum(frames ** 2, axis=1)], axis=1)
    taus = np.arange(tmax + 1)
    e_tau = cs[:, taus + n] - cs[:, taus]
    d = cs[:, n:n + 1] + e_tau - 2 * corr
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(np.cumsum(d[:, 1:], axis=1), 1e-9)
    times, f0s = [], []
    for st, row in zip(starts, cmnd):
        r = row[tmin:tmax + 1]
        if r.min() < thr:  # deepest dip, not the first one below threshold (avoids octave-up errors)
            times.append((st + n / 2) / SR); f0s.append(SR / (tmin + int(np.argmin(r))))
    return np.array(times), np.array(f0s)


def pitch_hz(audio, start, end):
    """Median F0 (Hz) of a stretch of audio, or None when too short / unvoiced."""
    a = int(max(0, start) * SR); b = int(min(len(audio) / SR, end) * SR)
    if b - a < SR // 4:
        return None
    _, f0 = yin_f0(audio[a:b])
    return float(np.median(f0)) if len(f0) >= 5 else None


def gender(f0):
    return "?" if f0 is None else ("F" if f0 >= 165 else "M")


def label_words_by_pitch(audio, words, spans):
    """No diarization: F/M per word from voiced frames (F >= 180 Hz, M <= 150 Hz, in between ignored);
    words without a clear vote inherit their neighbour's label."""
    frames = []  # (time, class)
    for s, e in spans:
        a = int(max(0, s - 0.2) * SR); b = int(min(len(audio) / SR, e + 0.2) * SR)
        t, f0 = yin_f0(audio[a:b])
        for ti, fi in zip(t, f0):
            if fi >= 180 or fi <= 150:
                frames.append((a / SR + ti, "F" if fi >= 180 else "M"))
    frames.sort()
    ft = np.array([f[0] for f in frames]); fc = np.array([f[1] for f in frames])
    for w in words:
        lo = np.searchsorted(ft, w["start"] - 0.05); hi = np.searchsorted(ft, w["end"] + 0.05)
        votes = fc[lo:hi]
        if len(votes) >= 2:
            f = int((votes == "F").sum()); m = len(votes) - f
            w["spk"] = "F" if f > m else ("M" if m > f else "?")
        else:
            w["spk"] = "?"
    last, last_end = "?", 0.0
    for w in words:  # fill gaps from the previous word (short function words carry no voiced frames)
        if w["spk"] == "?" and last != "?" and w["start"] - last_end < 1.5:
            w["spk"] = last
        if w["spk"] != "?":
            last, last_end = w["spk"], w["end"]
    for i in range(len(words) - 2, -1, -1):  # then from the next word
        if words[i]["spk"] == "?" and words[i + 1]["spk"] != "?" and words[i + 1]["start"] - words[i]["end"] < 1.5:
            words[i]["spk"] = words[i + 1]["spk"]


def diarize(audio, token, model_name):
    """[(start, end, label)] from pyannote, or None when unavailable."""
    if not token:
        return None
    try:
        import torch
        from pyannote.audio import Pipeline
        pipe = Pipeline.from_pretrained(model_name, token=token)
        if pipe is None:
            raise RuntimeError("pipeline not available (accept the model terms on huggingface.co)")
        if torch.cuda.is_available():
            pipe.to(torch.device("cuda"))
        out = pipe({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SR})
        ann = getattr(out, "speaker_diarization", out)
        turns = [(float(seg.start), float(seg.end), str(lab)) for seg, _, lab in ann.itertracks(yield_label=True)]
        del pipe; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return turns
    except Exception as e:
        log("diarization skipped: %s" % str(e).split("\n")[0][:300])
        return None


def speaker_at(turns, start, end):
    best, best_ov = None, 0.0
    for s, e, lab in turns:
        ov = min(e, end) - max(s, start)
        if ov > best_ov:
            best, best_ov = lab, ov
    if best is None:  # word inside no turn: nearest turn
        best = min(turns, key=lambda t: min(abs(t[0] - end), abs(t[1] - start)))[2] if turns else None
    return best


def smooth_speakers(words, lang):
    """Pitch votes flip on single tokens; an isolated short run A-B-A becomes A-A-A (real turns persist)."""
    min_chars = 5 if lang in NO_SPACE_LANGS else 2
    while True:
        runs = []  # [label, first index, last index]
        for i, w in enumerate(words):
            if runs and runs[-1][0] == w["spk"]:
                runs[-1][2] = i
            else:
                runs.append([w["spk"], i, i])
        changed = False
        for k in range(1, len(runs) - 1):
            lab, a, b = runs[k]
            if runs[k - 1][0] != runs[k + 1][0] or runs[k - 1][0] == lab:
                continue
            dur = words[b]["end"] - words[a]["start"]
            chars = sum(len(w["word"].strip()) for w in words[a:b + 1])
            if dur < 1.0 or chars < min_chars:
                for w in words[a:b + 1]:
                    w["spk"] = runs[k - 1][0]
                changed = True
                break
        if not changed:
            return


def build_cues(words, lang, max_dur=6.0, max_gap=0.7):
    max_chars = 28 if lang in NO_SPACE_LANGS else 64
    joiner = "" if lang in NO_SPACE_LANGS else " "
    cues, cur = [], None

    def flush():
        nonlocal cur
        if cur and cur["text"].strip():
            cur["text"] = cur["text"].strip()
            cues.append(cur)
        cur = None

    for w in words:
        txt = w["word"].strip() if lang in NO_SPACE_LANGS else w["word"]
        if cur is not None:
            gap = w["start"] - cur["end"]
            dur = w["end"] - cur["start"]
            ends_sentence = bool(re.search(r"[。！？!?]\s*$", cur["text"]))
            # a segment's first token often carries a placeholder timestamp: never leave a 1-2 char cue behind
            tiny = len(cur["text"].strip()) < (3 if lang in NO_SPACE_LANGS else 2)
            if ((w["spk"] != cur["spk"] and not tiny) or (gap > max_gap and not tiny) or dur > max_dur
                    or len(cur["text"]) + len(txt) > max_chars or (ends_sentence and dur > 1.5)):
                flush()
        if cur is None:
            cur = {"start": w["start"], "end": w["end"], "text": txt.strip(), "spk": w["spk"]}
        else:
            cur["text"] += joiner + txt if lang in NO_SPACE_LANGS else txt
            cur["end"] = w["end"]
    flush()
    for c in cues:  # readable minimum duration, without overlapping the next cue
        c["end"] = max(c["end"], c["start"] + 1.0)
    for a, b in zip(cues, cues[1:]):
        if a["end"] > b["start"]:
            a["end"] = max(a["start"] + 0.3, b["start"] - 0.05)
    return cues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--compute", default="float16")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hf-token", default="")
    ap.add_argument("--diarize-model", default="pyannote/speaker-diarization-3.1")
    a = ap.parse_args()

    import torch
    from faster_whisper import WhisperModel, decode_audio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = a.compute if device == "cuda" else "int8"
    log("device %s, model %s (%s)" % (device, a.model, compute))
    audio = decode_audio(a.audio, sampling_rate=SR).astype(np.float32)
    total = len(audio) / SR
    progress(2)

    model = WhisperModel(a.model, device=device, compute_type=compute)
    progress(8)
    seg_iter, info = model.transcribe(
        audio, language=a.lang, beam_size=5, word_timestamps=True, temperature=[0.0, 0.3, 0.6],
        condition_on_previous_text=False, hallucination_silence_threshold=2.0, vad_filter=False,
    )
    words, spans, dropped, raw = [], [], 0, 0
    for s in seg_iter:
        raw += 1
        progress(8 + 52 * min(1.0, s.end / total))
        text = (s.text or "").strip()
        if not text or not s.words:
            continue
        if (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0) or s.compression_ratio > 2.4 or is_hallucination(text):
            dropped += 1
            log("drop %.1f-%.1f (nsp %.2f lp %.2f cr %.2f): %s" % (s.start, s.end, s.no_speech_prob, s.avg_logprob, s.compression_ratio, text[:40]))
            continue
        spans.append((float(s.start), float(s.end)))
        for w in s.words:
            if w.word.strip():
                words.append({"start": float(w.start), "end": float(w.end), "word": w.word, "spk": ""})
    log("segments %d, dropped %d, words %d" % (raw, dropped, len(words)))
    del model; gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    progress(62)

    cast = {}
    turns = diarize(audio, a.hf_token, a.diarize_model)
    progress(78)
    if turns:
        # name diarized speakers by voice pitch: F1, M1, F2, ...
        by_label = {}
        for w in words:
            w["spk"] = speaker_at(turns, w["start"], w["end"]) or ""
            by_label.setdefault(w["spk"], []).append(w)
        names, counts = {}, {"F": 0, "M": 0, "?": 0}
        for lab, ws in sorted(by_label.items(), key=lambda kv: -len(kv[1])):
            f0s = [f for f in (pitch_hz(audio, w["start"], w["end"]) for w in ws[:: max(1, len(ws) // 60)]) if f]
            f0 = float(np.median(f0s)) if f0s else None
            g = gender(f0); counts[g] += 1
            names[lab] = "%s%d" % (g, counts[g])
            cast[names[lab]] = {"lines": 0, "f0": round(f0) if f0 else None}
        for w in words:
            w["spk"] = names.get(w["spk"], "")
        smooth_speakers(words, a.lang)
        log("speakers: " + ", ".join("%s=%s" % (lab, names[lab]) for lab in names))
        cues = build_cues(words, a.lang)
    else:
        label_words_by_pitch(audio, words, spans)
        smooth_speakers(words, a.lang)
        cues = build_cues(words, a.lang)
    for c in cues:
        cast.setdefault(c["spk"], {"lines": 0, "f0": None})["lines"] += 1
    progress(95)

    segs = [{"start": round(c["start"], 3), "end": round(c["end"], 3), "text": c["text"], "spk": c["spk"]} for c in cues]
    log("cues %d, cast %s" % (len(segs), json.dumps(cast, ensure_ascii=False)))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "segments.json"), "w", encoding="utf-8") as f:
        json.dump({"lang": a.lang, "cast": cast, "segments": segs}, f, ensure_ascii=False, indent=1)
    write_srt(os.path.join(a.out_dir, "orig.srt"), segs)
    write_vtt(os.path.join(a.out_dir, "orig.vtt"), segs)
    progress(100)


if __name__ == "__main__":
    main()

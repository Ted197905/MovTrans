#!/usr/bin/env python3
"""
Transcription with word timestamps, hallucination filtering, speaker labels and subtitle-sized cues.

  transcribe.py --audio audio.wav --lang ja --model large-v3 --compute float16 --out-dir DIR [--hf-token TOKEN]

Pipeline: faster-whisper over the whole audio, cut into <=30 s clips that start at silero speech onsets (a window that
starts in silence/music makes Whisper skip the first utterances; VAD filtering itself would drop quiet dialogue between
moans). Windows whose output was rejected get one more pass with a shifted start. -> drop hallucinated / non-speech segments
-> speaker labels (pyannote diarization when --hf-token is given, else per-cue pitch -> F/M)
-> regroup words into cues (max 6 s / ~28 CJK chars, split at pauses and speaker changes).

Writes DIR/segments.json {"lang", "cast": {tag: {"lines", "f0"}}, "segments": [{start,end,text,spk}]},
DIR/orig.srt, DIR/orig.vtt. Progress goes to stderr as "progress <0..100>".
"""
import argparse
import gc
import json
import math
import os
import re
import zlib

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
    m = re.fullmatch(r"(.)\1{2,}", t) or re.fullmatch(r"(..)\1{2,}", t)  # ああああ / はぁはぁはぁ (but not 行く行く行く)
    if m and MOAN_JA.fullmatch(m.group(1)):
        return True
    if len(set(t)) <= 2 and len(t) >= 6 and MOAN_JA.fullmatch(t):
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


def clips_for(audio, max_len=30.0, pad=0.3):
    """Partition of the whole timeline into <=30 s clips, cutting just before each speech onset that follows a pause."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    total = len(audio) / SR
    cuts, prev_end = [0.0], 0.0
    for t in get_speech_timestamps(audio, VadOptions(threshold=0.3, min_silence_duration_ms=300, speech_pad_ms=0)):
        s, e = t["start"] / SR, t["end"] / SR
        if s - prev_end > 1.0 and s - pad - cuts[-1] > 1.0:
            cuts.append(s - pad)
        prev_end = e
    cuts.append(total)
    out = []
    for x, y in zip(cuts, cuts[1:]):
        n = max(1, math.ceil((y - x) / max_len)); step = (y - x) / n
        out += [[x + i * step, x + (i + 1) * step] for i in range(n)]
    return out


# Real phrases Whisper also likes to invent on non-speech; dropped only when the window itself looks non-speech.
SUSPICIOUS = {"ありがとうございました", "ありがとうございます", "おやすみ", "おや", "すみ", "良い一日を", "はい", "thank you", "thanks", "bye"}


LAUGH = re.compile(r"(ハハ|はは|ふふ|フフ|笑|haha|hehe|lol)", re.I)


def sound_kind(text):
    """Non-dialogue vocalisation Whisper wrote down: what an SDH track should show instead."""
    t = re.sub(r"[\s。、．，,.!！?？…・「」\"'()（）\-〜～ー]", "", text).lower()
    if not t:
        return None
    if LAUGH.search(text):
        return "laugh"
    if MOAN_JA.fullmatch(t) or MOAN_LATIN.fullmatch(t):
        return "moan"
    m = re.fullmatch(r"(.)\1{2,}", t) or re.fullmatch(r"(..)\1{2,}", t)
    if m and MOAN_JA.fullmatch(m.group(1)):
        return "moan"
    return None


def repetitive(text):
    """Whisper's compression ratio is per 30 s window, so one moan run would sink every line in the window;
    this is the same test on the segment's own text."""
    b = text.encode("utf-8")
    return len(b) >= 24 and len(b) / len(zlib.compress(b)) > 2.4


def looping(kept, s):
    """The same sentence three or more times in a row is a decoder loop: keep the first, drop the rest."""
    t = re.sub(r"[\s。、．，,.!！?？…]", "", s.text or "")
    same = 0
    for k in reversed(kept):
        if re.sub(r"[\s。、．，,.!！?？…]", "", k.text or "") == t and s.start - k.end < 8.0:
            same += 1
        else:
            break
    return same >= 2


def keep_segment(s, min_word_prob=0.0):
    text = (s.text or "").strip()
    if not text or not s.words:
        return False
    if (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0) or repetitive(text) or is_hallucination(text):
        return False
    core = re.sub(r"[\s。、．，,.!！?？…]", "", text).lower()
    if core in SUSPICIOUS and s.no_speech_prob > 0.4:
        return False
    return sum(w.probability for w in s.words) / len(s.words) >= min_word_prob


def fill_pass(model_name, audio, clip_list, lang, device, compute, total):
    """Second Whisper model (no word timestamps: distil models cannot align) + wav2vec2 forced alignment."""
    from faster_whisper import WhisperModel
    m = WhisperModel(model_name, device=device, compute_type=compute)
    seg_iter, _ = m.transcribe(audio, language=lang, beam_size=5, temperature=[0.0, 0.3], word_timestamps=False,
                               condition_on_previous_text=False, vad_filter=False,
                               clip_timestamps=[x for c in clip_list for x in c])
    starts = [c[0] for c in clip_list]
    segs = []
    for s in seg_iter:
        progress(60 + 8 * min(1.0, s.end / total))
        text = (s.text or "").strip()
        if not text or not keep_segment_plain(s):
            continue
        core = re.sub(r"[\s。、．，,.!！?？…]", "", text)
        if core in ("ごめん", "ごめんなさい", "すいません", "すみません") and any(abs(s.start - c) < 0.6 for c in starts):
            continue  # kotoba-whisper's own filler on a window that starts in noise
        segs.append({"start": float(s.start), "end": float(s.end), "text": text})
    del m; gc.collect()
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()
    log("fill pass: %d segment(s) kept" % len(segs))
    if not segs:
        return []
    words = []
    try:
        import whisperx
        align_model, meta = whisperx.load_align_model(language_code=lang, device=device)
        res = whisperx.align(segs, align_model, meta, audio, device, return_char_alignments=False)
        del align_model; gc.collect()
        for s in res["segments"]:
            for w in s.get("words", []):
                if "start" in w and w["word"].strip():
                    words.append({"start": float(w["start"]), "end": float(w["end"]), "word": w["word"], "spk": ""})
    except Exception as e:
        log("fill alignment failed (%s); using segment times" % str(e).split("\n")[0][:200])
        for s in segs:
            words.append({"start": s["start"], "end": s["end"], "word": s["text"], "spk": ""})
    for s in segs[:400]:
        log("fill %.1f-%.1f: %s" % (s["start"], s["end"], s["text"][:40]))
    return words


def keep_segment_plain(s):
    """keep_segment for segments without word probabilities."""
    text = (s.text or "").strip()
    if (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0) or repetitive(text) or is_hallucination(text):
        return False
    core = re.sub(r"[\s。、．，,.!！?？…]", "", text).lower()
    return not (core in SUSPICIOUS and s.no_speech_prob > 0.4)


def smooth_speakers(words, lang):
    """Pitch votes flip on single tokens: a short run joins the neighbouring run closest in time (isolated
    A-B-A flips vanish; a genuine one-word turn is absorbed too, which beats a fragment cue)."""
    min_chars = 5 if lang in NO_SPACE_LANGS else 2
    while True:
        runs = []  # [label, first index, last index]
        for i, w in enumerate(words):
            if runs and runs[-1][0] == w["spk"]:
                runs[-1][2] = i
            else:
                runs.append([w["spk"], i, i])
        changed = False
        for k, (lab, a, b) in enumerate(runs):
            dur = words[b]["end"] - words[a]["start"]
            chars = sum(len(w["word"].strip()) for w in words[a:b + 1])
            if dur >= 1.0 and chars >= min_chars and lab != "?":
                continue
            prev = runs[k - 1] if k > 0 else None
            nxt = runs[k + 1] if k + 1 < len(runs) else None
            gp = words[a]["start"] - words[prev[2]]["end"] if prev else None
            gn = words[nxt[1]]["start"] - words[b]["end"] if nxt else None
            if prev and (gn is None or gp <= gn) and gp < 1.5:
                target = prev[0]
            elif nxt and gn < 1.5:
                target = nxt[0]
            else:
                continue
            if target == lab:
                continue
            for w in words[a:b + 1]:
                w["spk"] = target
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
    ap.add_argument("--fill-model", default="", help="second Whisper model run on spans the first left empty (e.g. kotoba-tech/kotoba-whisper-v2.0-faster)")
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
    clips = clips_for(audio)
    log("%d clips" % len(clips))
    progress(8)

    def transcribe(clip_list, temps, base, span, min_word_prob=0.0):
        seg_iter, _ = model.transcribe(
            audio, language=a.lang, beam_size=5, word_timestamps=True, temperature=temps,
            condition_on_previous_text=False, vad_filter=False, clip_timestamps=[x for c in clip_list for x in c],
        )
        kept, rejected, n = [], [], 0
        for s in seg_iter:
            n += 1
            progress(base + span * min(1.0, s.end / total))
            if keep_segment(s, min_word_prob) and not looping(kept, s):
                kept.append(s)
            else:
                rejected.append((float(s.start), float(s.end)))
                kind = sound_kind((s.text or "").strip())
                if kind and s.end - s.start >= 0.8:
                    sounds.append({"start": round(float(s.start), 2), "end": round(min(float(s.end), float(s.start) + 30), 2), "kind": kind})
                log("drop %.1f-%.1f (nsp %.2f lp %.2f cr %.2f): %s" % (s.start, s.end, s.no_speech_prob, s.avg_logprob, s.compression_ratio, (s.text or "").strip()[:40]))
        return kept, rejected, n

    sounds = []
    kept, rejected, raw = transcribe(clips, [0.0, 0.3, 0.6], 8, 46)
    # windows Whisper turned into moans/hallucinations may still hold dialogue: retry them once from a shifted start
    retry = []
    for s, e in sorted(rejected):
        if retry and s - retry[-1][1] < 1.0:
            retry[-1][1] = max(retry[-1][1], e)
        else:
            retry.append([s, e])
    retry = [[s + 2.0, e] for s, e in retry if e - s >= 6.0]
    if retry:
        log("retrying %d rejected span(s)" % len(retry))
        covered = [(float(s.start), float(s.end)) for s in kept]
        kept2, _, _ = transcribe(retry, [0.0, 0.3], 54, 6, min_word_prob=0.5)
        for s in kept2:
            if not any(min(e, s.end) - max(b, s.start) > 0.3 for b, e in covered):
                log("retry-keep %.1f-%.1f (nsp %.2f lp %.2f cr %.2f wp %.2f): %s" % (s.start, s.end, s.no_speech_prob, s.avg_logprob,
                    s.compression_ratio, sum(w.probability for w in s.words) / len(s.words), (s.text or "").strip()[:40]))
                kept.append(s)
        kept.sort(key=lambda s: s.start)
    words, spans = [], []
    for s in kept:
        spans.append((float(s.start), float(s.end)))
        for w in s.words:
            if w.word.strip():
                words.append({"start": float(w.start), "end": float(w.end), "word": w.word, "spk": ""})
    log("segments %d, rejected %d, kept %d, words %d" % (raw, len(rejected), len(kept), len(words)))
    del model; gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    if a.fill_model:
        # a second model listens to everything the first left empty; a different model fails differently
        # (large-v3 turns moaning scenes into garbage windows, kotoba-whisper keeps the short lines between them)
        holes, pos = [], 0.0
        for s, e in sorted(spans) + [(total, total)]:
            if s - pos >= 3.0:
                holes.append([pos, s])
            pos = max(pos, e)
        fill_clips = []
        for hs, he in holes:
            for cs, ce in clips_for(audio[int(hs * SR):int(he * SR)]):
                fill_clips.append([hs + cs, hs + ce])
        log("fill pass: %s on %d span(s), %.0fs" % (a.fill_model, len(holes), sum(e - s for s, e in holes)))
        fill_words = fill_pass(a.fill_model, audio, fill_clips, a.lang, device, compute, total)
        for w in fill_words:
            words.append(w)
        spans += [(w["start"], w["end"]) for w in fill_words]
    words.sort(key=lambda w: w["start"])
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
    merged = []  # adjacent same-kind sounds become one SDH cue
    for snd in sorted(sounds, key=lambda x: x["start"]):
        if merged and merged[-1]["kind"] == snd["kind"] and snd["start"] - merged[-1]["end"] < 2.0:
            merged[-1]["end"] = max(merged[-1]["end"], snd["end"])
        else:
            merged.append(dict(snd))
    sounds = merged
    log("cues %d, cast %s" % (len(segs), json.dumps(cast, ensure_ascii=False)))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "segments.json"), "w", encoding="utf-8") as f:
        json.dump({"lang": a.lang, "cast": cast, "segments": segs, "sounds": sounds}, f, ensure_ascii=False, indent=1)
    write_srt(os.path.join(a.out_dir, "orig.srt"), segs)
    write_vtt(os.path.join(a.out_dir, "orig.vtt"), segs)
    progress(100)


if __name__ == "__main__":
    main()

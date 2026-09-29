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
from timing import align_cues, finish_cues, fix_word_spans, peak_db
from verify import decode_spans, keep as agree, vad_max, vad_probs

SR = 16000
NO_SPACE_LANGS = {"ja", "zh", "th"}

# Whole-segment phrases Whisper emits on silence, music or moaning (credits/outro training data).
HALLUCINATIONS = [
    "ご視聴ありがとうございました", "ご視聴ありがとうございます", "チャンネル登録", "高評価", "最後までご視聴",
    "字幕", "おやすみなさい", "また次の動画で", "次の動画", "お会いしましょう", "ご視聴", "開封して", "お疲れ様でした",
    "thank you for watching", "thanks for watching", "subtitles by", "subscribe", "like and subscribe",
    "please subscribe", "see you in the next video", "copyright", "amara.org",
    "시청해 주셔서 감사합니다", "구독", "좋아요", "谢谢观看", "感谢观看", "请订阅", "字幕由",
    "gracias por ver", "merci d'avoir regardé", "danke fürs zuschauen", "obrigado por assistir",
]


MOAN_JA = re.compile(r"[あぁいぃうぅえぇおぉんっーはひふへほ〜～]+")
MOAN_LATIN = re.compile(r"(?:[aeiou]+h*|h[aeiou]+|m+|hm+|uh+|ah+|oh+|mm+|mhm)")


NONSPEECH = {"笑い", "笑い声", "笑", "拍手", "音楽", "bgm", "効果音", "無音", "沈黙", "字幕", "music", "laughter", "applause", "silence"}


def is_hallucination(text):
    t = re.sub(r"[\s。、．，,.!！?？…・「」\"'()（）\-]", "", text).lower()
    if not t or MOAN_JA.fullmatch(t) or MOAN_LATIN.fullmatch(t):  # moans / interjections are not dialogue
        return True
    if t in NONSPEECH:  # the model described the sound instead of writing words
        return True
    m = re.fullmatch(r"(.)\1{2,}", t) or re.fullmatch(r"(..)\1{2,}", t)  # ああああ / はぁはぁはぁ (but not 行く行く行く)
    if m and MOAN_JA.fullmatch(m.group(1)):
        return True
    if len(set(t)) <= 2 and len(t) >= 6 and MOAN_JA.fullmatch(t):
        return True
    for h in HALLUCINATIONS:
        h2 = re.sub(r"[\s']", "", h).lower()
        if t == h2 or (h2 in t and len(t) <= len(h2) + 6):
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


def voice_hint(audio, cues):
    """Pitch hint for a diarized speaker from the F0 frames of its cues. The share of low frames (< 160 Hz) is
    used instead of a median: a man's cues often carry the woman's higher voice underneath, which drags a
    median up, while a woman's cues rarely contain many low frames."""
    mine = [c for c in cues if c["end"] - c["start"] >= 0.6]
    f0 = []
    for c in mine[:: max(1, len(mine) // 120)]:
        _, f = yin_f0(audio[int(c["start"] * SR):int(c["end"] * SR)])
        f0.extend(f.tolist())
    if len(f0) < 20:
        return {"f0": None, "voice": "unclear"}
    f0 = np.array(f0)
    low = float((f0 < 160).mean())
    return {"f0": int(np.median(f0)), "low_share": round(low, 2),
            "voice": "male-sounding" if low > 0.4 else "female-sounding" if low < 0.25 else "ambiguous"}


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


def diarize_sortformer(audio_path, python_bin):
    """[(start, end, label)] from NVIDIA streaming Sortformer, run in its own venv; None when unavailable."""
    if not python_bin or not os.path.isfile(python_bin):
        return None
    import subprocess
    import tempfile
    out = os.path.join(tempfile.gettempdir(), "sortformer_%d.json" % os.getpid())
    try:
        r = subprocess.run([python_bin, os.path.join(os.path.dirname(os.path.abspath(__file__)), "diarize_sortformer.py"),
                            "--audio", audio_path, "--out", out], capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip().splitlines()[-1][:300] if r.stderr.strip() else "exit %d" % r.returncode)
        turns = [(float(s), float(e), str(l)) for s, e, l in json.load(open(out, encoding="utf-8"))]
        log([ln for ln in r.stderr.splitlines() if ln.startswith("sortformer:")][-1] if "sortformer:" in r.stderr else "sortformer: %d turns" % len(turns))
        return turns
    except Exception as e:
        log("sortformer skipped: %s" % str(e).split("\n")[0][:300])
        return None
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass


def agreement(words, turns_a, turns_b):
    """Fraction of words on which two diarizations agree, after matching B's labels to A's by overlap."""
    pairs = {}
    for w in words:
        la, lb = speaker_at(turns_a, w["start"], w["end"]), speaker_at(turns_b, w["start"], w["end"])
        if la and lb:
            pairs.setdefault(lb, {}).setdefault(la, 0)
            pairs[lb][la] += 1
    mapping = {lb: max(d, key=d.get) for lb, d in pairs.items()}
    n = sum(sum(d.values()) for d in pairs.values())
    hit = sum(d[mapping[lb]] for lb, d in pairs.items())
    return hit / n if n else 0.0


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


PUNCT = re.compile(r"[\s。、．，,.!！?？…・「」\"'()（）\-]")


def undouble(words):
    """A segment that is the same phrase written twice ("お腹いっぱいになったら、お腹いっぱいになったら、") is a
    decoder stutter: keep the words of the first half."""
    core = "".join(PUNCT.sub("", w["word"]) for w in words)
    h = len(core) // 2
    if len(core) < 8 or len(core) % 2 or core[:h] != core[h:]:
        return words
    n, out = 0, []
    for w in words:
        out.append(w)
        n += len(PUNCT.sub("", w["word"]))
        if n >= h:
            break
    return out


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


def fill_pass(m, audio, clip_list, lang, device, total, verify):
    """Second Whisper model (no word timestamps: distil models cannot align) + wav2vec2 forced alignment.
    verify(segs) -> the segments the first model agrees with."""
    seg_iter, _ = m.transcribe(audio, language=lang, beam_size=5, temperature=[0.0, 0.3], word_timestamps=False,
                               condition_on_previous_text=False, vad_filter=False,
                               clip_timestamps=[x for c in clip_list for x in c])
    starts = [c[0] for c in clip_list]
    segs = []
    for s in seg_iter:
        progress(60 + 8 * min(1.0, s.end / total))
        text = (s.text or "").strip()
        if not text or not keep_segment_plain(s) or s.avg_logprob < -0.85:  # a fill line needs to be a confident one
            continue
        if peak_db(audio, s.start, s.end) < -55:  # nothing audible there
            continue
        core = re.sub(r"[\s。、．，,.!！?？…]", "", text)
        if core in ("ごめん", "ごめんなさい", "すいません", "すみません") and any(abs(s.start - c) < 0.6 for c in starts):
            continue  # kotoba-whisper's own filler on a window that starts in noise
        segs.append({"start": float(s.start), "end": float(s.end), "text": text})
    n = len(segs)
    segs = verify(segs)
    log("fill pass: %d segment(s), %d confirmed" % (n, len(segs)))
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


BREAK_JA = re.compile(r"([。、！？!?…]|[てでにはがをとねよかのも])$")  # punctuation or a particle: a place to cut
# the "particle" is part of a word when this follows it (です, ている, かも, から, とき...)
NO_BREAK_JA = {"です", "でし", "てい", "てる", "てた", "てく", "てみ", "てお", "てあ", "から", "かな", "かも", "とき", "とこ", "にな", "ので", "もう",
               "はい", "がっ", "ねえ", "よう", "のに", "では", "でも", "とも", "には", "にも", "かし"}


def build_cues(words, lang, max_dur=6.0, max_gap=0.7):
    """Words -> subtitle cues. A cue ends at a speaker change, a pause, 6 s or ~28 CJK chars; when it is the
    length that ends it, the cut moves back to the last punctuation/particle so words are not split in half."""
    max_chars = 28 if lang in NO_SPACE_LANGS else 64
    cjk = lang in NO_SPACE_LANGS
    cues, cur = [], []

    def text_of(ws):
        return ("".join(w["word"].strip() for w in ws) if cjk else "".join(w["word"] for w in ws)).strip()

    def flush(ws):
        t = text_of(ws)
        if t:
            cues.append({"start": ws[0]["start"], "end": ws[-1]["end"], "text": t, "spk": ws[0]["spk"]})

    for w in words:
        txt = w["word"].strip() if cjk else w["word"]
        if cur:
            ctext = text_of(cur)
            gap = w["start"] - cur[-1]["end"]
            dur = w["end"] - cur[0]["start"]
            ends_sentence = bool(re.search(r"[。！？!?]\s*$", ctext))
            # a segment's first token often carries a placeholder timestamp: never leave a 1-2 char cue behind
            tiny = len(ctext) < (3 if cjk else 2)
            too_long = dur > max_dur or len(ctext) + len(txt) > max_chars
            # a pitch-vote speaker flip in the middle of a phrase (no pause, no break character) is noise, not a new speaker
            spk_change = w["spk"] != cur[0]["spk"] and not tiny and (not cjk or gap > 0.3 or BREAK_JA.search(ctext))
            if spk_change or (gap > max_gap and not tiny) or (ends_sentence and dur > 1.5):
                flush(cur); cur = []
            elif too_long:
                cut = len(cur)  # default: cut here
                if cjk and len(cur) > 3:
                    # best break point in the tail: a pause first, then punctuation, then a particle; never
                    # before a small kana or a long-vowel mark (that is the middle of a word: か|っこいい)
                    best = 0.0
                    for j in range(1, len(cur)):
                        if len(text_of(cur[:j + 1])) < 6 or re.match(r"[ぁぃぅぇぉっゃゅょァィゥェォッャュョー]", cur[j + 1]["word"].strip() if j + 1 < len(cur) else txt):
                            continue
                        wtxt = cur[j]["word"].strip()
                        nxt = cur[j + 1]["word"].strip() if j + 1 < len(cur) else txt
                        score = min(cur[j + 1]["start"] - cur[j]["end"] if j + 1 < len(cur) else gap, 0.6)
                        if re.search(r"[。、！？!?…]$", wtxt):
                            score += 0.5
                        elif re.search(r"(です|ます|ました|でした|ですか|ますか|でしょ|だよ|だね|ない)$", text_of(cur[:j + 1])) and nxt[:1] not in "かねよ":
                            score += 0.4  # a sentence ending without punctuation
                        elif BREAK_JA.search(wtxt) and (wtxt[-1] + nxt[:1]) not in NO_BREAK_JA:
                            score += 0.2
                        if score >= best and score > 0:
                            best, cut = score, j + 1
                flush(cur[:cut]); cur = cur[cut:]
        cur.append(dict(w, word=txt if cjk else w["word"]))
    if cur:
        flush(cur)
    merged = []  # a 1-2 char tail ("ね", "うん") right after a cue joins it
    for c in cues:
        if merged and len(c["text"]) <= (2 if cjk else 1) and c["start"] - merged[-1]["end"] < 0.5 \
                and len(merged[-1]["text"]) + len(c["text"]) <= max_chars + 2:
            merged[-1]["text"] += ("" if cjk else " ") + c["text"]
            merged[-1]["end"] = c["end"]
        else:
            merged.append(c)
    return merged


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
    ap.add_argument("--diarizer", default="auto", choices=["auto", "pyannote", "sortformer", "both", "pitch"],
                    help="auto = Sortformer (pyenv/nemo) if installed, else pyannote if the token works, else pitch; both = Sortformer + pyannote cross-check")
    ap.add_argument("--nemo-python", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pyenv", "nemo", "bin", "python"))
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
            if keep_segment(s, min_word_prob) and not looping(kept, s) and peak_db(audio, s.start, s.end) >= -55:
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
    log("segments %d, rejected %d, kept %d" % (raw, len(rejected), len(kept)))
    probs = vad_probs(audio)
    fill_model = WhisperModel(a.fill_model, device=device, compute_type=compute) if a.fill_model else None

    def confirmed(segs, other, what):
        """Hallucination check: keep a line only when the VAD hears speech or the other model decodes the same words."""
        if other is None or not segs:
            return segs
        texts = decode_spans(other, audio, [(x["start"], x["end"]) for x in segs], a.lang)
        out = []
        for x, t in zip(segs, texts):
            v = vad_max(probs, x["start"], x["end"])
            if agree(x["text"], t, v):
                out.append(x)
            else:
                log("unconfirmed %s %.1f-%.1f (vad %.2f): %s | other: %s" % (what, x["start"], x["end"], v, x["text"][:30], t[:30]))
        return out

    main_segs = [{"start": float(s.start), "end": float(s.end), "text": (s.text or "").strip(), "seg": s} for s in kept]
    main_segs = confirmed(main_segs, fill_model, "main")
    log("main pass: %d of %d segment(s) confirmed" % (len(main_segs), len(kept)))
    progress(56)
    words, spans = [], []
    for x in main_segs:
        s = x["seg"]
        spans.append((x["start"], x["end"]))
        ws = [{"start": float(w.start), "end": float(w.end), "word": w.word, "spk": ""} for w in s.words if w.word.strip()]
        words += fix_word_spans(undouble(ws), a.lang)

    if fill_model:
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
        fill_words = fill_pass(fill_model, audio, fill_clips, a.lang, device, total, lambda segs: confirmed(segs, model, "fill"))
        for w in fill_words:
            words.append(w)
        spans += [(w["start"], w["end"]) for w in fill_words]
    words.sort(key=lambda w: w["start"])
    del model, fill_model; gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    log("words %d" % len(words))
    progress(62)

    cast = {}
    # Sortformer first: on this material it keeps one cluster per person across scenes, pyannote 3.1 splits a
    # person into several clusters by acoustics (5 clusters for 2 people). pyannote = cross-check / fallback.
    turns = None
    if a.diarizer in ("auto", "sortformer", "both"):
        turns = diarize_sortformer(a.audio, a.nemo_python)
        if turns:
            log("diarization: Sortformer, %d turns, %d speakers" % (len(turns), len({t[2] for t in turns})))
    if a.diarizer == "both" or (a.diarizer in ("auto", "pyannote") and not turns):
        turns2 = diarize(audio, a.hf_token, a.diarize_model)
        if turns2 and turns:
            log("diarization cross-check: Sortformer vs pyannote (%d speakers) agree on %.0f%% of words"
                % (len({t[2] for t in turns2}), 100 * agreement(words, turns, turns2)))
        elif turns2:
            turns = turns2
            log("diarization: pyannote, %d turns, %d speakers" % (len(turns), len({t[2] for t in turns})))
    progress(78)
    if turns:
        # diarized clusters are named S1, S2, ... by size; the voice pitch is only a hint for the translator,
        # because shouting men and moaning women overlap in pitch (F/M by pitch alone was often wrong)
        by_label = {}
        for w in words:
            w["spk"] = speaker_at(turns, w["start"], w["end"]) or ""
            by_label.setdefault(w["spk"], []).append(w)
        names = {}
        for k, (lab, ws) in enumerate(sorted(by_label.items(), key=lambda kv: -len(kv[1])), 1):
            names[lab] = "S%d" % k
            cast[names[lab]] = {"lines": 0, "f0": None, "voice": "unclear"}  # pitch hint is measured on the cues below
        for w in words:
            w["spk"] = names.get(w["spk"], "")
        # diarization is trusted as it is (smoothing is for pitch-vote noise); only unlabeled words take a neighbour
        last = ""
        for w in words:
            if not w["spk"] and last:
                w["spk"] = last
            last = w["spk"] or last
        log("speakers: " + ", ".join("%s=%s" % (lab, names[lab]) for lab in names))
        cues = build_cues(words, a.lang)
    else:
        label_words_by_pitch(audio, words, spans)
        smooth_speakers(words, a.lang)
        cues = build_cues(words, a.lang)
    # Whisper's word times are rough (a segment's first word absorbs the pause before it, so the cue would appear
    # while the previous line is still being said): every cue is force-aligned to the audio
    moved = align_cues(audio, cues, a.lang, device)
    log("aligned %d of %d cues" % (moved, len(cues)))
    for c in finish_cues(cues):
        log("drop duplicate %.1f-%.1f: %s" % (c["start"], c["end"], c["text"][:40]))
    progress(92)
    for c in cues:
        cast.setdefault(c["spk"], {"lines": 0, "f0": None})["lines"] += 1
    if turns:
        for spk, info in cast.items():
            info.update(voice_hint(audio, [c for c in cues if c["spk"] == spk]))
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

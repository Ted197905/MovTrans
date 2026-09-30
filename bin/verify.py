"""Hallucination check by agreement: on moaning, breathing and music Whisper writes plausible sentences that a second
model does not hear. A line is kept when a VAD hears speech there, or when the other model, decoding just that span,
writes (nearly) the same thing. Silero VAD alone cannot do this: it scores whispered lines like moans."""
import difflib
import re

import numpy as np

SR = 16000
FRAME = 512
PUNCT = re.compile(r"[\s。、．，,.!！?？…・「」\"'()（）\-〜～ー]")
# things every Whisper variant writes on non-speech (kotoba is distilled from large-v3): need a VAD vote as well
FILLERS = {"ごめん", "ごめんなさい", "すいません", "すみません", "ごちそう", "ごちそうさま", "ごちそうさまでした", "ありがとう",
           "ありがとうございました", "ありがとうございます", "おめでとう", "おめでとうございます", "お疲れ様", "お疲れ様でした",
           "よろしくお願いします", "いただきます", "やったー", "はい", "うん", "よいしょ", "おやすみなさい", "おやすみ", "さあ", "ご",
           "thank you", "thanks", "okay", "ok"}


def vad_probs(audio):
    """Silero speech probability per 32 ms frame."""
    from faster_whisper.vad import get_vad_model
    n = len(audio) // FRAME * FRAME
    return np.asarray(get_vad_model()(audio[:n])).reshape(-1)


def vad_max(probs, start, end):
    q = probs[max(0, int(start * SR / FRAME)):int(end * SR / FRAME) + 1]
    return float(q.max()) if len(q) else 0.0


def similarity(a, b):
    """How much of line a the other decode b contains. b comes from a wider window, so it may carry the neighbouring
    words too: a is matched against every stretch of b of about its own length and the best one counts."""
    a, b = PUNCT.sub("", a).lower(), PUNCT.sub("", b).lower()
    if not a or not b:
        return 0.0
    w = len(a) + 2
    if len(b) <= w:
        return difflib.SequenceMatcher(None, a, b).ratio()
    return max(difflib.SequenceMatcher(None, a, b[i:i + w]).ratio() for i in range(0, len(b) - w + 1))


def decode_spans(model, audio, spans, lang, pad=1.0):
    """The other model's text for each (start, end) span, decoded on its own (no 30 s window to invent a story in).
    The pad matters: on a clip of one or two seconds Whisper answers with stock phrases (a real "大好き" came back
    as "お疲れ様です"), with a second of context on each side it hears the line."""
    if not spans:
        return []
    clips = [[max(0.0, s - pad), min(len(audio) / SR, e + pad)] for s, e in spans]
    it, _ = model.transcribe(audio, language=lang, beam_size=5, temperature=[0.0, 0.3], word_timestamps=False,
                             condition_on_previous_text=False, vad_filter=False, clip_timestamps=[x for c in clips for x in c])
    texts, j = [""] * len(clips), 0
    for s in it:
        while j < len(clips) - 1 and s.start >= clips[j + 1][0] - 0.01:
            j += 1
        texts[j] += (s.text or "").strip()
    return texts


def keep(text, other, vad):
    """The other model must hear about the same words (sim >= 0.5, or >= 0.3 with some VAD support). A loud moan
    or scream makes the VAD fire, so VAD alone never keeps a line: on those, both models invent different text
    (video 10: "おめでとう", "アサイドラスコスカー" at VAD 0.5-0.6 with no agreement)."""
    core = PUNCT.sub("", text).lower()
    sim = similarity(text, other)
    if core in FILLERS:  # both models write these on music (おめでとうございます over the opening tune): VAD must agree
        return vad >= 0.3 and sim >= 0.3
    if sim >= 0.99:
        return True
    if len(core) <= 2:
        return vad >= 0.3 and sim >= 0.3
    return sim >= 0.5 or (sim >= 0.3 and vad >= 0.1)

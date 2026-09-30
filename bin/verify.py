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


_kakasi = None


def reading(text):
    """Japanese text as hiragana (pykakasi), so 本当だ and ほんとだ, 可愛い and かわいい compare as the same words."""
    global _kakasi
    if _kakasi is None:
        try:
            import pykakasi
            _kakasi = pykakasi.kakasi()
        except Exception:
            _kakasi = False
    if not _kakasi:
        return text
    try:
        return "".join(x["hira"] for x in _kakasi.convert(text))
    except Exception:
        return text


def similarity(a, b):
    """How much of line a the other decode b contains. b comes from a wider window, so it may carry the neighbouring
    words too: a is matched against every stretch of b of about its own length and the best one counts."""
    a, b = reading(PUNCT.sub("", a)).lower(), reading(PUNCT.sub("", b)).lower()
    if not a or not b:
        return 0.0
    w = len(a) + 2
    if len(b) <= w:
        return difflib.SequenceMatcher(None, a, b).ratio()
    return max(difflib.SequenceMatcher(None, a, b[i:i + w]).ratio() for i in range(0, len(b) - w + 1))


def decode_spans(model, audio, spans, lang, pad=1.0):
    """The other model's text for each (start, end) span, decoded on its own (no 30 s window to invent a story in).
    Each span is cut out of the audio and decoded separately: clip_timestamps would merge overlapping windows
    (faster-whisper never seeks backwards), so with a pad the clips leaked into each other.
    The pad matters: on a clip of one or two seconds Whisper answers with stock phrases (a real "大好き" came back
    as "お疲れ様です"), with a second of context on each side it hears the line."""
    texts = []
    for s, e in spans:
        lo, hi = max(0, int((s - pad) * SR)), min(len(audio), int((e + pad) * SR))
        try:
            it, _ = model.transcribe(audio[lo:hi], language=lang, beam_size=5, temperature=[0.0, 0.3], word_timestamps=False,
                                     condition_on_previous_text=False, vad_filter=False)
            texts.append("".join((x.text or "").strip() for x in it))
        except Exception:
            texts.append("")
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
    if sim >= 0.5 or (sim >= 0.3 and vad >= 0.1):
        return True
    # nobody else heard the same words: a clearly voiced, ordinary-looking line still stays (four women talking at
    # once defeat the re-decode); a katakana-only string is the typical gibberish written on a moan
    return vad >= 0.5 and len(core) >= 3 and re.fullmatch(r"[ァ-ヶー・]+", core) is None

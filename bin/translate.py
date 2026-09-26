#!/usr/bin/env python3
"""
Context-aware Korean subtitle translation through Ollama.

  translate.py --segments segments.json --scenes scenes.json --lang ja --out-dir DIR \
               --ollama http://127.0.0.1:11434 --model MODEL --rating rated|unrated

1. Style guide ("bible"): one call over the overview + the whole script -> characters, relationships,
   speech level per speaker, tone, glossary. Saved as DIR/style.txt.
2. Batches of BATCH lines, each with the bible, the scene notes of that stretch, the previous Korean lines
   and a few upcoming source lines. Numbered-line output (LLMs keep numbering far better than JSON).
3. Lines that came back missing, untranslated or in another script are redone one at a time.

Writes DIR/ko.srt and DIR/ko.vtt.
"""
import argparse
import json
import os
import re
import time

from subs import log, ollama_chat, ollama_unload, progress, strip_think, write_srt, write_vtt

BATCH = 25
LOOKAHEAD = 4
FOREIGN = re.compile(r"[぀-ヿ一-鿿฀-๿Ѐ-ӿ]")  # kana, CJK ideographs, Thai, Cyrillic
# "12. text", tolerating leaked speaker tags: "12. [F] text", "12. F. text", "12. (M1) text", "12. F: text"
NUMBERED = re.compile(r"^\s*(\d{1,3})\s*[.):]\s*(?:[\[(]?[FM?]\d?[\])]?\s*[.:\-]?\s*)?(.*?)\s*$")

RATING = {
    "rated": "Content level: RATED, like a US theatrical release. Keep the meaning, insults and innuendo, but phrase "
             "profanity and sexual content the way Korean theatrical release subtitles do: natural and non-graphic, "
             "without explicit slang for sexual acts or body parts.",
    "unrated": "Content level: UNRATED, like an adult video. Translate slang, profanity and explicit sexual content "
               "literally with direct, explicit Korean wording; never soften, censor or euphemize.",
}

SYSTEM = (
    "You are a veteran Korean subtitle translator (영상 번역가). You write subtitles the way a Korean native would "
    "actually say the line in that situation: natural spoken Korean, not textbook translation. Match each "
    "speaker's personality, mood and relationship; keep every speaker's speech level (반말/존댓말) and the way "
    "they address each other consistent for the whole video, as fixed in the style guide. Different speakers "
    "must not sound like one narrator. Short lines stay short; interjections become the Korean interjection a "
    "person would really use. Keep each subtitle readable: at most two lines of about 16 Korean characters. "
    "Every output line must be fully Korean (Hangul); never leave source-language words, romanization or "
    "other scripts. Never add notes, explanations, speaker tags or brackets. {rating}"
)

BIBLE_PROMPT = (
    "Below is the overview of a video and its full dialogue script ({lang}). Speaker tags: F = female voice, "
    "M = male voice, a number tells speakers of the same sex apart, ? = unknown.\n\n"
    "Overview:\n{summary}\n\nScript:\n{script}\n\n"
    "Write a concise style guide, in Korean, for translating these subtitles as a Korean subtitler would "
    "(no more than 350 characters, plain lines, no markdown):\n"
    "1. 등장인물: 태그별로 누구인지 (이름/호칭이 대사에 나오면 그대로), 성별, 대략 나이, 성격, 역할\n"
    "2. 관계와 말투: 각 인물이 상대에게 쓰는 말투(반말/존댓말/높임), 서로를 부르는 호칭. 영상 전체에서 고정\n"
    "3. 전체 톤과 장르, 번역 시 지킬 점 (표현 수위는 지시대로)\n"
    "4. 반복되는 고유명사/용어와 그 한국어 표기"
)

USER = (
    "Style guide:\n{bible}\n\n"
    "Scene notes for this stretch:\n{scenes}\n\n"
    "Previous subtitles (already translated, for continuity; do not repeat them):\n{prev}\n\n"
    "Upcoming lines after this batch (context only, do not translate):\n{ahead}\n\n"
    "Translate lines 1-{n} from {lang} into Korean subtitles. Answer with exactly {n} lines in the form "
    "\"<number>. <Korean subtitle>\", same numbering, nothing else.\n\n{lines}"
)


def bad(src, ko):
    return not ko or ko == src or FOREIGN.search(ko) is not None


def parse_numbered(reply, n):
    out = {}
    for line in strip_think(reply).splitlines():
        m = NUMBERED.match(line)
        if m and 1 <= int(m.group(1)) <= n and m.group(2):
            out.setdefault(int(m.group(1)), re.sub(r"^[\"'「」]+|[\"'「」]+$", "", m.group(2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--model", required=True)
    ap.add_argument("--rating", choices=sorted(RATING), default="rated")
    ap.add_argument("--num-ctx", type=int, default=16384)
    a = ap.parse_args()

    data = json.load(open(a.segments, encoding="utf-8"))
    segs = data["segments"]
    sc = json.load(open(a.scenes, encoding="utf-8"))
    summary = sc.get("summary") or "(none)"
    scenes = sc.get("scenes") or []
    tag = lambda s: ("[%s] " % s["spk"]) if s.get("spk") else ""
    system = SYSTEM.format(rating=RATING[a.rating])
    os.makedirs(a.out_dir, exist_ok=True)

    def chat(user, **kw):
        return strip_think(ollama_chat(a.ollama, a.model, [{"role": "system", "content": system},
                                                           {"role": "user", "content": user}], num_ctx=a.num_ctx, **kw))

    # 1. style guide from the whole script (first ~600 lines fit the context comfortably)
    script = "\n".join("%s%s" % (tag(s), s["text"]) for s in segs[:600])
    bible = "(none)"
    try:
        bible = chat(BIBLE_PROMPT.format(lang=a.lang, summary=summary, script=script)).strip()[:1200]
        log("style guide:\n" + bible)
    except Exception as e:
        log("style guide failed: %s" % e)
    with open(os.path.join(a.out_dir, "style.txt"), "w", encoding="utf-8") as f:
        f.write(bible)
    progress(5)

    def one(s):
        r = chat("Style guide:\n%s\n\nTranslate this one %s line into a Korean subtitle. Reply with the Korean text only.\n\n%s%s"
                 % (bible, a.lang, tag(s), s["text"]))
        return re.sub(r"^[\"'「」\s]+|[\"'「」\s]+$", "", r.splitlines()[0] if r.strip() else "")

    out, prev = [], []
    batches = [segs[i:i + BATCH] for i in range(0, len(segs), BATCH)]
    log("model %s, rating %s, %d lines in %d batches" % (a.model, a.rating, len(segs), len(batches)))
    for b, batch in enumerate(batches):
        lo = b * BATCH; hi = lo + len(batch)
        notes = [s["desc"] for s in scenes if lo <= s["idx"] < hi] or [s["desc"] for s in scenes if s["idx"] < hi][-2:]
        lines = "\n".join("%d. %s%s" % (k + 1, tag(s), s["text"]) for k, s in enumerate(batch))
        ahead = "\n".join("%s%s" % (tag(s), s["text"]) for s in segs[hi:hi + LOOKAHEAD]) or "(end)"
        user = USER.format(bible=bible, scenes="\n".join("- " + x for x in notes) or "(none)",
                           prev="\n".join(prev[-8:]) or "(start of video)", ahead=ahead, n=len(batch), lang=a.lang, lines=lines)
        got = {}
        for attempt in range(3):
            if attempt:
                time.sleep(5)  # an Ollama 500 usually means the runner crashed and is reloading
            try:
                got = parse_numbered(chat(user), len(batch))
                if len(got) >= len(batch) - 2:
                    break
                log("batch %d: got %d of %d lines" % (b, len(got), len(batch)))
            except Exception as e:
                log("batch %d attempt %d failed: %s" % (b, attempt + 1, e))
        ko, redo = [], 0
        for k, s in enumerate(batch):
            t = got.get(k + 1, "")
            if bad(s["text"], t):
                redo += 1
                try:
                    t2 = one(s)
                    t = t2 if not bad(s["text"], t2) else (re.sub(FOREIGN, "", t2).strip() or t)
                except Exception as e:
                    log("line redo failed: %s" % e)
            ko.append(t or s["text"])
        if redo:
            log("batch %d: %d line(s) redone" % (b, redo))
        for s, t in zip(batch, ko):
            out.append({"start": s["start"], "end": s["end"], "text": t})
        prev = ["%s%s" % (tag(s), t) for s, t in zip(batch, ko)]
        progress(5 + 95 * hi / len(segs))

    try:
        ollama_unload(a.ollama, a.model)  # free VRAM for the next job's Whisper run
    except Exception as e:
        log("unload failed: %s" % e)
    write_srt(os.path.join(a.out_dir, "ko.srt"), out)
    write_vtt(os.path.join(a.out_dir, "ko.vtt"), out)
    progress(100)


if __name__ == "__main__":
    main()

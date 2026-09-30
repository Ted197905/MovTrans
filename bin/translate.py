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
import difflib
import json
import os
import re
import time

from subs import drop_persistent_text, log, ollama_chat, ollama_unload, progress, strip_think, warn, write_srt, write_vtt

BATCH = 25
LOOKAHEAD = 3
POLISH_BATCH = 40
PROOF_BATCH = 40
FOREIGN = re.compile(r"[぀-ヿ一-鿿฀-๿Ѐ-ӿ]")  # kana, CJK ideographs, Thai, Cyrillic
# "12. text", tolerating leaked speaker tags: "12. [F] text", "12. F. text", "12. (M1) text", "12. F: text"
NUMBERED = re.compile(r"^\s*(\d{1,3})\s*[.):]\s*(?:[\[(]?(?:[FM?]\d?|S\d{1,2}(?:/NAR)?|SUB|TXT)[\])]?\s*[.:\-]?\s*)?(.*?)\s*$")

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
    "must not sound like one narrator. Speaker tags: [S1], [S2]... one diarized voice each (the style guide says "
    "who they are); [F]/[M]/[?] female/male/unknown by voice pitch when there was no diarization; [SUB] a subtitle "
    "burned into the picture (translate it as the line), [TXT] an on-screen "
    "caption such as a title, place, date or time (translate it as a caption, not as speech). A line that is "
    "narration (explaining the story to the viewer rather than spoken to someone in the scene, typically the "
    "narrator tag named in the style guide, and every line tagged [Sn/NAR]) is always written in Korean documentary "
    "narration style: polite formal endings (-습니다/-입니다/-했습니다), never casual 반말 endings. Short lines stay short; interjections become the Korean interjection a "
    "person would really use. Keep each subtitle readable: at most two lines of about 16 Korean characters. "
    "Every output line must be fully Korean (Hangul); never leave source-language words, romanization or "
    "other scripts. Japanese personal names are written by their Japanese reading, never by the Korean reading "
    "of the characters. Never transliterate other words by sound: interjections and slang (やばい, イク, すごい, "
    "気持ちいい...) are rendered by what they mean in that situation (やばい said in arousal or at climax is "
    "미치겠어/안 돼/너무 좋아, never 위험해; 出して at climax is 싸 줘). The source lines are speech recognition "
    "output and can still contain mishearings (a near-homophone, a word cut at the line end): translate what the "
    "speaker evidently meant in context. A line that is only meaningless syllables (recognition noise, not "
    "speech) is answered with a single \"-\" so it can be dropped. Never add notes, explanations, speaker tags or "
    "brackets. {rating}"
)

PROOF = (
    "The lines below are automatic speech recognition output ({lang}) from one video, in order, each with its "
    "speaker tag. Recognition errors are common: near-homophones (奈々様 or アンナさん heard for 旦那様/旦那さん, "
    "母 for もう/まあ, 寝て for なって, あざなち for 朝立ち), a word cut off at the end of a line, and gibberish syllables invented on "
    "moaning, breathing or music. Proofread the lines using the context of the whole batch. Rules: change a line "
    "only when you are confident it was misheard, and then only the misheard word(s); keep everything else exactly "
    "as written (same wording, punctuation and speech level; no polishing, no added words); a line that is "
    "meaningless syllables or clearly not speech becomes a single \"-\"; never merge, split, drop or reorder lines. "
    "Answer with exactly {n} lines in the form \"<number>. <line>\", same numbering, without the speaker tags, "
    "nothing else.\n\nVoices:\n{cast}\n\nLines:\n{lines}"
)

BIBLE_PROMPT = (
    "Below is the dialogue script of a video ({lang}), sampled in order. Speaker tags: [S1], [S2], ... are voice "
    "clusters from speaker diarization (normally one person each, but one person can get a second tag in another "
    "scene, and a small tag can be noise; the voice list gives a pitch hint), or F/M/? = "
    "female/male/unknown by voice pitch only when no diarization ran (then several people can share a tag). Work "
    "out who each tag is from what is said and how they are addressed.\n\nVoices:\n{cast}\n\n"
    "[SUB] = subtitle burned into the picture, [TXT] = on-screen caption (titles, place/time, narration).\n\n"
    "Script:\n{script}\n\n"
    "An automatic scene description is attached for atmosphere only; it is often wrong about people and roles, "
    "and the script above overrides it:\n{summary}\n\n"
    "Write a concise style guide, in Korean, for translating these subtitles as a Korean subtitler would "
    "(no more than 400 characters, plain lines, no markdown):\n"
    "1. 등장인물: 실제로 등장하는 사람들 (이름/호칭이 대사에 나오면 그대로), 성별, 대략 나이, 성격, 역할, 어느 태그로 나오는지\n"
    "2. 관계와 말투: 각 인물이 상대에게 쓰는 말투(반말/존댓말/높임), 서로를 부르는 호칭의 한국어 표기 (예: 旦那様 -> 서방님, ご主人様 -> 주인님; 원문 표기를 그대로 쓰지 않는다). 내레이션/해설이 있으면 그 문체(예: 다큐 해설체 '-습니다'). 영상 전체에서 고정\n"
    "3. 전체 톤과 장르, 번역 시 지킬 점 (표현 수위는 지시대로)\n"
    "4. 반복되는 고유명사(인명, 지명, 상품명)의 한국어 표기만. 일본 인명은 일본어 읽기로 적고 한자의 한국 음독은 쓰지 않는다 (이 영상에 없는 이름은 적지 않는다). "
    "일반 어휘, 은어, 신체 부위, 감탄사는 적지 않는다: 그런 말은 음차하지 않고 뜻으로 옮긴다"
)

POLISH = (
    "You are now the reviewing editor (감수). Below are source lines with their draft Korean subtitles. Rewrite each "
    "draft so it reads like a subtitle a Korean native subtitler would deliver: fix mistranslations against the "
    "source, keep each speaker's fixed speech level and way of addressing others (style guide), make the wording "
    "natural spoken Korean for this situation and mood, keep it short (two lines of ~16 characters max). Keep a "
    "good draft as it is. One subtitle per line, same numbering, Korean only, no tags, no comments.\n\n"
    "Style guide:\n{bible}\n\nScene notes:\n{scenes}\n\nLines (source => draft):\n{pairs}\n\n"
    "Answer with exactly {n} lines in the form \"<number>. <final Korean subtitle>\"."
)

USER = (
    "Style guide:\n{bible}\n\n"
    "Scene notes for this stretch:\n{scenes}\n\n"
    "Previous subtitles (already translated, for continuity; do not repeat them):\n{prev}\n\n"
    "Upcoming lines after this batch (context only, do not translate):\n{ahead}\n\n"
    "Translate lines 1-{n} from {lang} into Korean subtitles. Answer with exactly {n} lines in the form "
    "\"<number>. <Korean subtitle>\", same numbering, nothing else.\n\n{lines}"
)


# Whisper mishearings of the husband address this genre repeats all video long. Applied after the proofread when the
# script itself says 旦那様/旦那さん (5+ times) and the misheard form is rare (5 or fewer: a real character named 奈々
# or アンナ would be all over the script).
CONFUSIONS = {"奈々様": "旦那様", "アナ様": "旦那様", "あんな様": "旦那様", "アンナ様": "旦那様", "奈々さん": "旦那さん", "アンナさん": "旦那さん",
              "ナナ様": "旦那様", "奈々": "旦那様", "アンナ": "旦那さん"}
LATIN = re.compile(r"[A-Za-z]{3,}")

# words the model tends to leave in the source script; a plain Korean rendering beats deleting them
LEAKS = {"旦那様": "서방님", "旦那さん": "남편", "旦那": "남편", "ご主人様": "주인님", "ご主人": "남편", "お兄ちゃん": "오빠", "お姉ちゃん": "누나",
         "ママ": "엄마", "パパ": "아빠", "先生": "선생님", "社長": "사장님", "お母さん": "엄마", "お父さん": "아빠"}


def clean(ko):
    """Leaked speaker/screen tags out; a source word left untranslated becomes its usual Korean rendering."""
    ko = re.sub(r"\[(?:SUB|TXT|S\d{1,2}(?:/NAR)?|NAR|F|M|\?)\]", "", ko)
    for k, v in LEAKS.items():
        ko = ko.replace(k, v)
    return re.sub(r"\s+", " ", ko).strip(" ,")


def bad(src, ko):
    """Empty, untouched, still in the source script, or Latin letters that are not the source itself (a logo word
    such as MOODYZ may stay; "prezent" for PRESENTS may not)."""
    return not ko or ko == src or FOREIGN.search(ko) is not None or (LATIN.search(ko) is not None and ko.strip() != src.strip())


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
    ap.add_argument("--screen", default="", help="screen.json from ocr.py (burned-in subtitles / captions)")
    ap.add_argument("--no-polish", action="store_true", help="skip the editor pass")
    a = ap.parse_args()

    data = json.load(open(a.segments, encoding="utf-8"))
    segs = data["segments"]
    sounds = data.get("sounds") or []
    screen = []
    if a.screen and os.path.isfile(a.screen):
        for x in sorted(json.load(open(a.screen, encoding="utf-8")), key=lambda x: (x["start"], x.get("text", ""))):
            if not x.get("text"):
                continue
            x = dict(x, text=" ".join(x["text"].split()))  # one line: the numbered protocol cannot carry newlines
            if len("".join(x["text"].split())) < 2 or x["end"] - x["start"] < 1.5:
                continue  # seen in a single 1 fps frame: small/unstable text (listings, tickers), not a caption
            if not re.search(r"[^\W\d_]", x["text"]):
                continue  # digits and punctuation only: a rating badge or a counter, not a caption
            if x.get("type") == "subtitle" and x.get("y", 80) < 65:
                x["type"] = "caption"  # a burned-in dialogue subtitle sits at the bottom; higher up it is a title card
            key = "".join(x["text"].split())
            prev = next((m for m in reversed(screen[-8:]) if "".join(m["text"].split()) == key and x["start"] - m["end"] <= 4.0), None)
            if prev:
                prev["end"] = max(prev["end"], x["end"])
            else:
                screen.append(dict(x))
        screen = drop_persistent_text(screen, max((s["end"] for s in segs), default=0))
    # a burned-in subtitle carries the line already: the ASR cue under it is dropped, the subtitle is translated instead
    subs_iv = [(x["start"], x["end"]) for x in screen if x["type"] == "subtitle"]
    if subs_iv:
        before = len(segs)
        segs = [s for s in segs if not any(min(e, s["end"]) - max(b, s["start"]) > 0.5 * (s["end"] - s["start"]) for b, e in subs_iv)]
        log("%d ASR line(s) replaced by burned-in subtitles" % (before - len(segs)))
    for x in screen:  # translated together with the dialogue so the style guide applies
        segs.append({"start": x["start"], "end": max(x["end"], x["start"] + 1.5), "text": x["text"], "spk": "",
                     "screen": x["type"], "y": x.get("y", 50)})
    segs.sort(key=lambda s: s["start"])
    sc = json.load(open(a.scenes, encoding="utf-8"))
    summary = sc.get("summary") or "(none)"
    scenes = sc.get("scenes") or []
    tag = lambda s: ("[SUB] " if s.get("screen") == "subtitle" else "[TXT] " if s.get("screen") else
                     ("[%s/NAR] " % s["spk"]) if s.get("spk") in narr_tags else ("[%s] " % s["spk"]) if s.get("spk") else "")
    castd = data.get("cast") or {}
    for k, v in castd.items():  # ask the model, per voice, whether it is a narrator (formality alone does not tell)
        mine = [s["text"] for s in segs if s.get("spk") == k and not s.get("screen")]
        v["narration"] = False
        if len(mine) >= 20:
            sample = "\n".join(sorted(mine, key=len, reverse=True)[:15])
            try:
                r = ollama_chat(a.ollama, a.model, [{"role": "user", "content":
                    "These lines are all spoken by one voice in a video (%s):\n%s\n\nIs this voice a narrator, i.e. a "
                    "voice-over that describes or explains the story to the viewer (third person, no one answers it), "
                    "rather than a person talking with others in the scene? Answer with one word: yes or no." % (a.lang, sample)}],
                    num_ctx=a.num_ctx, num_predict=8)  # same num_ctx as every other call: a different one reloads the model
                v["narration"] = strip_think(r).strip().lower().startswith("yes")
            except Exception as e:
                log("narrator check failed for %s: %s" % (k, e))
            if v["narration"]:
                log("%s looks like the narrator" % k)
    cast = "\n".join("- [%s]: %d lines, %s voice%s%s" % (k, v.get("lines", 0), v.get("voice") or {"F": "female", "M": "male"}.get(k[:1], "unknown"),
                      (" (~%d Hz)" % v["f0"]) if v.get("f0") else "",
                      " - long formal declarative lines: most likely the NARRATOR" if v.get("narration") else "")
                      for k, v in castd.items()) or "(unknown)"
    narr_tags = [k for k, v in castd.items() if v.get("narration")]
    system = SYSTEM.format(rating=RATING[a.rating])
    os.makedirs(a.out_dir, exist_ok=True)

    def chat(user, **kw):
        kw.setdefault("num_predict", 2000)
        return strip_think(ollama_chat(a.ollama, a.model, [{"role": "system", "content": system},
                                                           {"role": "user", "content": user}], num_ctx=a.num_ctx, **kw))

    # 0. proofread the recognition output in context (homophones, cut words); noise lines are dropped.
    # The result is cached in proof.json so that a re-run of this stage alone skips the 25 min pass.
    fixed = dropped = 0
    dialog = [s for s in segs if not s.get("screen")]
    cache_path = os.path.join(a.out_dir, "proof.json")
    cache = {}
    if os.path.isfile(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
        if cache.get("n") == len(dialog):
            for s, t in zip(dialog, cache["lines"]):
                if t == "-":
                    s["drop"] = True
                elif t:
                    s["text"] = t
            log("proofread: cached")
            dialog = []
        else:
            cache = {}
    for i in range(0, len(dialog), PROOF_BATCH):
        chunk = dialog[i:i + PROOF_BATCH]
        lines = "\n".join("%d. %s%s" % (k + 1, tag(s), s["text"]) for k, s in enumerate(chunk))
        try:
            r = ollama_chat(a.ollama, a.model, [{"role": "user", "content": PROOF.format(lang=a.lang, n=len(chunk), cast=cast, lines=lines)}],
                            num_ctx=a.num_ctx, num_predict=60 * len(chunk))
            got = parse_numbered(r, len(chunk))
        except Exception as e:
            log("proofread %d failed: %s" % (i // PROOF_BATCH, e))
            warn("인식 교정 배치 %d 실패: 이 구간은 교정 없이 번역" % (i // PROOF_BATCH))
            continue
        for k, s in enumerate(chunk):
            t = got.get(k + 1, "").strip()
            if t in ("-", "—", "ー"):
                s["drop"] = True
                dropped += 1
                log("noise %.1f: %s" % (s["start"], s["text"][:40]))
            elif t and t != s["text"] and difflib.SequenceMatcher(None, t, s["text"]).ratio() >= 0.5:
                log("fix %.1f: %s -> %s" % (s["start"], s["text"][:30], t[:30]))
                s["text"] = t
                fixed += 1
        progress(1 + 4 * (i + len(chunk)) / max(1, len(dialog)))
    if dialog:
        log("proofread: %d line(s) fixed, %d noise line(s) dropped" % (fixed, dropped))
        os.makedirs(a.out_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"n": len(dialog), "lines": ["-" if s.get("drop") else s["text"] for s in dialog]}, f, ensure_ascii=False)
    segs = [s for s in segs if not s.get("drop")]
    script_text = "".join(s["text"] for s in segs if not s.get("screen"))
    if script_text.count("旦那様") + script_text.count("旦那さん") >= 5:
        n = 0
        for wrong, right in CONFUSIONS.items():
            if 0 < script_text.count(wrong) <= 5:
                for s in segs:
                    if not s.get("screen") and wrong in s["text"]:
                        s["text"] = s["text"].replace(wrong, right)
                        n += 1
        if n:
            log("husband address: %d line(s) corrected (%s)" % (n, ", ".join(CONFUSIONS)))

    # 1. style guide from the whole script (first ~600 lines fit the context comfortably)
    step = max(1, len(segs) // 300)  # ~300 lines spread over the whole video (CPU-side prefill is the slow part)
    script = "\n".join("%s%s" % (tag(s), s["text"]) for s in segs[::step][:300])
    bible = "(none)"
    try:
        bible = chat(BIBLE_PROMPT.format(lang=a.lang, summary=summary, script=script, cast=cast)).strip()[:1200]
        log("style guide:\n" + bible)
    except Exception as e:
        log("style guide failed: %s" % e)
        warn("스타일 가이드 생성 실패: 말투/호칭 일관성 없이 번역")
    with open(os.path.join(a.out_dir, "style.txt"), "w", encoding="utf-8") as f:
        f.write(bible)
    progress(5)

    def one(s):
        r = chat("Style guide:\n%s\n\nTranslate this one %s line into a Korean subtitle. Reply with the Korean text only.\n\n%s%s"
                 % (bible, a.lang, tag(s), s["text"]), num_predict=120)
        return re.sub(r"^[\"'「」\s]+|[\"'「」\s]+$", "", r.splitlines()[0] if r.strip() else "")

    out, prev = [], []
    batches = [segs[i:i + BATCH] for i in range(0, len(segs), BATCH)]
    log("model %s, rating %s, %d lines in %d batches" % (a.model, a.rating, len(segs), len(batches)))
    for b, batch in enumerate(batches):
        lo = b * BATCH; hi = lo + len(batch)
        t0, t1 = batch[0]["start"], batch[-1]["end"]  # scene notes by time (screen cues shift the line indices)
        notes = [s["desc"] for s in scenes if t0 - 10 <= s["t"] <= t1 + 10] or [s["desc"] for s in scenes if s["t"] < t1][-2:]
        notes = [n[:140] for n in notes[:: max(1, len(notes) // 5)][:6]]  # prompt size drives CPU prefill time
        lines = "\n".join("%d. %s%s" % (k + 1, tag(s), s["text"]) for k, s in enumerate(batch))
        ahead = "\n".join("%s%s" % (tag(s), s["text"]) for s in segs[hi:hi + LOOKAHEAD]) or "(end)"
        user = USER.format(bible=bible, scenes="\n".join("- " + x for x in notes) or "(none)",
                           prev="\n".join(prev[-6:]) or "(start of video)", ahead=ahead, n=len(batch), lang=a.lang, lines=lines)
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
            if t.strip() in ("-", "—") and not s.get("screen"):
                ko.append("-")
                continue
            t = clean(t)
            if bad(s["text"], t):
                redo += 1
                try:
                    t2 = clean(one(s))
                    t = t2 if not bad(s["text"], t2) else (re.sub(FOREIGN, "", t2).strip() or t)
                except Exception as e:
                    log("line redo failed: %s" % e)
            ko.append(t or s["text"])
        if redo:
            log("batch %d: %d line(s) redone" % (b, redo))
        if not got:
            warn("번역 배치 %d (%.0f초 부근) 실패: 원문 그대로 남음" % (b, batch[0]["start"]))
        for s, t in zip(batch, ko):
            cue = {"start": s["start"], "end": s["end"], "text": t, "spk": s.get("spk", "")}
            if s.get("screen") == "caption":
                cue["pos"] = "top"
                cue["y"] = s.get("y", 50)
            out.append(cue)
        prev = ["%s%s" % (tag(s), t) for s, t in zip(batch, ko)]
        progress(5 + (50 if not a.no_polish else 95) * hi / len(segs))

    if not a.no_polish:
        polished = 0
        for i in range(0, len(out), POLISH_BATCH):
            chunk = out[i:i + POLISH_BATCH]
            src = segs[i:i + POLISH_BATCH]
            t0, t1 = chunk[0]["start"], chunk[-1]["end"]
            notes = [n["desc"][:140] for n in [s for s in scenes if t0 - 10 <= s["t"] <= t1 + 10][::3][:6]]
            pairs = "\n".join("%d. %s%s => %s" % (k + 1, tag(s), s["text"], c["text"]) for k, (s, c) in enumerate(zip(src, chunk)))
            try:
                got = parse_numbered(chat(POLISH.format(bible=bible, scenes="\n".join("- " + x for x in notes) or "(none)",
                                                        pairs=pairs, n=len(chunk))), len(chunk))
            except Exception as e:
                log("polish %d failed: %s" % (i // POLISH_BATCH, e))
                warn("감수 배치 %d 실패: 초벌 번역 유지" % (i // POLISH_BATCH))
                continue
            for k, (s, c) in enumerate(zip(src, chunk)):
                t = clean(got.get(k + 1, ""))
                if c["text"] == "-":
                    continue
                if t and not bad(s["text"], t) and len(t) <= 3 * max(8, len(c["text"])):
                    if t != c["text"]:
                        polished += 1
                    c["text"] = t
            progress(55 + 44 * (i + len(chunk)) / len(out))
        log("polish: %d line(s) changed" % polished)

    try:
        ollama_unload(a.ollama, a.model)  # free VRAM for the next job's Whisper run
    except Exception as e:
        log("unload failed: %s" % e)
    noise = sum(1 for c in out if c["text"] == "-")
    if noise:
        log("%d noise line(s) dropped by the translator" % noise)
    out = [c for c in out if c["text"] != "-"]
    out = composite_captions(out)
    write_srt(os.path.join(a.out_dir, "ko.srt"), out)
    write_vtt(os.path.join(a.out_dir, "ko.vtt"), out)
    sdh = build_sdh(out, sounds, castd)
    write_srt(os.path.join(a.out_dir, "ko.sdh.srt"), sdh)
    write_vtt(os.path.join(a.out_dir, "ko.sdh.vtt"), sdh)
    progress(100)


SOUND_KO = {"moan": "[신음]", "laugh": "[웃음]"}
SPK_KO = {"F": "여", "M": "남"}


def composite_captions(cues):
    """Captions shown at the same time become one cue with one line per caption, top to bottom as on the picture.
    A title that appears line by line therefore grows downward (line 1, then lines 1-2, then 1-3) instead of each
    new line being stacked above the previous one by the player."""
    caps = [c for c in cues if c.get("pos") == "top"]
    if not caps:
        return cues
    rest = [c for c in cues if c.get("pos") != "top"]
    edges = sorted({t for c in caps for t in (c["start"], c["end"])})
    merged = []
    for t0, t1 in zip(edges, edges[1:]):
        active = [c for c in caps if c["start"] <= t0 and c["end"] >= t1]
        if not active or t1 - t0 < 0.25:
            continue
        text = "\n".join(c["text"] for c in sorted(active, key=lambda c: (c.get("y", 50), c["start"])))
        if merged and merged[-1]["text"] == text and abs(merged[-1]["end"] - t0) < 0.01:
            merged[-1]["end"] = t1
        else:
            merged.append({"start": t0, "end": t1, "text": text, "spk": "", "pos": "top"})
    return sorted(rest + merged, key=lambda c: (c["start"], c.get("pos") == "top"))


def build_sdh(cues, sounds, cast=None):
    """SDH track: a speaker label when the speaker changes (여/남 by voice, 화자N for an unclear diarized voice),
    and sound cues where nobody speaks."""
    def label(spk):
        if (cast or {}).get(spk, {}).get("narration"):
            return "내레이션"
        if spk.startswith("S"):
            v = (cast or {}).get(spk, {}).get("voice", "")
            return ("여" if v == "female-sounding" else "남" if v == "male-sounding" else "화자") + spk[1:]
        if spk[:1] in SPK_KO:
            return SPK_KO[spk[:1]] + spk[1:]
        return None

    out, last = [], None
    for c in cues:
        t = c["text"]
        lab = label(c.get("spk") or "")
        if lab and lab != last and not c.get("pos"):
            t = "%s: %s" % (lab, t)
        if lab:
            last = lab
        out.append(dict(c, text=t))
    for snd in sounds:
        if snd["kind"] not in SOUND_KO:
            continue
        if any(min(c["end"], snd["end"]) - max(c["start"], snd["start"]) > 0 for c in cues):
            continue
        out.append({"start": snd["start"], "end": snd["end"], "text": SOUND_KO[snd["kind"]]})
    out.sort(key=lambda c: c["start"])
    return out


if __name__ == "__main__":
    main()

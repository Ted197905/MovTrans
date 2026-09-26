"""Shared helpers for the bin/*.py workers: progress protocol, SRT/VTT writers, Ollama client."""
import json
import sys
import urllib.error
import urllib.request


def progress(pct):
    sys.stderr.write("progress %d\n" % max(0, min(100, int(pct))))
    sys.stderr.flush()


def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _ts(t, sep):
    t = max(0.0, float(t))
    h = int(t // 3600); m = int(t % 3600 // 60); s = int(t % 60); ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        s += 1; ms = 0
    return "%02d:%02d:%02d%s%03d" % (h, m, s, sep, ms)


def write_srt(path, segs):
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segs, 1):
            f.write("%d\n%s --> %s\n%s\n\n" % (i, _ts(s["start"], ","), _ts(s["end"], ","), s["text"].strip()))


def write_vtt(path, segs):
    """Cues with "pos": "top" are placed near the top of the picture (on-screen captions)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for s in segs:
            setting = " line:8% align:center" if s.get("pos") == "top" else ""
            f.write("%s --> %s%s\n%s\n\n" % (_ts(s["start"], "."), _ts(s["end"], "."), setting, s["text"].strip()))


_NO_THINK = {}  # model -> False when the model rejects the "think" option


def ollama_chat(url, model, messages, images=None, keep_alive="10m", timeout=1800, num_ctx=None, think=False, temperature=0.2):
    """One /api/chat round trip. images: list of base64 strings attached to the last user message.
    think=False switches reasoning off for thinking models (Qwen3.5/3.6); instruct models ignore or reject it."""
    msgs = [dict(m) for m in messages]
    if images:
        msgs[-1]["images"] = images
    body = {"model": model, "messages": msgs, "stream": False, "keep_alive": keep_alive,
            "options": {"temperature": temperature}}
    if num_ctx:
        body["options"]["num_ctx"] = num_ctx
    if _NO_THINK.get(model, True) and think is not None:
        body["think"] = think
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace")
        if e.code == 400 and "think" in msg and "think" in body:
            _NO_THINK[model] = False
            return ollama_chat(url, model, messages, images, keep_alive, timeout, num_ctx, None, temperature)
        raise urllib.error.HTTPError(e.url, e.code, msg[:300], e.headers, None)
    return data["message"]["content"]


def ollama_unload(url, model):
    body = {"model": model, "keep_alive": 0}
    req = urllib.request.Request(url.rstrip("/") + "/api/generate", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        r.read()


def drop_persistent_text(cues, duration):
    """On-screen text that keeps coming back over a large part of the video is a title/watermark overlay,
    not a caption: the vision model reads it in every frame it is asked about."""
    by = {}
    for c in cues:
        by.setdefault("".join(c["text"].split()), []).append(c)
    out = []
    for text, cs in by.items():
        span = max(c["end"] for c in cs) - min(c["start"] for c in cs)
        if len(cs) >= 6 and span > 0.2 * max(duration, 1):
            continue
        out.extend(cs)
    out.sort(key=lambda c: (c["start"], c["text"]))
    return out


def strip_think(text):
    """Qwen3 may emit <think>...</think>; drop it."""
    while "<think>" in text and "</think>" in text:
        a = text.index("<think>"); b = text.index("</think>") + len("</think>")
        text = text[:a] + text[b:]
    return text.strip()


def extract_json(text):
    """Find the first JSON object/array in a model reply."""
    text = strip_think(text)
    if "```" in text:
        parts = text.split("```")
        for p in parts[1::2]:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            try:
                return json.loads(p)
            except ValueError:
                continue
    for open_c, close_c in (("{", "}"), ("[", "]")):
        a = text.find(open_c); b = text.rfind(close_c)
        if a != -1 and b > a:
            try:
                return json.loads(text[a:b + 1])
            except ValueError:
                pass
    raise ValueError("no JSON in model reply: " + text[:200])

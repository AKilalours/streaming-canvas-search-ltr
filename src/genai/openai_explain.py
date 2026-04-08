"""
StreamLens — Production explanation engine
Phenomenal explanations. Specific to every film. Matched to real taste.
Retry on 429. Redis cache so each film calls OpenAI once ever.
"""
from __future__ import annotations
import os, json, urllib.request, urllib.error, re, base64, time, hashlib

_URL = "https://api.openai.com/v1/chat/completions"

# ── Cache (Redis + in-process fallback) ──────────────────────────────────────
_MEM: dict[str, str] = {}

def _ck(*parts: str) -> str:
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]

def _cget(key: str) -> str | None:
    try:
        import redis as _r
        v = _r.Redis.from_url(os.environ.get("REDIS_URL","redis://localhost:6379"),
                              socket_timeout=1).get(f"exp:{key}")
        if v: return v.decode()
    except Exception: pass
    return _MEM.get(key)

def _cset(key: str, val: str, ttl: int = 604800) -> None:   # 7-day default
    _MEM[key] = val
    try:
        import redis as _r
        _r.Redis.from_url(os.environ.get("REDIS_URL","redis://localhost:6379"),
                          socket_timeout=1).setex(f"exp:{key}", ttl, val)
    except Exception: pass


# ── Core OpenAI call with retry ───────────────────────────────────────────────
def _call(model: str, messages: list[dict], max_tokens: int = 200) -> str:
    key = os.environ.get("OPENAI_API_KEY","").strip()
    if not key.startswith("sk-"): return ""
    payload = json.dumps({
        "model": model, "max_tokens": max_tokens,
        "temperature": 0.45, "messages": messages,
    }).encode()
    for attempt in range(4):
        req = urllib.request.Request(_URL, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        })
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep((2 ** attempt) * 1.5)   # 1.5 → 3 → 6 → 12s
                continue
            return ""
        except Exception:
            return ""
    return ""


# ── Parse corpus text field ───────────────────────────────────────────────────
def _parse(text: str) -> tuple[str, str, str, str]:
    tm = re.search(r'Title:\s*([^|]+)', text)
    gm = re.search(r'Genres?:\s*([^|]+)', text)
    km = re.search(r'Tags?:\s*([^|\n]+)', text)
    ym = re.search(r'\((\d{4})\)', tm.group(1) if tm else text)
    title = re.sub(r'\s*\(\d{4}\)\s*$', '',
                   tm.group(1).strip() if tm else text[:60]).strip()
    return (
        title,
        ym.group(1) if ym else "",
        gm.group(1).strip() if gm else "",
        km.group(1).strip() if km else "",
    )


def _tags(raw: str, n: int = 4) -> str:
    skip = {"film","movie","good","great","nice","classic","popular",
            "cult film","drugs","based on","watch","seen","known","famous"}
    picks = [x.strip() for x in raw.split(",")
             if x.strip().lower() not in skip and len(x.strip()) > 2]
    return ", ".join(picks[:n]) or (raw.split(",")[0].strip() if raw else "its story")


def _voice(profile: str, prefs: str) -> tuple[str, str, str]:
    db = {
        "chrisen": (
            "Chrisen",
            "high-energy action, psychological thrillers, crime dramas, mind-bending sci-fi, dark comedy",
            "sharp and direct — like a film-obsessed friend"
        ),
        "gilbert": (
            "Gilbert",
            "feel-good romance, heartwarming comedy, family animation, coming-of-age stories",
            "warm and enthusiastic — like someone who loves great storytelling"
        ),
        "alex": (
            "Alex",
            "adventure, animation, children's films, fantasy",
            "fun and energetic — like a movie night recommendation"
        ),
    }
    p = db.get(profile.lower())
    if p: return p
    return profile.capitalize(), prefs or "great cinema", "warm and personal"


# ── WHY THIS ─────────────────────────────────────────────────────────────────
def explain_why_this(title: str, text: str, profile: str,
                     profile_prefs: str, language: str = "English") -> str:
    """
    Exactly 2 sentences. Fast. Personal. Punchy.
    Sentence 1: one sharp hook about THIS film's best quality.
    Sentence 2: one direct connection to THIS person's taste.
    Cached per film + profile + language.
    """
    ck = _ck("why", title[:40], profile, language)
    hit = _cget(ck)
    if hit: return hit

    t, yr, genres, raw_tags = _parse(text)
    gl   = [g.strip() for g in genres.split(",") if g.strip()]
    g1   = gl[0] if gl else "Drama"
    gstr = ", ".join(gl[:2]) if gl else "Drama"
    btag = _tags(raw_tags, 3)
    year = f" ({yr})" if yr else ""
    name, taste, tone = _voice(profile, profile_prefs)
    taste1 = taste.split(",")[0].strip()
    taste2 = taste.split(",")[1].strip() if "," in taste else taste1

    system = (
        f"Movie recommender. {language} only.\n"
        f"OUTPUT: 2 sentences. 25-32 words TOTAL. Hard limit.\n"
        f"S1: One SPECIFIC thing about this film — a scene, a twist, "
        f"what it actually does differently. NOT 'clever', 'delightful', 'charming'.\n"
        f"S2: One sharp reason {name} who loves {taste1} will love it. "
        f"End strong — never end with 'throughout', 'as well', 'too', 'also', 'you enjoy'.\n"
        f"BANNED WORDS: heartwarming, vibrant, nostalgia, delightful, charming, "
        f"rollercoaster, journey, throughout, brimming, whimsical.\n"
        f"Write like a film-obsessed friend texting you. Specific. Fast. Done."
    )
    user = (
        f"{t}{year} | {gstr} | {btag}\n"
        f"{name} loves {taste1}, {taste2}\n"
        f"2 sentences. {language}. Max 32 words total."
    )

    result = _call("gpt-4o-mini", [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ], max_tokens=90)

    if not result:
        result = _fallback(t, year, g1, gstr, btag, name, taste, language)

    _cset(ck, result)
    return result


# ── RAG DEEP EXPLANATION ──────────────────────────────────────────────────────
def explain_rag(title: str, text: str, profile: str,
                profile_prefs: str, language: str = "English",
                similar_titles: list | None = None) -> str:
    """
    3-line structured breakdown. Each line: SHORT and punchy.
    Why → About → Also Enjoy. Total: ~50 words max.
    Fast to read. Dense with value. No waffle.
    """
    ck = _ck("rag", title[:40], profile, language)
    hit = _cget(ck)
    if hit: return hit

    t, yr, genres, raw_tags = _parse(text)
    gl    = [g.strip() for g in genres.split(",") if g.strip()]
    gstr  = ", ".join(gl[:2]) if gl else "Drama"
    btags = _tags(raw_tags, 4)
    year  = f" ({yr})" if yr else ""
    name, taste, tone = _voice(profile, profile_prefs)
    taste1 = taste.split(",")[0].strip()
    taste2 = taste.split(",")[1].strip() if "," in taste else taste1
    sim    = ", ".join(similar_titles[:2]) if similar_titles else ""

    LABELS: dict[str, tuple[str,str,str]] = {
        "English":    ("⚡ WHY YOU",    "🎬 ABOUT",    "🎥 ALSO TRY"),
        "Arabic":     ("⚡ لماذا",       "🎬 القصة",    "🎥 شاهد أيضاً"),
        "French":     ("⚡ POURQUOI",   "🎬 L'HISTOIRE","🎥 À VOIR AUSSI"),
        "Spanish":    ("⚡ POR QUÉ",    "🎬 DE QUÉ VA", "🎥 TAMBIÉN VE"),
        "Hindi":      ("⚡ क्यों",       "🎬 कहानी",    "🎥 यह भी देखो"),
        "Japanese":   ("⚡ なぜ",        "🎬 あらすじ",  "🎥 次はこれ"),
        "Korean":     ("⚡ 이유",        "🎬 줄거리",    "🎥 이것도"),
        "German":     ("⚡ WARUM",      "🎬 WORUM",    "🎥 AUCH ANSEHEN"),
        "Portuguese": ("⚡ POR QUÊ",    "🎬 SOBRE",    "🎥 VEJA TAMBÉM"),
        "Italian":    ("⚡ PERCHÉ",     "🎬 DI COSA",  "🎥 GUARDA ANCHE"),
        "Russian":    ("⚡ ПОЧЕМУ",     "🎬 О ЧЁМ",    "🎥 ЕЩЁ ПОСМОТРИ"),
        "Chinese":    ("⚡ 为什么",      "🎬 剧情",     "🎥 也可以看"),
        "Turkish":    ("⚡ NEDEN",      "🎬 KONU",     "🎥 BUNLARI DA İZLE"),
        "Tamil":      ("⚡ ஏன்",        "🎬 கதை",      "🎥 இதுவும் பாருங்கள்"),
        "Telugu":     ("⚡ ఎందుకు",     "🎬 కథ",       "🎥 ఇవీ చూడండి"),
        "Malayalam":  ("⚡ എന്തുകൊണ്ട്", "🎬 കഥ",      "🎥 ഇതും കാണൂ"),
    }
    L = LABELS.get(language, LABELS["English"])

    system = (
        f"You write ultra-concise movie recommendations. "
        f"Write ONLY in {language}. Zero English if not English.\n\n"
        f"Output EXACTLY 3 lines — nothing else:\n"
        f"{L[0]}: [10-12 words MAX — one razor-sharp reason {name} will love it based on their taste]\n"
        f"{L[1]}: [10-12 words MAX — what the film is actually about, one specific plot hook]\n"
        f"{L[2]}: [2-3 real film titles, comma separated, similar vibe]\n\n"
        f"Rules:\n"
        f"- Every line must be specific to THIS exact film\n"
        f"- {L[0]}: connect to {name}'s love of {taste1} and {taste2}\n"
        f"- {L[1]}: one real plot hook, no spoilers, no 'a great film'\n"
        f"- {L[2]}: real film titles only — no made-up names\n"
        f"- BANNED WORDS: nostalgia, brimming, whimsical, heartwarming, delightful, charming\n"
        f"- SHORT. PUNCHY. SPECIFIC. Like a smart friend's WhatsApp message."
    )
    user = (
        f"Film: {t}{year} | {gstr} | {btags}\n"
        f"For: {name} who loves {taste1}, {taste2}\n"
        + (f"Liked: {sim}\n" if sim else "") +
        f"\n3 lines in {language}. Max 12 words each."
    )

    result = _call("gpt-4o-mini", [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ], max_tokens=140)

    if not result:
        result = explain_why_this(title, text, profile, profile_prefs, language)

    _cset(ck, result)
    return result


# ── VLM POSTER DESCRIPTION ────────────────────────────────────────────────────
def describe_poster_gpt4v(poster_url: str, title: str,
                           language: str = "English") -> str:
    """GPT-4o vision: vivid 1-2 sentence poster description. Cached 30 days."""
    if not poster_url: return ""

    ck = _ck("vlm", poster_url[-20:], language)
    hit = _cget(ck)
    if hit: return hit

    key = os.environ.get("OPENAI_API_KEY","").strip()
    if not key.startswith("sk-"): return ""

    try:
        req = urllib.request.Request(poster_url,
            headers={"User-Agent": "StreamLens/1.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            img_bytes = r.read()
            ct = r.headers.get("Content-Type","image/jpeg").split(";")[0]
        b64 = base64.b64encode(img_bytes).decode()
        data_url = f"data:{ct};base64,{b64}"
    except Exception:
        return ""

    payload = json.dumps({
        "model": "gpt-4o",
        "max_tokens": 120, "temperature": 0.3,
        "messages": [{"role":"user","content":[
            {"type":"image_url",
             "image_url":{"url":data_url,"detail":"low"}},
            {"type":"text","text":(
                f"Write in {language} only. 1-2 sentences. "
                f"Describe what you actually SEE in this {title} movie poster: "
                f"the dominant colors, the mood it creates, any faces or figures visible, "
                f"the atmosphere and feeling it evokes. "
                f"Be vivid and specific. Start directly — no 'the poster shows'."
            )},
        ]}],
    }).encode()

    req2 = urllib.request.Request(_URL, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    })
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req2, timeout=20) as r:
                result = json.loads(r.read())["choices"][0]["message"]["content"].strip()
                _cset(ck, result, ttl=86400*30)
                return result
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep((2**attempt)*2)
                continue
            return ""
        except Exception:
            return ""
    return ""


# ── TRANSLATE ─────────────────────────────────────────────────────────────────
def translate_clean(text: str, language: str) -> str:
    if language == "English" or not text.strip(): return text
    ck = _ck("tr", text[:60], language)
    hit = _cget(ck)
    if hit: return hit
    result = _call("gpt-4o-mini", [
        {"role":"system","content":
         f"Translate to {language}. Fluent natural {language} only. Return translation only."},
        {"role":"user","content":text},
    ], max_tokens=200) or text
    _cset(ck, result)
    return result


# ── SMART FALLBACK ────────────────────────────────────────────────────────────
def _fallback(title: str, year: str, g1: str, gstr: str,
              tags: str, name: str, taste: str, language: str) -> str:
    """Real-sounding fallback using actual film data. No more generic garbage."""
    tag1 = tags.split(",")[0].strip() if tags else g1
    tag2 = tags.split(",")[1].strip() if "," in tags else ""
    taste1 = taste.split(",")[0].strip()

    tpl = {
        "English": (
            f"{title}{year} is a compelling {gstr} film that stands out for its {tag1}"
            + (f" and {tag2}" if tag2 else "")
            + f". If {name} is drawn to {taste1}, this one delivers exactly that."
        ),
        "Arabic": (
            f"{title}{year} فيلم {g1} رائع يتميز بـ{tag1}. "
            f"إذا أحب {name} {taste1}، فهذا الفيلم سيكون مثالياً."
        ),
        "French": (
            f"{title}{year} est un {g1} captivant, remarquable pour {tag1}. "
            f"Si {name} aime {taste1}, ce film est exactement ce qu'il faut."
        ),
        "Spanish": (
            f"{title}{year} es un {g1} notable por {tag1}. "
            f"Si {name} disfruta de {taste1}, esta película lo va a encantar."
        ),
        "Hindi": (
            f"{title}{year} एक बेहतरीन {g1} फ़िल्म है जो {tag1} के लिए जानी जाती है। "
            f"{name} को {taste1} पसंद है, तो यह बिल्कुल सही है।"
        ),
        "Japanese": (
            f"{title}{year}は{g1}映画で、{tag1}が際立つ作品です。"
            f"{name}が{taste1}を好むなら、きっと気に入るでしょう。"
        ),
        "Korean": (
            f"{title}{year}은 {g1} 장르 영화로, {tag1}이 두드러집니다. "
            f"{name}이 {taste1}을 좋아한다면 이 영화는 딱입니다."
        ),
        "Tamil": (
            f"{title}{year} ஒரு சிறந்த {g1} திரைப்படம், {tag1} காரணமாக தனித்து நிற்கிறது. "
            f"{name} {taste1} விரும்பினால், இது சரியான தேர்வு."
        ),
        "Telugu": (
            f"{title}{year} అద్భుతమైన {g1} చిత్రం, {tag1} కోసం ప్రత్యేకంగా నిలుస్తుంది. "
            f"{name} {taste1} ఇష్టపడితే, ఇది తప్పక చూడండి."
        ),
    }
    return tpl.get(language, tpl["English"])


# ── DEBUG ─────────────────────────────────────────────────────────────────────
def _call_openai(messages: list[dict], temperature: float = 0.35,
                 max_tokens: int = 100) -> str:
    """Direct call — used by /debug/openai_test endpoint."""
    return _call("gpt-4o-mini", messages, max_tokens)

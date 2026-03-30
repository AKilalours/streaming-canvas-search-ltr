"""
StreamLens — Production explanation engine
Natural language. Specific to the movie. Matched to user taste.
"""
from __future__ import annotations
import os, json, urllib.request, re, base64

_URL = "https://api.openai.com/v1/chat/completions"


def _call(model: str, messages: list[dict], max_tokens: int = 120) -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key.startswith("sk-"): return ""
    payload = json.dumps({
        "model": model, "max_tokens": max_tokens,
        "temperature": 0.35, "messages": messages,
    }).encode()
    req = urllib.request.Request(_URL, data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


def _parse(text: str) -> tuple[str, str, str, str]:
    title_m = re.search(r'Title:\s*([^|]+)', text)
    genre_m = re.search(r'Genres?:\s*([^|]+)', text)
    tags_m  = re.search(r'Tags?:\s*([^|\n]+)', text)
    year_m  = re.search(r'\((\d{4})\)', title_m.group(1) if title_m else text)
    title   = re.sub(r'\s*\(\d{4}\)\s*$', '',
                     title_m.group(1).strip() if title_m else text[:50]).strip()
    return (
        title,
        year_m.group(1) if year_m else "",
        genre_m.group(1).strip() if genre_m else "",
        tags_m.group(1).strip() if tags_m else "",
    )


def _best_tags(tags: str, n: int = 3) -> str:
    """Pick most interesting tags — skip generic words."""
    skip = {"film","movie","good","great","nice","cult film","drugs"}
    picks = [x.strip() for x in tags.split(",")
             if x.strip().lower() not in skip and len(x.strip()) > 2]
    return ", ".join(picks[:n]) if picks else tags.split(",")[0].strip()


def explain_why_this(title: str, text: str, profile: str,
                     profile_prefs: str, language: str = "English") -> str:
    """
    2 sentences max. Sounds like a friend recommending a movie.
    Specific to this film's actual qualities. Matched to user taste.
    """
    t, year, genres, tags = _parse(text)
    genre1   = genres.split(",")[0].strip() if genres else "Drama"
    best_tag = _best_tags(tags)
    pref1    = profile_prefs.split(",")[0].strip() if profile_prefs else genre1
    yr       = f" ({year})" if year else ""

    return _call("gpt-4o-mini", [
        {"role": "system", "content": (
            f"You are a movie-loving friend recommending a film. "
            f"Write ONLY in {language}. "
            f"Write 2 sentences maximum. "
            f"First sentence: what makes THIS specific film great (mention its actual qualities). "
            f"Second sentence: why this person who loves {pref1} will connect with it. "
            f"Sound warm and natural. No bullet points. No translation. No 'this film'."
        )},
        {"role": "user", "content": (
            f"Film: {t}{yr}\n"
            f"Genre: {genre1}\n"
            f"What it's known for: {best_tag}\n"
            f"Person loves: {pref1}\n\n"
            f"Recommend this in {language} — 2 sentences, warm and specific."
        )},
    ], max_tokens=130) or _fallback(genre1, best_tag, pref1, language)


def explain_rag(title: str, text: str, profile: str,
                profile_prefs: str, language: str = "English",
                similar_titles: list | None = None) -> str:
    """
    3 lines. WHY YOU'LL LOVE IT / WHAT IT'S ABOUT / YOU'LL ALSO ENJOY
    Specific film names. Real qualities. Natural language.
    """
    t, year, genres, tags = _parse(text)
    genre1    = genres.split(",")[0].strip() if genres else "Drama"
    best_tags = _best_tags(tags, 3)
    pref_top  = ", ".join(p.strip() for p in profile_prefs.split(",")[:2]) \
                if profile_prefs else genre1
    yr        = f" ({year})" if year else ""
    sim_str   = ", ".join(similar_titles[:2]) if similar_titles else ""

    return _call("gpt-4o-mini", [
        {"role": "system", "content": (
            f"You are a movie expert writing short, personal recommendations. "
            f"Write ONLY in {language}. "
            f"Write exactly 3 lines with these labels (translate labels too): "
            f"WHY YOU'LL LOVE IT: / WHAT IT'S ABOUT: / YOU'LL ALSO ENJOY: "
            f"Rules: "
            f"- Each line: 1 sentence, max 18 words "
            f"- Mention specific things about THIS film (not generic) "
            f"- YOU'LL ALSO ENJOY must name specific film titles "
            f"- Sound like a friend, not a review site "
            f"- Pure {language} — no English, no translation note"
        )},
        {"role": "user", "content": (
            f"Film: {t}{yr}\n"
            f"Genre: {genre1} | Highlights: {best_tags}\n"
            f"Person enjoys: {pref_top}\n"
            + (f"They previously liked: {sim_str}\n" if sim_str else "") +
            f"\nWrite 3 lines in {language}."
        )},
    ], max_tokens=180) or explain_why_this(title, text, profile,
                                           profile_prefs, language)


def describe_poster_gpt4v(poster_url: str, title: str,
                           language: str = "English") -> str:
    """GPT-4o vision: 1 sentence. What the poster feels like."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key.startswith("sk-") or not poster_url: return ""
    try:
        req = urllib.request.Request(poster_url,
            headers={"User-Agent": "StreamLens/1.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            img_bytes = r.read()
            ct = r.headers.get("Content-Type", "image/jpeg").split(";")[0]
        b64 = base64.b64encode(img_bytes).decode()
        data_url = f"data:{ct};base64,{b64}"
    except Exception:
        return ""

    payload = json.dumps({
        "model": "gpt-4o",
        "max_tokens": 60, "temperature": 0.2,
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": data_url, "detail": "low"}},
            {"type": "text", "text": (
                f"In {language} only, 1 sentence (max 15 words): "
                f"what feeling does this {title} movie poster give you? "
                f"Describe what you actually see — colors, mood, atmosphere."
            )},
        ]}],
    }).encode()
    req2 = urllib.request.Request(_URL, data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req2, timeout=15) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


def translate_clean(text: str, language: str) -> str:
    if language == "English": return text
    return _call("gpt-4o-mini", [
        {"role": "system", "content":
            f"Translate to {language}. Natural language only. Return translation only."},
        {"role": "user", "content": text},
    ], max_tokens=150) or text


_FB = {
    "English":    "If you love {pref}, {genre} film known for {tag} will be your kind of night.",
    "Arabic":     "إذا أحببت {pref}، فإن هذا الفيلم المعروف بـ {tag} سيكون مثاليًا لك.",
    "French":     "Si tu aimes {pref}, ce film connu pour {tag} est exactement ce qu'il te faut.",
    "Spanish":    "Si te gusta {pref}, esta película conocida por {tag} es perfecta para ti.",
    "Hindi":      "अगर तुम्हें {pref} पसंद है, तो {tag} के लिए मशहूर यह फ़िल्म तुम्हें पसंद आएगी।",
    "Japanese":   "{pref}が好きなら、{tag}で有名なこの映画はきっと気に入るよ。",
    "German":     "Wenn du {pref} liebst, ist dieser Film bekannt für {tag} genau richtig für dich.",
    "Portuguese": "Se você curte {pref}, esse filme famoso por {tag} é perfeito pra você.",
    "Korean":     "{pref}을 좋아한다면, {tag}으로 유명한 이 영화가 딱 맞을 거예요.",
    "Italian":    "Se ami {pref}, questo film noto per {tag} farà al caso tuo.",
    "Chinese":    "如果你喜欢{pref}，这部以{tag}著称的电影绝对是你的菜。",
    "Russian":    "Если вы любите {pref}, этот фильм, известный {tag}, точно вам понравится.",
    "Turkish":    "{pref} seviyorsanız, {tag} ile bilinen bu film tam size göre.",
    "Dutch":      "Als je van {pref} houdt, is deze film bekend om {tag} precies wat je zoekt.",
    "Swedish":    "Om du gillar {pref} är den här filmen känd för {tag} perfekt för dig.",
    "Polish":     "Jeśli lubisz {pref}, ten film znany z {tag} jest właśnie dla ciebie.",
    "Thai":       "ถ้าชอบ {pref} หนังที่รู้จักจาก {tag} เรื่องนี้เหมาะกับคุณมาก",
}

def _fallback(genre, tag, pref, language):
    tpl = _FB.get(language, _FB["English"])
    return tpl.format(
        genre=genre.split(",")[0].strip() or "Drama",
        tag=tag.split(",")[0].strip() if tag else "its story",
        pref=pref.split(",")[0].strip() if pref else "good films"
    )

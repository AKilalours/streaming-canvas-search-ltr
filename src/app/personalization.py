from __future__ import annotations
import json
from pathlib import Path
from typing import Any

_USERS = None

def load_users(path: str = "data/users/users.json") -> dict[str, Any]:
    global _USERS
    if _USERS is None:
        p = Path(path)
        _USERS = json.loads(p.read_text()) if p.exists() else {}
    return _USERS

def keyword_overlap_score(user_id: str | None, text: str, users: dict[str, Any]) -> float:
    if not user_id or user_id not in users:
        return 0.0
    kws = users[user_id].get("keywords", [])
    if not kws:
        return 0.0
    t = text.lower()
    hits = sum(1 for kw in kws if kw.lower() in t)
    return hits / max(1, len(kws))

def explain_payload(query: str, doc_id: str, base_score: float, overlap: float, final_score: float) -> dict[str, Any]:
    return {
        "query": query,
        "doc_id": doc_id,
        "signals": {
            "base_score": base_score,
            "user_keyword_overlap": overlap,
        },
        "final_score": final_score,
        "reason": "Boosted by user keyword overlap" if overlap > 0 else "No personalization boost applied",
    }

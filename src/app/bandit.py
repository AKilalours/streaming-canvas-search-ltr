"""
StreamLens — Thompson Sampling Bandit
======================================
Principled exploration strategy used by Netflix, Spotify, Booking.com.

Why Thompson Sampling beats ε-greedy:
  ε-greedy: fixed 15% random exploration for ALL users
  Thompson:  adapts per user — new users explore more, known users exploit

How it works:
  Each item maintains Beta(α, β) distribution:
    α = number of positive interactions (clicks, watches)
    β = number of negative interactions (skips, no-click)
  
  At serving time: sample from Beta(α, β) for each item
  Items with high uncertainty get higher samples → exploration
  Items with proven performance get consistently high samples → exploitation
  
  As more data arrives: Beta distribution narrows → less exploration needed
"""
from __future__ import annotations
import numpy as np
import json
import os
from pathlib import Path

# Storage for Beta distribution parameters
# α (alpha) = successes, β (beta) = failures
_ALPHA: dict[str, dict[str, float]] = {}  # {user_id: {item_id: alpha}}
_BETA:  dict[str, dict[str, float]] = {}  # {user_id: {item_id: beta}}

# Persistence path
_STATE_PATH = "artifacts/bandit/thompson_state.json"
os.makedirs("artifacts/bandit", exist_ok=True)

def _load_state():
    """Load persisted bandit state."""
    global _ALPHA, _BETA
    if os.path.exists(_STATE_PATH):
        state = json.load(open(_STATE_PATH))
        _ALPHA = state.get("alpha", {})
        _BETA  = state.get("beta", {})

def _save_state():
    """Persist bandit state."""
    json.dump({"alpha": _ALPHA, "beta": _BETA}, open(_STATE_PATH, "w"))

def thompson_sample(user_id: str, items: list[dict]) -> list[dict]:
    """
    Thompson Sampling reranking.
    
    For each item, sample from its Beta distribution.
    Items with high uncertainty (few interactions) get
    higher variance → more likely to be explored.
    
    Args:
        user_id: User identifier
        items:   List of items with 'doc_id' and 'score' fields
    
    Returns:
        Items reranked by Thompson sample * original score
    """
    if user_id not in _ALPHA:
        _ALPHA[user_id] = {}
        _BETA[user_id]  = {}

    user_alpha = _ALPHA[user_id]
    user_beta  = _BETA[user_id]

    reranked = []
    for item in items:
        doc_id = item.get("doc_id", "")
        
        # Get Beta parameters (default: α=1, β=1 = uniform prior)
        a = user_alpha.get(doc_id, 1.0)
        b = user_beta.get(doc_id, 1.0)
        
        # Sample from Beta distribution
        thompson_score = float(np.random.beta(a, b))
        
        # Combine with LTR score
        # LTR score = relevance quality
        # Thompson score = exploration bonus
        ltr_score = item.get("score", 0.5)
        combined  = 0.85 * ltr_score + 0.15 * thompson_score
        
        reranked.append({**item, "thompson_score": thompson_score,
                         "combined_score": combined})

    # Sort by combined score
    reranked.sort(key=lambda x: x["combined_score"], reverse=True)
    return reranked


def record_interaction(user_id: str, doc_id: str, 
                        event_type: str, watch_pct: float = 0.0):
    """
    Update Beta distribution based on user interaction.
    
    Positive signals (α++): click, watch_complete, high watch_pct
    Negative signals (β++): skip, low watch_pct
    
    Args:
        user_id:    User identifier
        doc_id:     Document/item identifier
        event_type: 'click', 'watch_complete', 'skip', 'impression'
        watch_pct:  Fraction watched (0.0-1.0)
    """
    if user_id not in _ALPHA:
        _ALPHA[user_id] = {}
        _BETA[user_id]  = {}

    # Initialize if first time
    if doc_id not in _ALPHA[user_id]:
        _ALPHA[user_id][doc_id] = 1.0
        _BETA[user_id][doc_id]  = 1.0

    # Update based on signal type
    if event_type == "watch_complete" or watch_pct > 0.8:
        _ALPHA[user_id][doc_id] += 2.0   # Strong positive
    elif event_type == "click" or watch_pct > 0.3:
        _ALPHA[user_id][doc_id] += 1.0   # Weak positive
    elif event_type == "skip" or watch_pct < 0.1:
        _BETA[user_id][doc_id]  += 1.0   # Negative
    # "impression" with no interaction: slight negative
    elif event_type == "impression":
        _BETA[user_id][doc_id]  += 0.2


def get_exploration_rate(user_id: str) -> float:
    """
    Returns effective exploration rate for a user.
    New users: ~0.5 (high exploration)
    Known users: ~0.05-0.15 (low exploration)
    """
    if user_id not in _ALPHA:
        return 0.5  # Cold start: maximum exploration

    total_interactions = sum(
        _ALPHA[user_id].get(did, 1) + _BETA[user_id].get(did, 1) - 2
        for did in _ALPHA[user_id]
    )

    # Exploration decreases as interactions accumulate
    # Formula: ε = 0.5 / (1 + total_interactions / 50)
    return min(0.5, 0.5 / (1 + total_interactions / 50))


# Load state on import
_load_state()

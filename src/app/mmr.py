"""MMR Diversity Reranking for StreamLens"""
import numpy as np


def mmr_rerank(items: list[dict], embeddings: np.ndarray,
               lam: float = 0.7, k: int = 10) -> list[dict]:
    """
    Maximal Marginal Relevance reranking.

    lam=1.0: pure relevance (standard ranking)
    lam=0.0: pure diversity
    lam=0.7: 70% relevance + 30% diversity (recommended)

    Used by: Google Search, Spotify Discover Weekly,
             Netflix shelf diversity algorithm
    """
    if len(items) <= k:
        return items

    scores = np.array([it.get("score", 0.0) for it in items])
    # Normalize scores to [0,1]
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    selected_idx = []
    remaining = list(range(len(items)))

    # Always pick top-1 first
    best = int(np.argmax(scores))
    selected_idx.append(best)
    remaining.remove(best)

    # Iteratively pick items that maximize MMR
    while len(selected_idx) < k and remaining:
        mmr_scores = []
        sel_embs = embeddings[selected_idx]

        for i in remaining:
            # Relevance component
            rel = lam * scores[i]

            # Diversity component: max similarity to already selected
            if len(sel_embs) > 0:
                sims = embeddings[i] @ sel_embs.T
                max_sim = float(sims.max())
            else:
                max_sim = 0.0

            div = (1 - lam) * max_sim
            mmr_scores.append(rel - div)

        best_remaining = remaining[int(np.argmax(mmr_scores))]
        selected_idx.append(best_remaining)
        remaining.remove(best_remaining)

    return [items[i] for i in selected_idx]

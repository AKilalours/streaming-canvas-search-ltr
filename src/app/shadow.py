# src/app/shadow.py
"""
Shadow Mode / Canary Deployment
=================================
Runs two model versions simultaneously and logs comparison.
This is how Netflix actually promotes models — shadow traffic before full rollout.

Pattern:
  - Primary model serves the user (live traffic)
  - Shadow model runs in background on same request
  - Differences logged to Redis for offline analysis
  - Promotion decision based on shadow metrics
"""
from __future__ import annotations
import json, time, hashlib
from typing import Any


class ShadowRunner:
    """
    Runs shadow model alongside primary for canary evaluation.
    """

    def __init__(self, redis_client=None) -> None:
        self.redis = redis_client
        self.shadow_log_key = "shadow:comparisons"
        self.comparison_count = 0

    def run_shadow(
        self,
        query: str,
        primary_results: list[dict],
        shadow_results: list[dict],
        model_a: str = "hybrid_ltr_v1",
        model_b: str = "hybrid_ltr_v2",
    ) -> dict[str, Any]:
        """
        Compare primary vs shadow results and log differences.
        User always gets primary results.
        """
        primary_ids = [r.get("doc_id", "") for r in primary_results[:10]]
        shadow_ids  = [r.get("doc_id", "") for r in shadow_results[:10]]

        # Rank overlap — how similar are the two lists?
        overlap_10 = len(set(primary_ids[:10]) & set(shadow_ids[:10]))
        rank_correlation = self._kendall_tau(primary_ids[:10], shadow_ids[:10])

        # Position differences for shared items
        pos_diffs = []
        for i, doc_id in enumerate(primary_ids[:10]):
            if doc_id in shadow_ids:
                j = shadow_ids.index(doc_id)
                pos_diffs.append(abs(i - j))

        comparison = {
            "query_hash": hashlib.md5(query.encode()).hexdigest()[:8],
            "model_a": model_a,
            "model_b": model_b,
            "overlap_top10": overlap_10,
            "rank_correlation": round(rank_correlation, 4),
            "avg_position_diff": round(sum(pos_diffs) / len(pos_diffs), 2) if pos_diffs else 0,
            "primary_top3": primary_ids[:3],
            "shadow_top3": shadow_ids[:3],
            "top1_agreement": primary_ids[0] == shadow_ids[0] if primary_ids and shadow_ids else False,
            "timestamp": time.time(),
        }

        if self.redis:
            try:
                self.redis.lpush(self.shadow_log_key, json.dumps(comparison))
                self.redis.ltrim(self.shadow_log_key, 0, 9999)
            except Exception:
                pass

        self.comparison_count += 1
        return comparison

    def get_shadow_report(self, n: int = 100) -> dict[str, Any]:
        """Aggregate shadow comparison stats."""
        comparisons = []
        if self.redis:
            try:
                raw = self.redis.lrange(self.shadow_log_key, 0, n - 1)
                comparisons = [json.loads(r) for r in raw]
            except Exception:
                pass

        if not comparisons:
            return {"error": "no shadow comparisons logged yet", "total": 0}

        avg_overlap = sum(c["overlap_top10"] for c in comparisons) / len(comparisons)
        avg_corr = sum(c["rank_correlation"] for c in comparisons) / len(comparisons)
        top1_agree = sum(1 for c in comparisons if c.get("top1_agree")) / len(comparisons)

        return {
            "total_comparisons": len(comparisons),
            "avg_overlap_top10": round(avg_overlap, 2),
            "avg_rank_correlation": round(avg_corr, 4),
            "top1_agreement_rate": round(top1_agree, 4),
            "recommendation": (
                "Models are highly aligned — safe to promote"
                if avg_overlap >= 7 and avg_corr >= 0.8
                else "Models diverge significantly — investigate before promoting"
            ),
        }

    def _kendall_tau(self, list_a: list, list_b: list) -> float:
        """Simplified Kendall tau rank correlation."""
        common = [x for x in list_a if x in list_b]
        if len(common) < 2:
            return 0.0
        idx_a = {x: i for i, x in enumerate(list_a)}
        idx_b = {x: i for i, x in enumerate(list_b)}
        concordant = discordant = 0
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                da = idx_a[common[i]] - idx_a[common[j]]
                db = idx_b[common[i]] - idx_b[common[j]]
                if da * db > 0:
                    concordant += 1
                elif da * db < 0:
                    discordant += 1
        total = concordant + discordant
        return (concordant - discordant) / total if total > 0 else 0.0

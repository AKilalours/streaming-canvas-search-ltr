"""Regression test for the recall@N truncation bug in src/eval/evaluate.py.

Root cause: run_eval() stored each method's ranked list as `ids[:k]` (k=10),
then passed that same 10-item list to aggregate_methods_list(..., recall_k=100).
recall_at_k() slices `ranked[:100]`, but the list only ever held 10 documents,
so recall@100 could never exceed recall@10. Every row in
reports/latest/metrics.json and reports/reference/metrics.json shows
recall@100 == recall@10 as a result.

Fix: keep max(k, recall_k) documents per query; the metric functions do their
own truncation for nDCG@k and MAP@k.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval.metrics import aggregate_methods_list  # noqa: E402

K = 10
RECALL_K = 100


def _fixture():
    """One query, 50 relevant docs, all found between ranks 11 and 60.

    Truncating the ranked list to 10 hides every one of them.
    """
    ranked_full = [f"d{i}" for i in range(200)]
    qrels = {"q1": {f"d{i}": 1 for i in range(10, 60)}}
    return ranked_full, qrels


def test_truncating_to_k_destroys_recall_at_100():
    ranked_full, qrels = _fixture()
    buggy = aggregate_methods_list({"q1": ranked_full[:K]}, qrels, k=K, recall_k=RECALL_K)
    assert buggy[f"recall@{K}"] == buggy[f"recall@{RECALL_K}"]
    assert buggy[f"recall@{RECALL_K}"] == 0.0


def test_keeping_recall_k_documents_reports_real_recall():
    ranked_full, qrels = _fixture()
    keep_k = max(K, RECALL_K)
    fixed = aggregate_methods_list({"q1": ranked_full[:keep_k]}, qrels, k=K, recall_k=RECALL_K)
    assert fixed[f"recall@{K}"] == 0.0
    assert fixed[f"recall@{RECALL_K}"] > fixed[f"recall@{K}"]
    assert fixed[f"recall@{RECALL_K}"] == 1.0
    # nDCG@k must be unaffected by keeping a longer list.
    buggy = aggregate_methods_list({"q1": ranked_full[:K]}, qrels, k=K, recall_k=RECALL_K)
    assert abs(fixed[f"ndcg@{K}"] - buggy[f"ndcg@{K}"]) < 1e-12


def test_evaluate_keeps_recall_k_documents():
    """Guard the call site itself so the truncation cannot come back."""
    src = (Path(__file__).resolve().parents[1] / "src/eval/evaluate.py").read_text()
    assert "keep_k = max(k, recall_k)" in src
    for line in src.splitlines():
        if "ranked[" in line and "][qid] =" in line:
            assert "[:k]" not in line, f"ranked list still truncated to k: {line.strip()}"
    assert "final_ids = (reranked_ids + tail)[:keep_k]" in src

from eval.metrics import average_precision_at_k, ndcg_at_k, recall_at_k


def test_metrics_sanity_perfect_ranking():
    qrels = {"d1": 2, "d2": 1, "d3": 0}
    ranked = ["d1", "d2", "d3"]
    assert ndcg_at_k(ranked, qrels, 3) == 1.0
    assert average_precision_at_k(ranked, qrels, 3) == 1.0
    assert recall_at_k(ranked, qrels, 2) == 1.0


def test_metrics_sanity_worse_ranking():
    qrels = {"d1": 2, "d2": 1}
    ranked = ["dX", "d2", "d1"]
    assert ndcg_at_k(ranked, qrels, 3) < 1.0
    assert average_precision_at_k(ranked, qrels, 3) < 1.0

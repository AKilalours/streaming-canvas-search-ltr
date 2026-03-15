# src/eval/failure_analysis.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from eval.evaluate import load_eval_inputs, run_ranked_lists
from eval.metrics import ndcg_at_k
from utils.logging import get_logger

log = get_logger("eval.failure_analysis")


def _get_eval_int(cfg: dict[str, Any], key: str, default: int) -> int:
    ev = cfg.get("eval", {}) or {}
    # accept both spellings
    if key == "candidate_k":
        return int(ev.get("candidate_k", ev.get("candidates_k", default)))
    return int(ev.get(key, default))


def _get_eval_float(cfg: dict[str, Any], key: str, default: float) -> float:
    ev = cfg.get("eval", {}) or {}
    return float(ev.get(key, default))


def _ndcg(qrels: dict[str, int], ranked: list[str], k: int) -> float:
    """
    Compatibility wrapper: calls ndcg_at_k using keywords so it works whether
    metrics.ndcg_at_k is defined as (qrels, ranked, k) or (ranked, qrels, k).
    """
    try:
        return float(ndcg_at_k(qrels=qrels, ranked=ranked, k=k))  # type: ignore[arg-type]
    except TypeError:
        return float(ndcg_at_k(ranked=ranked, qrels=qrels, k=k))  # type: ignore[arg-type]


def _as_ranked_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(d) for d in x]
    if isinstance(x, tuple):
        return [str(d) for d in x]
    if isinstance(x, dict):
        # wrong shape; can't safely interpret
        return []
    try:
        return [str(d) for d in list(x)]
    except Exception:
        return []


def _count_relevant(qrels: dict[str, int], ranked: list[str], k: int, min_rel: int) -> int:
    return sum(1 for did in ranked[:k] if int(qrels.get(did, 0)) >= min_rel)


def _first_relevant_rank(qrels: dict[str, int], ranked: list[str], k: int, min_rel: int) -> int | None:
    for i, did in enumerate(ranked[:k], start=1):
        if int(qrels.get(did, 0)) >= min_rel:
            return i
    return None


def _overlap_at_k(a: list[str], b: list[str], k: int) -> int:
    return len(set(a[:k]).intersection(set(b[:k])))


def _top_docs_block(
    *,
    ranked: list[str],
    corpus: dict[str, dict[str, Any]],
    qrels: dict[str, int],
    k: int,
) -> str:
    lines: list[str] = []
    for i, did in enumerate(ranked[:k], start=1):
        r = int(qrels.get(did, 0))
        title = (corpus.get(did, {}).get("title") or "").strip()
        lines.append(f"{i:02d}. rel={r} doc_id={did} — {title}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default="hybrid")
    ap.add_argument("--candidate", default="hybrid_ltr")
    ap.add_argument("--top_n", type=int, default=25)
    ap.add_argument("--min_rel", type=int, default=1)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    k = _get_eval_int(cfg, "k", 10)
    candidate_k = _get_eval_int(cfg, "candidate_k", 200)
    rerank_k = _get_eval_int(cfg, "rerank_k", 50)
    alpha = _get_eval_float(cfg, "alpha", 0.5)

    corpus, queries, qrels_all, assets = load_eval_inputs(cfg)

    base = args.baseline
    cand = args.candidate
    methods = (base, cand)

    ranked = run_ranked_lists(
        queries=queries,
        corpus=corpus,
        qrels=qrels_all,
        assets=assets,
        k=k,
        candidate_k=candidate_k,
        rerank_k=rerank_k,
        alpha=alpha,
        methods=methods,
    )

    if base not in ranked or cand not in ranked:
        raise SystemExit(
            f"[failure_analysis] methods missing. ranked has: {list(ranked.keys())}, "
            f"requested baseline={base}, candidate={cand}"
        )

    # (delta_ndcg, cand_ndcg, base_ndcg, qid)
    rows: list[tuple[float, float, float, str]] = []
    for qid in queries.keys():
        qr = (qrels_all.get(qid, {}) or {})
        b_list = _as_ranked_list(ranked[base].get(qid, []))
        c_list = _as_ranked_list(ranked[cand].get(qid, []))

        b_ndcg = _ndcg(qrels=qr, ranked=b_list, k=k)
        c_ndcg = _ndcg(qrels=qr, ranked=c_list, k=k)
        rows.append((c_ndcg - b_ndcg, c_ndcg, b_ndcg, qid))

    rows.sort(key=lambda x: x[0])  # worst first
    worst = rows[: max(1, int(args.top_n))]

    md: list[str] = []
    md.append("# Failure Analysis")
    md.append("")
    md.append(f"- baseline: **{base}**")
    md.append(f"- candidate: **{cand}**")
    md.append(f"- k: {k}, candidate_k: {candidate_k}, rerank_k: {rerank_k}, alpha: {alpha}")
    md.append(f"- min_rel: {int(args.min_rel)}")
    md.append("")
    md.append("## Worst deltas (candidate - baseline)")
    md.append("")

    for delta, c_ndcg, b_ndcg, qid in worst:
        qtext = queries.get(qid, "")
        qr = (qrels_all.get(qid, {}) or {})
        b_list = _as_ranked_list(ranked[base].get(qid, []))
        c_list = _as_ranked_list(ranked[cand].get(qid, []))

        b_rel = _count_relevant(qr, b_list, k, min_rel=int(args.min_rel))
        c_rel = _count_relevant(qr, c_list, k, min_rel=int(args.min_rel))
        ov = _overlap_at_k(b_list, c_list, k)
        b_first = _first_relevant_rank(qr, b_list, k, min_rel=int(args.min_rel))
        c_first = _first_relevant_rank(qr, c_list, k, min_rel=int(args.min_rel))

        md.append(f"### Δ nDCG@{k} = {delta:.4f}  (qid={qid})")
        md.append("")
        md.append(f"**Query:** {qtext}")
        md.append("")
        md.append(f"- baseline ndcg@{k}: {b_ndcg:.4f} | candidate ndcg@{k}: {c_ndcg:.4f}")
        md.append(f"- relevant@{k}: baseline={b_rel} | candidate={c_rel} | overlap@{k}={ov}")
        md.append(f"- first relevant rank@{k}: baseline={b_first} | candidate={c_first}")
        md.append("")

        md.append(f"**{base} top-{k}:**")
        md.append("```")
        md.append(_top_docs_block(ranked=b_list, corpus=corpus, qrels=qr, k=k))
        md.append("```")
        md.append("")

        md.append(f"**{cand} top-{k}:**")
        md.append("```")
        md.append(_top_docs_block(ranked=c_list, corpus=corpus, qrels=qr, k=k))
        md.append("```")
        md.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")

    log.info("Wrote %s", str(out_path))


if __name__ == "__main__":
    main()

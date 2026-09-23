# src/pipelines/data_quality.py
"""
Data validation for the MovieLens retrieval dataset.

Runs before training so a broken or partial dataset stops the pipeline instead
of producing a model that is then scored on the same broken data.

Every check here is computed from the files on disk. Hard failures raise
DataQualityError; softer findings are returned as warnings in the report so the
run log shows them.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REQUIRED_FILES = ("corpus.jsonl", "queries.jsonl", "qrels.json")


class DataQualityError(RuntimeError):
    pass


@dataclass
class DataQualityReport:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "stats": self.stats,
            "errors": self.errors,
            "warnings": self.warnings,
            "fingerprint": self.fingerprint,
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise DataQualityError(f"{path}:{i} is not valid JSON: {e}") from e
    return rows


def fingerprint_files(paths: list[Path]) -> str:
    """sha256 over file names + contents, so any data change changes the id."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x)):
        h.update(str(p.as_posix()).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def check_data_quality(
    processed_dir: str | Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
    min_queries: dict[str, int] | None = None,
    min_docs: int = 1000,
    max_cross_split_duplicate_frac: float = 1.0,
) -> DataQualityReport:
    """
    Validate a processed dataset directory laid out as <dir>/<split>/{corpus,queries,qrels}.

    Hard checks (any failure -> passed=False):
      files_present, json_parses, required_keys, unique_ids, corpus_min_size,
      corpus_consistent_across_splits, qrels_reference_known_queries,
      qrels_reference_known_docs, every_query_has_relevant, min_queries_per_split,
      cross_split_duplicates_within_limit
    """
    root = Path(processed_dir)
    min_queries = min_queries or {}
    rep = DataQualityReport(passed=True)

    def fail(check: str, msg: str) -> None:
        rep.checks[check] = False
        rep.errors.append(f"{check}: {msg}")

    def ok(check: str) -> None:
        rep.checks.setdefault(check, True)

    # 1. files present
    all_files: list[Path] = []
    for s in splits:
        for fn in REQUIRED_FILES:
            p = root / s / fn
            if not p.exists():
                fail("files_present", f"missing {p}")
            else:
                all_files.append(p)
    ok("files_present")
    if not rep.checks["files_present"]:
        rep.passed = False
        return rep

    corpus_ids_by_split: dict[str, set[str]] = {}
    query_text_by_split: dict[str, set[str]] = {}

    for s in splits:
        corpus = _read_jsonl(root / s / "corpus.jsonl")
        queries = _read_jsonl(root / s / "queries.jsonl")
        try:
            qrels = json.loads((root / s / "qrels.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            fail("json_parses", f"{s}/qrels.json: {e}")
            continue
        ok("json_parses")

        # 2. required keys
        bad_docs = [r for r in corpus if not r.get("doc_id") or "text" not in r]
        bad_qs = [r for r in queries if not r.get("query_id") or not str(r.get("text", "")).strip()]
        if bad_docs:
            fail("required_keys", f"{s}: {len(bad_docs)} corpus rows missing doc_id/text")
        if bad_qs:
            fail("required_keys", f"{s}: {len(bad_qs)} queries missing query_id/text")
        ok("required_keys")

        # 3. unique ids
        doc_ids = [str(r.get("doc_id")) for r in corpus]
        q_ids = [str(r.get("query_id")) for r in queries]
        if len(doc_ids) != len(set(doc_ids)):
            fail("unique_ids", f"{s}: {len(doc_ids) - len(set(doc_ids))} duplicate doc_ids")
        if len(q_ids) != len(set(q_ids)):
            fail("unique_ids", f"{s}: {len(q_ids) - len(set(q_ids))} duplicate query_ids")
        ok("unique_ids")

        doc_set, q_set = set(doc_ids), set(q_ids)
        corpus_ids_by_split[s] = doc_set
        query_text_by_split[s] = {str(r.get("text", "")).strip().lower() for r in queries}

        # 4. corpus size
        if len(doc_set) < min_docs:
            fail("corpus_min_size", f"{s}: {len(doc_set)} docs < {min_docs}")
        ok("corpus_min_size")

        # 5. qrels integrity
        if not isinstance(qrels, dict):
            fail("qrels_reference_known_queries", f"{s}: qrels.json is not an object")
            continue
        unknown_q = [q for q in qrels if q not in q_set]
        if unknown_q:
            fail("qrels_reference_known_queries", f"{s}: {len(unknown_q)} qrels query_ids not in queries.jsonl")
        ok("qrels_reference_known_queries")

        unknown_d = sum(1 for rels in qrels.values() for d in rels if d not in doc_set)
        if unknown_d:
            fail("qrels_reference_known_docs", f"{s}: {unknown_d} qrels doc_ids not in corpus")
        ok("qrels_reference_known_docs")

        no_rel = [q for q in q_ids if not any(int(v) > 0 for v in (qrels.get(q) or {}).values())]
        if no_rel:
            fail("every_query_has_relevant", f"{s}: {len(no_rel)} queries have no relevant doc")
        ok("every_query_has_relevant")

        # 6. minimum query counts
        need = int(min_queries.get(s, 1))
        if len(q_set) < need:
            fail("min_queries_per_split", f"{s}: {len(q_set)} queries < {need}")
        ok("min_queries_per_split")

        rel_counts = sorted(sum(1 for v in r.values() if int(v) > 0) for r in qrels.values())
        rep.stats[s] = {
            "num_docs": len(doc_set),
            "num_queries": len(q_set),
            "num_unique_query_texts": len(query_text_by_split[s]),
            "median_relevant_per_query": statistics.median(rel_counts) if rel_counts else 0,
        }

    # 7. same corpus across splits (eval and training must index the same catalog)
    sets = list(corpus_ids_by_split.values())
    if sets and any(x != sets[0] for x in sets[1:]):
        fail("corpus_consistent_across_splits", "doc_id sets differ between splits")
    ok("corpus_consistent_across_splits")

    # 8. query-text overlap between train and held-out splits
    train_t = query_text_by_split.get("train", set())
    for s in splits:
        if s == "train" or s not in query_text_by_split:
            continue
        held = query_text_by_split[s]
        dup = len(held & train_t)
        frac = dup / max(1, len(held))
        rep.stats.setdefault("cross_split_duplicates", {})[f"{s}_in_train"] = {
            "count": dup,
            "fraction_of_unique_texts": round(frac, 4),
        }
        if frac > max_cross_split_duplicate_frac:
            fail(
                "cross_split_duplicates_within_limit",
                f"{s}: {frac:.1%} of unique query texts also appear in train "
                f"(limit {max_cross_split_duplicate_frac:.1%})",
            )
        elif dup:
            rep.warnings.append(
                f"{s}: {dup} unique query texts ({frac:.1%}) also appear verbatim in train"
            )
    ok("cross_split_duplicates_within_limit")

    rep.fingerprint = fingerprint_files(all_files)
    rep.passed = all(rep.checks.values())
    return rep

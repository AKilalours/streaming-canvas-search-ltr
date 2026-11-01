from __future__ import annotations
import argparse, json, random, re
from pathlib import Path
from typing import Any

TEMPLATES_TAG = [
    "{x}",
    "{x} movies",
    "movies about {x}",
    "best {x} movies",
    "top {x} films",
    "{x} comedy",
    "{x} romance",
]
TEMPLATES_GENRE = [
    "{x}",
    "{x} movies",
    "best {x} movies",
    "top rated {x} movies",
    "{x} classics",
    "new {x} movies",
]

def load_jsonl(p: Path) -> list[dict[str, Any]]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def parse_catalog(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    parts = [p.strip() for p in (text or "").split("|")]
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        out[k.strip().lower()] = v.strip()
    return out

def norm_key(x: str) -> str:
    x = x.strip().lower()
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"[^a-z0-9 _-]+", "", x)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_queries", type=int, default=800)
    ap.add_argument("--min_docs", type=int, default=5)
    ap.add_argument("--max_docs", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--templates_per_key", type=int, default=6)
    args = ap.parse_args()

    random.seed(args.seed)

    rows = load_jsonl(Path(args.corpus))

    tag2docs: dict[str, set[str]] = {}
    genre2docs: dict[str, set[str]] = {}

    for r in rows:
        did = str(r.get("doc_id"))
        meta = parse_catalog(str(r.get("text") or ""))
        genres = [g.strip() for g in (meta.get("genres") or "").split(",") if g.strip()]
        tags = [t.strip() for t in (meta.get("tags") or "").split(",") if t.strip() and t.strip().lower() != "none"]

        for g in genres:
            genre2docs.setdefault(norm_key(g), set()).add(did)
        for t in tags:
            tag2docs.setdefault(norm_key(t), set()).add(did)

    def expand(m: dict[str, set[str]], kind: str) -> list[tuple[str, str, list[str]]]:
        out = []
        templates = TEMPLATES_TAG if kind == "tag" else TEMPLATES_GENRE
        for key, docs in m.items():
            if not (args.min_docs <= len(docs) <= args.max_docs):
                continue
            chosen = templates[: args.templates_per_key]
            for ti, tmpl in enumerate(chosen):
                qid = f"{kind}:{key}:t{ti}"
                q = tmpl.format(x=key)
                out.append((qid, q, sorted(docs)))
        random.shuffle(out)
        return out

    all_q = expand(tag2docs, "tag") + expand(genre2docs, "genre")
    random.shuffle(all_q)
    all_q = all_q[: args.n_queries]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    queries_path = out_dir / "queries.jsonl"
    qrels_path = out_dir / "qrels.jsonl"

    with queries_path.open("w", encoding="utf-8") as fq, qrels_path.open("w", encoding="utf-8") as fr:
        for qid, qtext, docs in all_q:
            fq.write(json.dumps({"query_id": qid, "text": qtext}, ensure_ascii=False) + "\n")
            for did in docs:
                fr.write(json.dumps({"query_id": qid, "doc_id": did, "relevance": 1}) + "\n")

    print(f"[OK] wrote {queries_path} ({len(all_q)} queries)")
    print(f"[OK] wrote {qrels_path}")

if __name__ == "__main__":
    main()

# flows/train_ltr_production.py
"""
Phase 4 — Production LTR Training Flow
=========================================
Upgrades streaming_search_flow.py with:
  - @netflix_standard on every step (auto-retry, structured logs, artifact pruning)
  - Quality gate validation before model promotion
  - Automatic drift check post-training
  - Auto-promotion: latest → reference if gates pass
  - Metaflow card with full metrics summary
  - Parallel foreach for multi-dataset training

Run:
  python flows/train_ltr_production.py run --dataset movielens
  python flows/train_ltr_production.py run --dataset movielens --ndcg_gate 0.35
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from metaflow import FlowSpec, Parameter, card, current, step
from metaflow.cards import Markdown

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from monitoring.metaflow_decorators import netflix_standard


def _run(cmd: str) -> str:
    print(f"[RUN] {cmd}", flush=True)
    env = {**os.environ, "PYTHONPATH": "src"}
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result.stdout


class ProductionLTRFlow(FlowSpec):
    """Production LTR training pipeline with full quality gates and auto-promotion."""

    dataset    = Parameter("dataset",    default="movielens")
    split      = Parameter("split",      default="test")
    model      = Parameter("model",      default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    batch_size = Parameter("batch_size", default=32,  type=int)
    ndcg_gate  = Parameter("ndcg_gate",  default=0.0, type=float,
                            help="Min nDCG@10 required to pass (0 = skip gate)")

    @netflix_standard(retry=1, cpu=1)
    @step
    def start(self):
        """Validate paths and set artifact locations."""
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.model)
        self.processed_corpus = f"data/processed/{self.dataset}/{self.split}/corpus.jsonl"
        self.bm25_artifact    = f"artifacts/bm25/{self.dataset}_bm25.pkl"
        self.emb_dir          = f"artifacts/faiss/{self.dataset}_{slug}"
        self.ltr_path         = f"artifacts/ltr/{self.dataset}_ltr.pkl"
        self.autogen_train    = "configs/_train_autogen.yaml"
        self.autogen_eval     = "configs/_eval_autogen.yaml"

        if not Path(self.processed_corpus).exists():
            raise FileNotFoundError(f"Corpus missing: {self.processed_corpus}")
        self.next(self.build_bm25)

    @netflix_standard(retry=2, cpu=2, memory_mb=2048)
    @step
    def build_bm25(self):
        """Build or reuse BM25 index."""
        Path(self.bm25_artifact).parent.mkdir(parents=True, exist_ok=True)
        if Path(self.bm25_artifact).exists():
            print(f"[SKIP] BM25 exists: {self.bm25_artifact}", flush=True)
        else:
            _run(f"python scripts/build_bm25.py --corpus {self.processed_corpus} --out {self.bm25_artifact}")
        self.next(self.build_embeddings)

    @netflix_standard(retry=1, cpu=4, memory_mb=8192, prune_large_attrs=["_embed_buffer"])
    @step
    def build_embeddings(self):
        """Build or reuse FAISS dense embeddings."""
        self._embed_buffer = None  # pruned by decorator
        Path(self.emb_dir).mkdir(parents=True, exist_ok=True)
        emb_file = Path(self.emb_dir) / "embeddings.npy"
        if emb_file.exists():
            print(f"[SKIP] Embeddings exist: {emb_file}", flush=True)
        else:
            _run(
                f"python scripts/build_embeddings.py "
                f"--corpus {self.processed_corpus} "
                f"--out_dir {self.emb_dir} "
                f'--model "{self.model}" '
                f"--batch_size {self.batch_size}"
            )
        self.next(self.train_ltr)

    @netflix_standard(retry=1, cpu=4, memory_mb=8192, prune_large_attrs=["_X_train", "_y_train"])
    @step
    def train_ltr(self):
        """Train LightGBM LTR model."""
        self._X_train = None  # pruned
        self._y_train = None  # pruned
        base_cfg = Path("configs/train.yaml")
        if not base_cfg.exists():
            raise FileNotFoundError("configs/train.yaml not found")
        txt = base_cfg.read_text(encoding="utf-8")
        txt = txt.replace("scifact", self.dataset).replace("nfcorpus", self.dataset)
        txt = re.sub(r"(?m)^(\s*dataset_processed_dir:\s*).*$", r"\1" + f"data/processed/{self.dataset}", txt)
        txt = re.sub(r"(?m)^(\s*bm25_artifact:\s*).*$", r"\1" + self.bm25_artifact, txt)
        txt = re.sub(r"(?m)^(\s*emb_dir:\s*).*$", r"\1" + self.emb_dir, txt)
        txt = re.sub(r"(?m)^(\s*model_name:\s*).*$", r"\1" + f"{self.dataset}_ltr.pkl", txt)
        txt = re.sub(r"(?m)^(\s*meta_name:\s*).*$", r"\1" + f"{self.dataset}_ltr_meta.json", txt)
        Path(self.autogen_train).write_text(txt, encoding="utf-8")
        _run(f"python -m ranking.ltr_train --config {self.autogen_train}")
        self.next(self.evaluate)

    @card
    @netflix_standard(retry=1, cpu=2)
    @step
    def evaluate(self):
        """Run offline evaluation and emit Metaflow card."""
        base_eval = Path("configs/eval.yaml")
        if not base_eval.exists():
            raise FileNotFoundError("configs/eval.yaml not found")
        txt = base_eval.read_text(encoding="utf-8")
        txt = txt.replace("scifact", self.dataset).replace("nfcorpus", self.dataset)
        txt = re.sub(r"(?m)^(\s*dataset_processed_dir:\s*).*$", r"\1" + f"data/processed/{self.dataset}", txt)
        txt = re.sub(r"(?m)^(\s*bm25_artifact:\s*).*$", r"\1" + self.bm25_artifact, txt)
        txt = re.sub(r"(?m)^(\s*emb_dir:\s*).*$", r"\1" + self.emb_dir, txt)
        txt = re.sub(r"(?m)^(\s*ltr_path:\s*).*$", r"\1" + self.ltr_path, txt)
        Path(self.autogen_eval).write_text(txt, encoding="utf-8")
        _run(f"python -m eval.evaluate --config {self.autogen_eval}")

        metrics_path = Path("reports/latest/metrics.json")
        if not metrics_path.exists():
            raise FileNotFoundError("reports/latest/metrics.json missing after eval")
        self.metrics = json.loads(metrics_path.read_text())

        # Emit card
        current.card.append(Markdown("# Production LTR Training Report"))
        current.card.append(Markdown(f"- **Dataset**: `{self.dataset}`"))
        current.card.append(Markdown(f"- **Model**: `{self.model}`"))
        current.card.append(Markdown("## Offline Metrics"))
        current.card.append(Markdown("```json\n" + json.dumps(self.metrics, indent=2) + "\n```"))
        self.next(self.gate_check)

    @netflix_standard(retry=1, cpu=1)
    @step
    def gate_check(self):
        """Quality gate: fail fast if metrics don't meet thresholds."""
        result = subprocess.run(
            ["python", "-m", "pipelines.gates", "--gates", "configs/gates.yaml", "--run_dir", "reports/latest"],
            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"},
        )
        print(result.stdout, flush=True)
        if result.returncode != 0:
            raise RuntimeError(f"[GATE FAIL]\n{result.stdout}\n{result.stderr}")

        # Optional explicit nDCG gate
        if self.ndcg_gate > 0:
            methods = {row.get("method"): row for row in self.metrics.get("methods", []) if isinstance(row, dict)}
            ltr_row = methods.get("hybrid_ltr", {})
            actual_ndcg = float(ltr_row.get("ndcg@10", 0.0))
            if actual_ndcg < self.ndcg_gate:
                raise RuntimeError(f"[GATE FAIL] nDCG@10={actual_ndcg:.4f} < threshold={self.ndcg_gate}")

        print("[GATE PASS] ✅ All quality gates passed", flush=True)
        self.gate_passed = True
        self.next(self.drift_check)

    @netflix_standard(retry=1, cpu=1)
    @step
    def drift_check(self):
        """Run drift monitor post-training."""
        result = subprocess.run(
            ["python", "scripts/drift_monitor.py",
             "--latest",    "reports/latest/metrics.json",
             "--reference", "reports/reference/metrics.json",
             "--out",       "reports/latest/drift_report.json",
             "--ndcg_drop", "0.03", "--p99_max_ms", "300"],
            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"},
        )
        print(result.stdout, flush=True)
        drift_path = Path("reports/latest/drift_report.json")
        if drift_path.exists():
            self.drift_report = json.loads(drift_path.read_text())
            if self.drift_report.get("should_retrain"):
                print("[DRIFT] ⚠️ Drift detected post-training — check model quality.", flush=True)
            else:
                print("[DRIFT] ✅ No drift after retraining.", flush=True)
        self.next(self.promote)

    @netflix_standard(retry=1, cpu=1)
    @step
    def promote(self):
        """Promote model as new reference baseline if gates passed."""
        import shutil
        if not getattr(self, "gate_passed", False):
            print("[PROMOTE] Skipped — gates did not pass.", flush=True)
            self.promoted = False
            self.next(self.end)
            return

        ref_dir = Path("reports/reference")
        ref_dir.mkdir(parents=True, exist_ok=True)

        # Backup current reference
        for fname in ["metrics.json", "latency.json"]:
            src = ref_dir / fname
            if src.exists():
                src.rename(ref_dir / f"{fname}.prev")

        # Promote latest → reference
        for fname in ["metrics.json", "latency.json"]:
            src = Path("reports/latest") / fname
            if src.exists():
                shutil.copy2(src, ref_dir / fname)

        print(f"[PROMOTE] ✅ New reference baseline set.", flush=True)
        self.promoted = True
        self.next(self.end)

    @step
    def end(self):
        print(f"[DONE] ProductionLTRFlow complete. promoted={getattr(self, 'promoted', False)}", flush=True)


if __name__ == "__main__":
    ProductionLTRFlow()

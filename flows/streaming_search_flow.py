from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

from metaflow import FlowSpec, Parameter, card, current, step
from metaflow.cards import Markdown


def _run(cmd: str) -> None:
    print(f"[RUN] {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


def _health_ok(url: str, timeout_s: float = 0.5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def _start_uvicorn(host: str, port: int) -> Optional[subprocess.Popen]:
    """
    Starts uvicorn only if the API isn't already up.
    Returns a Popen handle if started, else None.
    """
    base = f"http://{host}:{port}"
    if _health_ok(base + "/health"):
        print(f"[latency] API already running at {base}", flush=True)
        return None

    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    print(f"[latency] starting uvicorn: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    for _ in range(60):
        if _health_ok(base + "/health"):
            print(f"[latency] API is ready at {base}", flush=True)
            return proc
        time.sleep(0.25)

    out = ""
    try:
        out = proc.stdout.read() if proc.stdout else ""
    except Exception:
        pass
    try:
        proc.terminate()
    except Exception:
        pass
    raise RuntimeError("uvicorn did not become ready. Logs:\n" + out)


def _stop_proc(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


class StreamingSearchLTRFlow(FlowSpec):
    dataset = Parameter("dataset", default="nfcorpus")
    split = Parameter("split", default="test")
    model = Parameter(
        "model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    batch_size = Parameter("batch_size", default=32, type=int)

    @step
    def start(self):
        self.processed_corpus = Path(f"data/processed/{self.dataset}/{self.split}/corpus.jsonl")
        self.bm25_artifact = Path(f"artifacts/bm25/{self.dataset}_bm25.pkl")

        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.model)
        self.emb_dir = Path(f"artifacts/faiss/{self.dataset}_{slug}")

        self.ltr_path = Path(f"artifacts/ltr/{self.dataset}_ltr.pkl")

        self.next(self.ensure_processed)

    @step
    def ensure_processed(self):
        if not self.processed_corpus.exists():
            raise FileNotFoundError(f"missing processed corpus: {self.processed_corpus}")
        print(f"[OK] processed exists: {self.processed_corpus}", flush=True)
        self.next(self.build_bm25)

    @step
    def build_bm25(self):
        self.bm25_artifact.parent.mkdir(parents=True, exist_ok=True)
        if self.bm25_artifact.exists():
            print(f"[SKIP] BM25 exists: {self.bm25_artifact}", flush=True)
        else:
            _run(f"python scripts/build_bm25.py --corpus {self.processed_corpus} --out {self.bm25_artifact}")
        self.next(self.build_embeddings)

    @step
    def build_embeddings(self):
        self.emb_dir.mkdir(parents=True, exist_ok=True)
        emb_file = self.emb_dir / "embeddings.npy"
        if emb_file.exists():
            print(f"[SKIP] embeddings exist: {emb_file}", flush=True)
        else:
            _run(
                "python scripts/build_embeddings.py "
                f"--corpus {self.processed_corpus} "
                f"--out_dir {self.emb_dir} "
                f'--model "{self.model}" '
                f"--batch_size {int(self.batch_size)}"
            )
        self.next(self.train_ltr)

    @step
    def train_ltr(self):
        base_cfg = Path("configs/train.yaml")
        if not base_cfg.exists():
            raise FileNotFoundError("configs/train.yaml not found")

        txt = base_cfg.read_text(encoding="utf-8")

        # Kill any leftover scifact references
        txt = txt.replace("scifact", self.dataset)

        # Force paths for this run
        txt = re.sub(
            r"(?m)^(\s*dataset_processed_dir:\s*).*$",
            r"\1" + f"data/processed/{self.dataset}",
            txt,
        )
        txt = re.sub(r"(?m)^(\s*bm25_artifact:\s*).*$", r"\1" + str(self.bm25_artifact), txt)
        txt = re.sub(r"(?m)^(\s*emb_dir:\s*).*$", r"\1" + str(self.emb_dir), txt)

        # These names are used by your trainer to write artifacts under artifacts/ltr/
        txt = re.sub(r"(?m)^(\s*model_name:\s*).*$", r"\1" + f"{self.dataset}_ltr.pkl", txt)
        txt = re.sub(r"(?m)^(\s*meta_name:\s*).*$", r"\1" + f"{self.dataset}_ltr_meta.json", txt)

        autogen = Path("configs/_train_autogen.yaml")
        autogen.write_text(txt, encoding="utf-8")
        print(f"[OK] wrote {autogen} with run-specific paths", flush=True)

        _run(f"python -m ranking.ltr_train --config {autogen}")
        self.next(self.evaluate)

    @card
    @step
    def evaluate(self):
        cfg = Path("configs/eval.yaml")
        if not cfg.exists():
            raise FileNotFoundError("configs/eval.yaml not found")

        txt = cfg.read_text(encoding="utf-8")

        # Kill any leftover scifact references
        txt = txt.replace("scifact", self.dataset)

        # Force paths for this run
        txt = re.sub(
            r"(?m)^(\s*dataset_processed_dir:\s*).*$",
            r"\1" + f"data/processed/{self.dataset}",
            txt,
        )
        txt = re.sub(r"(?m)^(\s*bm25_artifact:\s*).*$", r"\1" + str(self.bm25_artifact), txt)
        txt = re.sub(r"(?m)^(\s*emb_dir:\s*).*$", r"\1" + str(self.emb_dir), txt)
        txt = re.sub(r"(?m)^(\s*ltr_path:\s*).*$", r"\1" + str(self.ltr_path), txt)

        autogen = Path("configs/_eval_autogen.yaml")
        autogen.write_text(txt, encoding="utf-8")
        print(f"[OK] wrote {autogen} with run-specific paths", flush=True)

        _run(f"python -m src.eval.evaluate --config {autogen}")

        metrics_path = Path("reports/latest/metrics.json")
        if not metrics_path.exists():
            raise FileNotFoundError("reports/latest/metrics.json not produced by evaluation")

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        current.card.append(Markdown("# Evaluation"))
        current.card.append(Markdown(f"- dataset: `{self.dataset}`"))
        current.card.append(Markdown(f"- split: `{self.split}`"))
        current.card.append(Markdown(f"- model: `{self.model}`"))
        current.card.append(Markdown("## metrics.json"))
        current.card.append(Markdown("```json\n" + json.dumps(metrics, indent=2) + "\n```"))

        self.next(self.latency)

    @card
    @step
    def latency(self):
        # Use env vars so resume doesn't KeyError on missing parameters
        host = os.getenv("API_HOST", "127.0.0.1")
        port = int(os.getenv("API_PORT", "8000"))
        base = f"http://{host}:{port}"

        proc = None
        out_txt = ""
        err = None

        try:
            proc = _start_uvicorn(host, port)

            env = os.environ.copy()
            env["PYTHONPATH"] = "src" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

            out_txt = subprocess.check_output(
                [sys.executable, "src/eval/latency_bench.py"],
                env=env,
                stderr=subprocess.STDOUT,
                text=True,
            )

        except subprocess.CalledProcessError as e:
            err = f"latency_bench failed (exit={e.returncode})"
            out_txt = e.output or ""
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        finally:
            _stop_proc(proc)

        Path("reports/latest").mkdir(parents=True, exist_ok=True)
        Path("reports/latest/latency_bench.txt").write_text(out_txt or "", encoding="utf-8")

        current.card.append(Markdown("# Latency Benchmark"))
        current.card.append(Markdown(f"- API base: `{base}`"))
        current.card.append(Markdown(f"- status: {'**FAILED** ' + err if err else '**OK**'}"))
        current.card.append(Markdown("## output"))
        current.card.append(Markdown("```text\n" + (out_txt or "").strip()[:8000] + "\n```"))

        self.next(self.end)

    @step
    def end(self):
        print("[DONE] Flow completed.", flush=True)


if __name__ == "__main__":
    StreamingSearchLTRFlow()

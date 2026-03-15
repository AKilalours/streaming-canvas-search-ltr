from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml
from metaflow import FlowSpec, Parameter, card, current, step
from metaflow.cards import Markdown


def _run(args: list[str]) -> None:
    print("[RUN]", " ".join(args), flush=True)
    subprocess.check_call(args)


def _health_ok(url: str, timeout_s: float = 0.5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def _start_uvicorn(host: str, port: int) -> subprocess.Popen | None:
    base = f"http://{host}:{port}"
    if _health_ok(base + "/health"):
        print(f"[latency] API already running at {base}", flush=True)
        return None

    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", host,
        "--port", str(port),
        "--log-level", "warning",
    ]
    print(f"[latency] starting uvicorn: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

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


def _stop_proc(proc: subprocess.Popen | None) -> None:
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


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(obj: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


class StreamingSearchLTRFlow(FlowSpec):
    dataset = Parameter("dataset", default="nfcorpus")
    split = Parameter("split", default="test")
    model = Parameter("model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    batch_size = Parameter("batch_size", default=32, type=int)

    @step
    def start(self):
        self.run_id = str(current.run_id)

        self.processed_corpus = Path(f"data/processed/{self.dataset}/{self.split}/corpus.jsonl")
        self.bm25_artifact = Path(f"artifacts/bm25/{self.dataset}_bm25.pkl")

        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.model)
        self.emb_dir = Path(f"artifacts/faiss/{self.dataset}_{slug}")
        self.ltr_path = Path(f"artifacts/ltr/{self.dataset}_ltr.pkl")

        self.next(self.ensure_processed)

    @step
    def ensure_processed(self):
        if not self.processed_corpus.exists():
            raise FileNotFoundError(
                f"missing processed corpus: {self.processed_corpus}\n"
                f"Run your preprocess step first to create data/processed/{self.dataset}/{self.split}/corpus.jsonl"
            )
        print(f"[OK] processed exists: {self.processed_corpus}", flush=True)
        self.next(self.build_bm25)

    @step
    def build_bm25(self):
        self.bm25_artifact.parent.mkdir(parents=True, exist_ok=True)
        if self.bm25_artifact.exists():
            print(f"[SKIP] BM25 exists: {self.bm25_artifact}", flush=True)
        else:
            _run([sys.executable, "scripts/build_bm25.py", "--corpus", str(self.processed_corpus), "--out", str(self.bm25_artifact)])
        self.next(self.build_embeddings)

    @step
    def build_embeddings(self):
        self.emb_dir.mkdir(parents=True, exist_ok=True)
        emb_file = self.emb_dir / "embeddings.npy"
        if emb_file.exists():
            print(f"[SKIP] embeddings exist: {emb_file}", flush=True)
        else:
            _run([
                sys.executable, "scripts/build_embeddings.py",
                "--corpus", str(self.processed_corpus),
                "--out_dir", str(self.emb_dir),
                "--model", self.model,
                "--batch_size", str(int(self.batch_size)),
            ])
        self.next(self.train_ltr)

    @step
    def train_ltr(self):
        base_cfg = Path("configs/train.yaml")
        if not base_cfg.exists():
            raise FileNotFoundError("configs/train.yaml not found")

        cfg = _load_yaml(base_cfg)

        # Force dataset + artifacts (no “missing key” issues)
        cfg["dataset"] = self.dataset
        cfg["split"] = self.split
        cfg["dataset_processed_dir"] = f"data/processed/{self.dataset}"
        cfg["bm25_artifact"] = str(self.bm25_artifact)
        cfg["emb_dir"] = str(self.emb_dir)

        # Force LTR output naming
        cfg["model_name"] = f"{self.dataset}_ltr.pkl"
        cfg["meta_name"] = f"{self.dataset}_ltr_meta.json"

        autogen = Path("configs/_train_autogen.yaml")
        _dump_yaml(cfg, autogen)
        print(f"[OK] wrote {autogen} with run-specific paths", flush=True)

        _run([sys.executable, "-m", "src.ranking.ltr_train", "--config", str(autogen)])
        self.next(self.evaluate)

    @step
    def evaluate(self):
        base_cfg = Path("configs/eval.yaml")
        if not base_cfg.exists():
            raise FileNotFoundError("configs/eval.yaml not found")

        cfg = _load_yaml(base_cfg)

        cfg["dataset"] = self.dataset
        cfg["split"] = self.split
        cfg["dataset_processed_dir"] = f"data/processed/{self.dataset}"
        cfg["bm25_artifact"] = str(self.bm25_artifact)
        cfg["emb_dir"] = str(self.emb_dir)

        # CRITICAL: ensure LTR path EXISTS in config, not “maybe regex matched”
        cfg["ltr_path"] = str(self.ltr_path)

        autogen = Path("configs/_eval_autogen.yaml")
        _dump_yaml(cfg, autogen)
        print(f"[OK] wrote {autogen} with run-specific paths", flush=True)

        _run([sys.executable, "-m", "src.eval.evaluate", "--config", str(autogen)])

        metrics_path = Path("reports/latest/metrics.json")
        if not metrics_path.exists():
            raise FileNotFoundError("reports/latest/metrics.json not produced by evaluation")

        self.next(self.latency)

    @step
    def latency(self):
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

        # Package outputs for this run (so you can submit/share)
        run_dir = Path(f"reports/runs/{self.run_id}")
        run_dir.mkdir(parents=True, exist_ok=True)

        for p in [
            Path("reports/latest/metrics.json"),
            Path("reports/latest/latency.json"),
            Path("reports/latest/latency_bench.txt"),
            Path("configs/_train_autogen.yaml"),
            Path("configs/_eval_autogen.yaml"),
            self.ltr_path,
            self.bm25_artifact,
        ]:
            if p.exists():
                (run_dir / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8") if p.suffix in {".json", ".yaml", ".txt"} else (run_dir / p.name).write_bytes(p.read_bytes())

        # Record status
        (run_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "dataset": self.dataset,
                    "split": self.split,
                    "model": self.model,
                    "api_base": base,
                    "latency_status": "FAILED" if err else "OK",
                    "latency_error": err,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        self.next(self.report)

    @card
    @step
    def report(self):
        # One “phenomenal” card for everything
        metrics_path = Path("reports/latest/metrics.json")
        latency_path = Path("reports/latest/latency.json")
        latency_txt = Path("reports/latest/latency_bench.txt")

        current.card.append(Markdown("# Streaming Search + LTR Report"))
        current.card.append(Markdown(f"- run_id: `{self.run_id}`"))
        current.card.append(Markdown(f"- dataset/split: `{self.dataset}/{self.split}`"))
        current.card.append(Markdown(f"- embed model: `{self.model}`"))

        if metrics_path.exists():
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            current.card.append(Markdown("## metrics.json"))
            current.card.append(Markdown("```json\n" + json.dumps(m, indent=2) + "\n```"))
        else:
            current.card.append(Markdown("## metrics.json\n**MISSING**"))

        if latency_path.exists():
            l = json.loads(latency_path.read_text(encoding="utf-8"))
            current.card.append(Markdown("## latency.json"))
            current.card.append(Markdown("```json\n" + json.dumps(l, indent=2) + "\n```"))
        else:
            current.card.append(Markdown("## latency.json\n**MISSING**"))

        if latency_txt.exists():
            current.card.append(Markdown("## latency_bench.txt (head)"))
            current.card.append(Markdown("```text\n" + latency_txt.read_text(encoding="utf-8")[:8000] + "\n```"))

        current.card.append(Markdown(f"## Run artifacts\nSaved to: `reports/runs/{self.run_id}/`"))

        self.next(self.end)

    @step
    def end(self):
        print("[DONE] Flow completed.", flush=True)


if __name__ == "__main__":
    StreamingSearchLTRFlow()

PYTHONPATH=src
DATASET=configs/dataset.yaml
EVAL=configs/eval.yaml
SERVE=configs/serve.yaml
GATES=configs/gates.yaml

.PHONY: help setup lint format test data index-bm25 index-faiss eval serve repro clean

help:
	@echo "Targets:"
	@echo "  setup       - create venv + install deps (uv)"
	@echo "  lint        - ruff check"
	@echo "  format      - ruff format"
	@echo "  test        - pytest"
	@echo "  data        - build processed dataset"
	@echo "  index-bm25  - build BM25 artifacts"
	@echo "  index-faiss - build dense embeddings (+ faiss index) artifacts"
	@echo "  eval        - run offline evaluation"
	@echo "  repro       - end-to-end run -> reports/<run_id>/ and reports/latest/"
	@echo "  serve       - run FastAPI server"
	@echo "  clean       - remove caches (keeps venv + data)"

setup:
	uv venv
	uv sync

lint:
	PYTHONPATH=$(PYTHONPATH) uv run ruff check src tests

format:
	PYTHONPATH=$(PYTHONPATH) uv run ruff format src tests

test:
	PYTHONPATH=$(PYTHONPATH) uv run pytest

data:
	PYTHONPATH=$(PYTHONPATH) uv run python -m dataio.build_dataset --config $(DATASET)

index-bm25:
	PYTHONPATH=$(PYTHONPATH) uv run python -m retrieval.bm25_index --config $(DATASET)

index-faiss:
	PYTHONPATH=$(PYTHONPATH) uv run python -m retrieval.embed_index --config $(DATASET)

eval:
	PYTHONPATH=$(PYTHONPATH) uv run python -m eval.evaluate --config $(EVAL)

repro:
	PYTHONPATH=$(PYTHONPATH) uv run python -m pipelines.repro --dataset $(DATASET) --eval $(EVAL)

serve:
	PYTHONPATH=$(PYTHONPATH) uv run python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__

train-ltr:
	PYTHONPATH=$(PYTHONPATH) uv run python -m ranking.ltr_train --config configs/train.yaml


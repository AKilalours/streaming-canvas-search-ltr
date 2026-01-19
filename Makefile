PYTHONPATH=src
DATASET=configs/dataset.yaml
TRAIN=configs/train.yaml
EVAL=configs/eval.yaml
SERVE=configs/serve.yaml
GATES=configs/gates.yaml

.PHONY: help setup lint test data index-bm25 eval serve repro gates clean

help:
	@echo "Targets:"
	@echo "  setup      - create venv + install deps (uv)"
	@echo "  lint       - ruff check"
	@echo "  test       - pytest"
	@echo "  data       - download + build processed dataset"
	@echo "  index-bm25 - build BM25 artifacts"
	@echo "  eval       - run offline evaluation (BM25 for now)"
	@echo "  serve      - run FastAPI server"
	@echo "  repro      - end-to-end run producing reports/<run_id>/"
	@echo "  gates      - run regression gates on a run folder"
	@echo "  clean      - remove local caches (keeps venv)"

setup:
	uv venv
	uv sync --dev

lint:
	PYTHONPATH=$(PYTHONPATH) uv run ruff check src tests


test:

	PYTHONPATH=$(PYTHONPATH) uv run pytest

data:
	PYTHONPATH=$(PYTHONPATH) uv run python -m dataio.build_dataset --config $(DATASET)

index-bm25:
	PYTHONPATH=$(PYTHONPATH) uv run python -m retrieval.bm25_index --config $(DATASET)

eval:
	PYTHONPATH=$(PYTHONPATH) uv run python -m eval.evaluate --config $(EVAL)

serve:
	PYTHONPATH=$(PYTHONPATH) uv run python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

repro:
	PYTHONPATH=$(PYTHONPATH) uv run python -m pipelines.repro --dataset $(DATASET) --eval $(EVAL) --gates $(GATES)

gates:
	PYTHONPATH=$(PYTHONPATH) uv run python -m pipelines.gates --gates $(GATES) --run_dir $(RUN_DIR)

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__
format:
	PYTHONPATH=$(PYTHONPATH) uv run ruff format src tests


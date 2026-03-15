PYTHONPATH ?= src
HOST      ?= 127.0.0.1
PORT      ?= 8000
WORKERS   ?= 4

EVAL         ?= configs/eval_movielens.yaml
MULTI_EVAL   ?= configs/multi_eval.yaml
GATES        ?= configs/gates.yaml
DATASET      ?= movielens
RPS          ?= 100
LOAD_DURATION ?= 60

.PHONY: all setup lint format test \
        serve kill_port \
        build up down logs restart \
        eval multi_eval latency_search latency_answer \
        gates gate_baseline \
        drift monitor_open \
        retrain flow_train flow_production \
        failure_report shadow_ab \
        reports_latest clean help \
        phase3_serendipity phase5_feedback phase6_load_test \
        rollback health_deep

# ── Dev Setup ──────────────────────────────────────────────────────────────────

setup:
	uv venv && uv sync

lint:
	PYTHONPATH=$(PYTHONPATH) uv run ruff check src tests

format:
	PYTHONPATH=$(PYTHONPATH) uv run ruff format src tests

test:
	PYTHONPATH=$(PYTHONPATH) uv run pytest -q

kill_port:
	@PID=$$(lsof -t -iTCP:$(PORT) -sTCP:LISTEN 2>/dev/null || true); \
	if [ -n "$$PID" ]; then echo "Killing PID $$PID on :$(PORT)"; kill -9 $$PID; fi; \
	echo "Port $(PORT) is free ✅"

serve: kill_port
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
	TORCH_NUM_THREADS=1 TORCH_NUM_INTEROP_THREADS=1 \
	TOKENIZERS_PARALLELISM=false \
	PYTHONPATH=$(PYTHONPATH) uv run uvicorn app.main:app \
	  --host $(HOST) --port $(PORT) --workers $(WORKERS) --reload

# ── Docker ─────────────────────────────────────────────────────────────────────

build:
	docker compose build api dashboard

up: build
	docker compose up -d
	@echo ""
	@echo "  Search API  ->  http://localhost:8000"
	@echo "  Swagger     ->  http://localhost:8000/docs"
	@echo "  Demo UI     ->  http://localhost:8000/demo"
	@echo "  Dashboard   ->  http://localhost:8501"
	@echo "  Grafana     ->  http://localhost:3000  (admin / searchltr2026)"
	@echo "  Prometheus  ->  http://localhost:9090"
	@echo "  MinIO       ->  http://localhost:9001  (minioadmin / minioadmin)"
	@echo ""
	@echo "  Phase 3 ──  http://localhost:8000/feed/diverse"
	@echo "  Phase 3 ──  http://localhost:8000/serendipity/score?q=action"
	@echo "  Phase 5 ──  http://localhost:8000/feed/household"
	@echo "  Phase 5 ──  http://localhost:8000/personalization/explain"
	@echo "  Phase 6 ──  http://localhost:8000/health/deep"
	@echo "  Phase 6 ──  http://localhost:8000/rate_limits"

down:
	docker compose down

restart:
	docker compose restart api

logs:
	docker compose logs -f api

# ── Eval Pipeline ──────────────────────────────────────────────────────────────

reports_latest:
	mkdir -p reports/latest reports/reference
	touch reports/latest/.gitkeep reports/reference/.gitkeep

eval: reports_latest
	@echo "Running eval inside api container (artifacts live there)..."
	docker compose exec -T api uv run python -m eval.evaluate --config $(EVAL)
	@echo "Copying results out of container..."
	docker compose cp api:/app/reports/latest/. reports/latest/

multi_eval:
	docker compose exec -T api uv run python -m pipelines.multi_eval --config $(MULTI_EVAL)

latency_search: reports_latest
	@echo "Running latency bench against live API..."
	PYTHONPATH=$(PYTHONPATH) uv run python -m eval.latency_bench \
	  --endpoint /search --n 200 --concurrency 20 --out reports/latest/latency.json \
	  --base http://localhost:8000

latency_answer: reports_latest
	PYTHONPATH=$(PYTHONPATH) uv run python -m eval.latency_bench \
	  --endpoint /answer --n 30 --concurrency 2 --timeout 240 \
	  --out reports/latest/latency_answer.json \
	  --base http://localhost:8000

gates: reports_latest
	@echo "Running quality gates..."
	PYTHONPATH=$(PYTHONPATH) uv run python -m pipelines.gates \
	  --gates $(GATES) --run_dir reports/latest

gate_baseline:
	mkdir -p reports/reference
	cp reports/latest/metrics.json  reports/reference/metrics.json
	cp reports/latest/latency.json  reports/reference/latency.json 2>/dev/null || true
	@echo "Reference baseline updated ✅"

# ── Drift & Monitoring ─────────────────────────────────────────────────────────

drift: reports_latest
	@echo "Running drift detection on host..."
	@if [ ! -f reports/reference/metrics.json ]; then \
	  echo "[drift] No reference baseline yet — copying latest as reference and skipping drift check."; \
	  cp reports/latest/metrics.json reports/reference/metrics.json; \
	  echo '{"status":"ok","note":"first run — no prior reference"}' > reports/latest/drift_report.json; \
	else \
	  PYTHONPATH=$(PYTHONPATH) uv run python scripts/drift_monitor.py \
	    --latest   reports/latest/metrics.json \
	    --reference reports/reference/metrics.json \
	    --out       reports/latest/drift_report.json \
	    --ndcg_drop 0.03 \
	    --p99_max_ms 300; \
	fi

monitor_open:
	open http://localhost:3000 || xdg-open http://localhost:3000

# ── Training Flows ─────────────────────────────────────────────────────────────

retrain:
	PYTHONPATH=$(PYTHONPATH) python flows/streaming_search_flow.py run \
	  --dataset $(DATASET)

flow_train:
	PYTHONPATH=$(PYTHONPATH) python flows/train_ltr.py run \
	  --dataset $(DATASET)

# Phase 4: Production training flow with gates + auto-promotion
flow_production:
	PYTHONPATH=$(PYTHONPATH) python flows/train_ltr_production.py run \
	  --dataset $(DATASET)

failure_report: reports_latest
	PYTHONPATH=$(PYTHONPATH) uv run python -m eval.failure_analysis \
	  --config $(EVAL) --out reports/latest/failure_cases.md

shadow_ab: reports_latest
	mkdir -p reports/latest_eval
	PYTHONPATH=$(PYTHONPATH) uv run python scripts/shadow_ab_eval.py \
	  --api http://localhost:8000 \
	  --out reports/latest_eval/shadow_ab.json

# ── Phase 3: Serendipity ───────────────────────────────────────────────────────

phase3_serendipity:
	@echo "Testing serendipity endpoints..."
	curl -s "http://localhost:8000/serendipity/score?q=action&k=10" | python3 -m json.tool
	@echo ""
	curl -s "http://localhost:8000/feed/diverse?profile=chrisen&epsilon=0.20&k=6" | python3 -m json.tool | head -40

# ── Phase 5: Feedback / Personalization ───────────────────────────────────────

phase5_feedback:
	@echo "Testing feedback collection..."
	curl -s -X POST "http://localhost:8000/feedback?user_id=chrisen&doc_id=1&event_type=click&query=action&rank=1" | python3 -m json.tool
	curl -s "http://localhost:8000/feedback/export?user_id=chrisen&min_label=1" | python3 -m json.tool | head -20

# ── Phase 6: Production Load Test ──────────────────────────────────────────────

phase6_load_test: reports_latest
	PYTHONPATH=$(PYTHONPATH) uv run python scripts/load_test.py \
	  --url http://localhost:8000 \
	  --rps $(RPS) \
	  --duration $(LOAD_DURATION) \
	  --method hybrid \
	  --out reports/latest/load_test.json
	@echo "Load test results → reports/latest/load_test.json"

health_deep:
	curl -s http://localhost:8000/health/deep | python3 -m json.tool

# Phase 6: Manual rollback (requires ADMIN_TOKEN env var)
rollback:
	@if [ -z "$$ADMIN_TOKEN" ]; then echo "Set ADMIN_TOKEN env var first"; exit 1; fi
	curl -s -X POST http://localhost:8000/admin/rollback \
	  -H "X-Admin-Token: $$ADMIN_TOKEN" | python3 -m json.tool

# ── Full Production Runbook ─────────────────────────────────────────────────────

# Run the complete production graduation sequence:
#   eval → latency → gates → drift → load_test → shadow_ab → gate_baseline
# Full Production Graduation Runbook:
#   rebuild → eval → latency → gates → drift → load_test → gate_baseline
production_graduate: build
	@echo "Pre-flight: killing any rogue processes on :8000..."
	@lsof -ti :8000 2>/dev/null | xargs kill -9 2>/dev/null || true
	@sleep 1
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  🎓 Production Graduation Runbook"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Step 1/6: Restarting API with new image..."
	docker compose up -d api
	@echo "Waiting for API to be healthy..."
	@sleep 15
	@echo ""
	@echo "Step 2/6: Running offline evaluation..."
	$(MAKE) eval
	@echo ""
	@echo "Step 3/6: Running latency benchmark..."
	$(MAKE) latency_search
	@echo ""
	@echo "Step 4/6: Quality gates check..."
	$(MAKE) gates
	@echo ""
	@echo "Step 5/6: Drift detection..."
	$(MAKE) drift
	@echo ""
	@echo "Step 6/6: Promoting to reference baseline..."
	$(MAKE) gate_baseline
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  ✅ Production graduation complete!"
	@echo "  Deep health: http://localhost:8000/health/deep"
	@echo "  Metrics:     http://localhost:8000/metrics/latest"
	@echo "  Grafana:     http://localhost:3000"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""

# ── Misc ───────────────────────────────────────────────────────────────────────

demo_url:
	@echo "Demo UI  : http://$(HOST):$(PORT)/demo"
	@echo "Swagger  : http://$(HOST):$(PORT)/docs"
	@echo "Health   : http://$(HOST):$(PORT)/health"
	@echo "Deep Health: http://$(HOST):$(PORT)/health/deep"
	@echo "Metrics  : http://$(HOST):$(PORT)/metrics"
	@echo "Rate Limits: http://$(HOST):$(PORT)/rate_limits"
	@echo "Serendipity: http://$(HOST):$(PORT)/serendipity/score?q=action"
	@echo "Household: http://$(HOST):$(PORT)/feed/household"

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ \
	  reports/latest/*.json reports/latest/*.md

help:
	@echo ""
	@echo "  Dev:          setup lint format test serve"
	@echo "  Docker:       build up down restart logs"
	@echo "  Eval:         eval multi_eval latency_search latency_answer"
	@echo "  Gates:        gates gate_baseline gate_baseline"
	@echo "  Drift:        drift monitor_open"
	@echo "  Training:     retrain flow_train flow_production"
	@echo "  Phase 3:      phase3_serendipity"
	@echo "  Phase 5:      phase5_feedback"
	@echo "  Phase 6:      phase6_load_test health_deep rollback"
	@echo "  Full run:     production_graduate"
	@echo ""

# ── Real Infrastructure Targets ────────────────────────────────────────────

scale_bench_quick: ## Quick scale test: 10/50/100 concurrent users
	@echo "Running quick scale benchmark (100 concurrent)..."
	PYTHONPATH=src uv run python scripts/scale_bench.py --quick --base-url http://localhost:8000

scale_bench: ## Full scale test: up to 1000 concurrent users
	@echo "Running full scale benchmark (up to 1000 concurrent)..."
	PYTHONPATH=src uv run python scripts/scale_bench.py --max-concurrency 1000 --base-url http://localhost:8000

graph_build: ## Build knowledge graph from MovieLens corpus
	@echo "Building knowledge graph..."
	curl -s "http://localhost:8000/graph/build" | python3 -m json.tool

graph_test: ## Test knowledge graph expansion
	@echo "Testing graph expansion for 'action thriller'..."
	curl -s "http://localhost:8000/graph/expand?q=action+thriller&k=10" | python3 -m json.tool

vlm_test: ## Test VLM artwork endpoint (requires OPENAI_API_KEY + TMDB_API_KEY)
	@echo "Testing VLM artwork (doc_id=1)..."
	curl -s "http://localhost:8000/vlm/artwork?doc_id=1&title=The+Dark+Knight&year=2008" | python3 -m json.tool

propensity_stats: ## Show propensity log statistics
	curl -s "http://localhost:8000/impression/stats" | python3 -m json.tool

vlm_setup: ## Instructions to enable real VLM
	@echo ""
	@echo "To enable real GPT-4V artwork analysis:"
	@echo "  1. Get a free TMDB API key: https://www.themoviedb.org/settings/api"
	@echo "  2. Get an OpenAI API key:   https://platform.openai.com/api-keys"
	@echo "  3. Add to docker-compose.yml under api > environment:"
	@echo "       OPENAI_API_KEY: your-key-here"
	@echo "       TMDB_API_KEY:   your-key-here"
	@echo "  4. Restart: docker compose restart api"
	@echo ""

# Re-run full evaluation with candidate_k=1000 (fixes recall metrics)
eval_full:
	docker compose exec -e PYTHONPATH=/app/src api /app/.venv/bin/python /app/src/eval/evaluate.py --config /app/configs/eval_movielens.yaml
	@echo "Eval complete. Check reports/latest/metrics.json"

eval_quick:
	curl -s "http://localhost:8000/eval/comprehensive" | python3 -m json.tool | head -60

# Show all gate results  
eval_gates:
	curl -s http://localhost:8000/eval/comprehensive | python3 -m json.tool | grep -A3 "gates"

# Show slice analysis
eval_slices:
	curl -s http://localhost:8000/eval/slice_analysis | python3 -m json.tool

eval_full_v2:
	docker compose exec -e PYTHONPATH=/app/src api /app/.venv/bin/python3.11 /app/src/eval/evaluate.py --config /app/configs/eval_movielens.yaml
	@echo "Eval complete — check reports/latest/metrics.json"

eval_quick:
	curl -s "http://localhost:8000/eval/comprehensive" | python3 -m json.tool
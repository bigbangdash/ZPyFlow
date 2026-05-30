# ZPyFlow Makefile
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   make help          — show this help
#   make build         — build the Rust extension locally (requires maturin)
#   make test          — run Python tests locally
#   make bench         — run all Python benchmarks locally
#
#   make dc-build      — build in Docker
#   make dc-test       — run tests in Docker
#   make dc-bench      — run all benchmarks in Docker
#   make dc-shell      — interactive Docker shell
#   make dc-clean      — remove Docker volumes (reset build cache)
#
# Docker Compose targets mirror the local targets with the dc- prefix.

.DEFAULT_GOAL := help
.PHONY: help build test bench bench-rust lint fmt audit clean \
        docs docs-serve docs-deploy \
        dc-build dc-test dc-bench dc-bench-rust dc-shell dc-clean \
        dc-bench-filter dc-bench-chained dc-bench-numpy dc-bench-agg dc-bench-objects \
        dc-bench-vector dc-bench-ml dc-bench-etl dc-bench-fraud dc-bench-groupby dc-bench-null

# ── Colour output ─────────────────────────────────────────────────────────────
CYAN  := \033[36m
RESET := \033[0m

help:  ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make $(CYAN)<target>$(RESET)\n\nTargets:\n"} \
	      /^[a-zA-Z_-]+:.*?##/ { printf "  $(CYAN)%-22s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)


# ── Local targets (requires Rust + maturin installed on host) ─────────────────

build:  ## Build the Rust extension in release mode
	maturin develop --release

build-debug:  ## Build in debug mode (faster compile, slower runtime)
	maturin develop

test: build  ## Run Python unit tests
	pytest tests/ -v --tb=short

test-fast: build  ## Run tests without verbose output
	pytest tests/ -q

bench: build  ## Run all Python benchmark suites
	python sandbox/benchmark/run.py --suite all

bench-filter: build  ## Run filter benchmarks
	python sandbox/benchmark/run.py --suite filter

bench-chained: build  ## Run chained pipeline benchmarks
	python sandbox/benchmark/run.py --suite chained

bench-numpy: build  ## Run numpy comparison benchmarks
	python sandbox/benchmark/run.py --suite vs_numpy

bench-agg: build  ## Run aggregation benchmarks
	python sandbox/benchmark/run.py --suite aggregation

bench-objects: build  ## Run Python object benchmarks
	python sandbox/benchmark/run.py --suite objects

bench-rust:  ## Run Criterion (Rust) benchmarks
	cargo bench --bench pipeline

bench-rust-simd:  ## Run SIMD selectivity benchmarks
	cargo bench --bench simd_filter

bench-save: build  ## Save current results as baseline
	SUITE=$${SUITE:-filter} python sandbox/benchmark/run.py --suite $${SUITE} --save

bench-compare: build  ## Compare against saved baseline (fails if >10% regression)
	SUITE=$${SUITE:-filter} python sandbox/benchmark/run.py --suite $${SUITE} --compare

lint:  ## Run Rust linter
	cargo clippy -- -D warnings

audit:  ## Scan dependencies for CVEs (requires: cargo install cargo-audit)
	cargo audit

fmt:  ## Format Rust and Python code
	cargo fmt
	@command -v ruff >/dev/null 2>&1 && ruff format . || true

docs:  ## Build documentation site locally (requires mkdocs-material)
	mkdocs build --config-file mkdocs.yml

docs-serve:  ## Serve documentation site locally on http://localhost:8000
	mkdocs serve --config-file mkdocs.yml

docs-deploy:  ## Deploy documentation site to GitHub Pages
	mkdocs gh-deploy --force --config-file mkdocs.yml

clean:  ## Remove local build artifacts
	cargo clean
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache dist build *.egg-info site/site


# ── Docker Compose targets ────────────────────────────────────────────────────

dc-build:  ## [Docker] Build the Rust extension
	docker compose run --rm build

dc-test:  ## [Docker] Run Python unit tests
	docker compose run --rm test

dc-test-k:  ## [Docker] Run specific tests  (make dc-test-k K="f64 and not lambda")
	docker compose run --rm test pytest tests/ -v -k "$(K)"

dc-bench:  ## [Docker] Run all Python benchmarks
	docker compose run --rm bench

dc-bench-filter:  ## [Docker] Run filter benchmarks
	docker compose run --rm bench-suite

dc-bench-chained:  ## [Docker] Run chained pipeline benchmarks
	SUITE=chained docker compose run --rm bench-suite

dc-bench-numpy:  ## [Docker] Run numpy comparison benchmarks
	SUITE=vs_numpy docker compose run --rm bench-suite

dc-bench-agg:  ## [Docker] Run aggregation benchmarks
	SUITE=aggregation docker compose run --rm bench-suite

dc-bench-objects:  ## [Docker] Run Python object benchmarks
	SUITE=objects docker compose run --rm bench-suite

dc-bench-vector:  ## [Docker] Run vector search benchmarks
	SUITE=vector_search docker compose run --rm bench-suite

dc-bench-ml:  ## [Docker] Run ML feature preprocessing benchmarks
	SUITE=ml_feature docker compose run --rm bench-suite

dc-bench-etl:  ## [Docker] Run ETL multi-stat pipeline benchmarks
	SUITE=etl docker compose run --rm bench-suite

dc-bench-fraud:  ## [Docker] Run fraud/risk scoring benchmarks
	SUITE=fraud docker compose run --rm bench-suite

dc-bench-groupby:  ## [Docker] Run GroupBy and pagination benchmarks
	SUITE=groupby docker compose run --rm bench-suite

dc-bench-null:  ## [Docker] Run null-mixed list benchmarks
	SUITE=null docker compose run --rm bench-suite

dc-bench-rust:  ## [Docker] Run Criterion benchmarks
	docker compose run --rm bench-rust

dc-bench-save:  ## [Docker] Save benchmark baseline
	SUITE=$${SUITE:-filter} docker compose run --rm bench-save

dc-bench-compare:  ## [Docker] Compare against saved baseline
	SUITE=$${SUITE:-filter} docker compose run --rm bench-compare

dc-shell:  ## [Docker] Open interactive development shell
	docker compose run --rm shell

dc-image:  ## [Docker] Build the shared zpyflow-dev image (run after Dockerfile changes)
	docker compose build dev

dc-clean:  ## [Docker] Remove named volumes (reset Cargo and target cache)
	docker compose down -v
	@echo "Cargo cache and target volume removed."

dc-ps:  ## [Docker] Show running containers
	docker compose ps

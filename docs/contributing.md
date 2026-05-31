# Contributing to ZPyFlow

Thank you for your interest in contributing!

---

## Development setup

All development tasks run inside Docker — no local Rust or Python setup needed
beyond Docker itself.

### Quick start

```bash
# First time: build the image (takes a few minutes — downloads Rust toolchain)
make dc-image

# Edit Rust source, then test
make dc-test

# Measure performance against all libraries
make dc-bench-agg

# Open a shell to inspect the built extension
make dc-shell
(inside) python -c "from zpyflow import Query, col; print(Query([1,2,3]).filter(col>1).to_list())"
```

### Available Make commands

| Command | What it does |
|---|---|
| `make dc-test` | Build + run Python unit tests |
| `make dc-test-k K="f64"` | Run tests matching keyword `f64` |
| `make dc-bench` | Build + run **all** benchmark suites |
| `make dc-bench-filter` | filter benchmarks only |
| `make dc-bench-chained` | chained pipeline benchmarks only |
| `make dc-bench-agg` | aggregation benchmarks (vs numpy / pandas / polars) |
| `make dc-bench-numpy` | numpy comparison benchmarks |
| `make dc-bench-objects` | Python object (dict / dataclass) benchmarks |
| `make dc-bench-vector` | vector search — top-K early stopping |
| `make dc-bench-ml` | ML feature preprocessing pipeline |
| `make dc-bench-etl` | ETL multi-stat aggregation (vs Polars / Pandas) |
| `make dc-bench-fraud` | fraud / risk scoring — review queue, exposure sum |
| `make dc-bench-groupby` | GroupBy and pagination (object path) |
| `make dc-bench-null` | null-mixed list benchmarks (None handling) |
| `make dc-bench-rust` | Rust (Criterion) benchmarks |
| `make dc-bench-save` | Save current results as baseline |
| `make dc-bench-compare` | Compare against saved baseline (fails on >10% regression) |
| `make dc-shell` | Interactive shell inside the container |
| `make dc-image` | Rebuild the Docker image |
| `make dc-clean` | Remove named volumes (reset Cargo + target cache) |

### Local build (requires Rust + maturin)

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Set up Python venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Build and test
maturin develop --release
pytest tests/ -q
```

See also: `make build`, `make test`, `make bench-rust`, `make lint`, `make fmt`.

---

## Project structure

```
src/           Rust source (PyO3 extension)
zpyflow/       Python package
tests/         Python integration tests
benches/       Rust Criterion benchmarks
sandbox/       Python benchmark suites
examples/      Standalone usage examples
docs/          Documentation
site/docs/     MkDocs site source
```

---

## Running benchmarks

```bash
# Save current results as baseline, make changes, then compare
make dc-bench-save
# ... edit code ...
make dc-bench-compare     # fails if mean regresses by >10%
```

---

## Submitting changes

1. Fork the repository and create a feature branch
2. Run `make dc-test` — all tests must pass
3. Open a pull request against `main`

Please keep PRs focused. Large refactors benefit from a spec task discussion first
(see `specs/` directory).

---

## Releasing

### Pre-release checks

```bash
# Build + test + type-check
docker compose run --rm test sh -c \
  'maturin develop --release && pytest tests/ -q && mypy zpyflow tests/typing_examples.py'

# Build wheel and smoke-test in a clean venv
docker compose run --rm test sh -c "
  maturin build --release &&
  python3 -m venv /tmp/check &&
  /tmp/check/bin/pip install -q /app/target/wheels/zpyflow-*.whl &&
  /tmp/check/bin/python -c \"
from zpyflow import Query, col
assert Query([1,2,3]).filter(col>1).to_list() == [2,3]
print('OK')
\"
"
```

### TestPyPI dry run (optional)

```bash
maturin publish --repository testpypi --skip-existing
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ zpyflow
```

### Publish to PyPI

```bash
# 1. Bump version in pyproject.toml and Cargo.toml
# 2. Commit and push to main
git tag v0.1.1
git push origin v0.1.1
# 3. Create a GitHub Release for the tag
#    → release.yml builds Linux + macOS wheels and publishes to PyPI
```

### Package metadata

- Package name: `zpyflow`
- Python support: CPython 3.10+ (abi3 wheel)
- License: MIT
- Extras: `zpyflow[numpy]`, `zpyflow[arrow]`, `zpyflow[dev]`

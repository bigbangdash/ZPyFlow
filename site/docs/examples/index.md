# Examples

Each example is self-contained and runnable after building the extension.

```bash
maturin develop --release
pip install numpy pandas pyarrow
```

## API & integration examples

| Page | What it covers |
|---|---|
| [Numeric DSL](numeric.md) | `filter`, `map`, `take` on plain lists and numpy arrays |
| [NumPy integration](numpy.md) | `from_numpy`, buffer protocol, speed comparison |
| [Object field DSL](field_dsl.md) | `field()` expressions, `preload()`, dict records |
| [GroupBy & group_agg](groupby.md) | `GroupBy`, `group_agg`, aggregation specs |
| [Adapters](adapters.md) | `from_csv`, `from_json_lines`, `from_arrow` |

## Runnable example files

The [`examples/`](../../../../examples/) directory contains standalone Python scripts:

| File | Scenario |
|---|---|
| `01_basic_numeric.py` | Plain list pipelines, DSL vs lambda |
| `02_numpy_integration.py` | numpy float64/int64 arrays |
| `03_pandas_integration.py` | Column preprocessing, row filtering |
| `04_dataclasses.py` | Dataclasses, Pydantic, domain objects |
| `05_log_processing.py` | Access logs, error rates, latency stats |
| `06_etl_pipeline.py` | Extract → Transform → Load with CSV/JSON |
| `07_ai_embeddings.py` | Vector search, batch inference, top-p sampling |
| `08_parallel_and_performance.py` | Parallel vs single-thread, memory |
| `usecase_fraud_detection.py` | Risk score thresholding, review queue |
| `usecase_search_recommendation.py` | ANN post-filtering, top-K candidate retrieval |
| `usecase_ml_feature_preprocessing.py` | Outlier removal + normalization |

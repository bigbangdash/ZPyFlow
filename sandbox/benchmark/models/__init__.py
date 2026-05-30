from .generators import (
    float_list, float_array, int_list,
    positive_float_list, half_positive_float_list, skewed_float_list,
    nullable_float_list,
    log_records, log_dicts, products,
    similarity_scores, embeddings,
    SIZES,
)

import tracemalloc


def measure_peak_kb(fn):
    """Peak Python-heap allocation for one fn() call, in KB.

    Uses tracemalloc — Rust-side allocations (PyO3 Vec buffers) are NOT counted.
    For ZPyFlow DSL this only measures the final Python list from to_list();
    for Python native it measures the intermediate list comprehension too.
    """
    tracemalloc.start()
    try:
        fn()
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return peak // 1024

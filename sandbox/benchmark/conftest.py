"""
pytest-benchmark configuration for ZPyFlow sandbox benchmarks.

Mirrors BenchmarkDotNet's job configuration in ZLinq's Program.cs —
controls warmup, iterations, and output format.
"""

import sys
import os

# Make models importable from benchmark files
sys.path.insert(0, os.path.dirname(__file__))


def pytest_configure(config):
    config.addinivalue_line("markers", "simd: SIMD-accelerated benchmarks")
    config.addinivalue_line("markers", "parallel: parallel execution benchmarks")
    config.addinivalue_line("markers", "small_n: small data (< 10K) — expect ZPyFlow overhead")
    config.addinivalue_line("markers", "large_n: large data (> 100K) — ZPyFlow advantage zone")
    config.addinivalue_line("markers", "vs_numpy: direct numpy comparison")
    config.addinivalue_line("markers", "vs_pandas: direct pandas comparison")
    config.addinivalue_line("markers", "objects: Python object (dict/dataclass) path")

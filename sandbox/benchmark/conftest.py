"""
pytest-benchmark configuration for ZPyFlow sandbox benchmarks.

Mirrors BenchmarkDotNet's job configuration in ZLinq's Program.cs —
controls warmup, iterations, and output format.
"""

import re as _re
import sys
import os

# Make models importable from benchmark files
sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Benchmark table colorization
# ---------------------------------------------------------------------------
_R = "\033[0m"           # reset
_BENCH_COLORS = {
    "zpf":    "\033[36m",   # cyan    — ZPyFlow
    "numpy":  "\033[33m",   # yellow  — NumPy
    "polars": "\033[35m",   # magenta — Polars
    "pandas": "\033[34m",   # blue    — Pandas
    "python": "\033[32m",   # green   — Python native
}

# Ordered: zpyflow before numpy so "test_zpyflow_from_numpy" → cyan, not yellow.
_COLOR_RULES = [
    (_re.compile(r"zpyflow",    _re.I), _BENCH_COLORS["zpf"]),
    (_re.compile(r"from_numpy", _re.I), _BENCH_COLORS["zpf"]),
    (_re.compile(r"from_arrow", _re.I), _BENCH_COLORS["zpf"]),
    (_re.compile(r"polars",     _re.I), _BENCH_COLORS["polars"]),
    (_re.compile(r"pandas",     _re.I), _BENCH_COLORS["pandas"]),
    (_re.compile(r"numpy",      _re.I), _BENCH_COLORS["numpy"]),
    (_re.compile(r"python",     _re.I), _BENCH_COLORS["python"]),
]

# Match only benchmark data rows (not headers / separator lines).
# Covers: test_xxx  |  "zpyflow t=100"  |  "python listcomp 50% pass"
_ROW_RE = _re.compile(r"^\s*(test_\w|zpyflow\s|python\s)", _re.I)


def _colorize(line: str) -> str:
    if not _ROW_RE.match(line):
        return line
    for pattern, color in _COLOR_RULES:
        if pattern.search(line):
            return f"{color}{line}{_R}"
    return line


def pytest_sessionstart(session):
    """Wrap the terminal writer so benchmark rows are color-coded by library."""
    try:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is None or not hasattr(reporter, "_tw"):
            return
        tw = reporter._tw
        orig = tw.write_line
        tw.write_line = lambda s="", **kw: orig(
            _colorize(s) if isinstance(s, str) else s, **kw
        )
    except Exception:
        pass


def pytest_terminal_summary(terminalreporter, exitstatus):
    """Print a color legend after benchmark tables."""
    bench_session = getattr(terminalreporter.config, "_benchmark_session", None)
    if not bench_session or not getattr(bench_session, "benchmarks", None):
        return
    tw = terminalreporter._tw
    tw.write_line("")
    legend = "  ".join(
        f"{color}■ {label}{_R}"
        for label, color in [
            ("ZPyFlow",       _BENCH_COLORS["zpf"]),
            ("NumPy",         _BENCH_COLORS["numpy"]),
            ("Polars",        _BENCH_COLORS["polars"]),
            ("Pandas",        _BENCH_COLORS["pandas"]),
            ("Python native", _BENCH_COLORS["python"]),
        ]
    )
    tw.write_line(f"Color key: {legend}")


# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line("markers", "simd: SIMD-accelerated benchmarks")
    config.addinivalue_line("markers", "parallel: parallel execution benchmarks")
    config.addinivalue_line("markers", "small_n: small data (< 10K) — expect ZPyFlow overhead")
    config.addinivalue_line("markers", "large_n: large data (> 100K) — ZPyFlow advantage zone")
    config.addinivalue_line("markers", "vs_numpy: direct numpy comparison")
    config.addinivalue_line("markers", "vs_pandas: direct pandas comparison")
    config.addinivalue_line("markers", "objects: Python object (dict/dataclass) path")

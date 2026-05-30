# generators.py — reproducible test data for benchmarks
# Mirrors ZLinq's sandbox/Benchmark/Models/ and TestData/

from __future__ import annotations

import random
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Numeric arrays
# ---------------------------------------------------------------------------

def float_list(n: int, seed: int = 42) -> list[float]:
    """Uniform float64 values in [-100, 100]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-100, 100, n).tolist()


def float_array(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-100, 100, n).astype(np.float64)


def int_list(n: int, seed: int = 42) -> list[int]:
    rng = np.random.default_rng(seed)
    return rng.integers(-1_000_000, 1_000_000, n).tolist()


def positive_float_list(n: int, seed: int = 42) -> list[float]:
    """All positive — worst case for filter(col > 0): 100% pass rate."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.01, 100, n).tolist()


def half_positive_float_list(n: int, seed: int = 42) -> list[float]:
    """~50% pass filter(col > 0) — best case for SIMD branch prediction."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).tolist()


def skewed_float_list(n: int, seed: int = 42) -> list[float]:
    """Log-normal — realistic for latency / price / revenue data."""
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=3.0, sigma=1.5, size=n).tolist()


def nullable_float_list(n: int, null_rate: float = 0.1, seed: int = 42) -> list[float | None]:
    """~50% positive floats with null_rate fraction replaced by None.

    Models a real-world column where some rows have missing values.
    Default: 10% nulls scattered uniformly (every 10th element).
    Non-null elements follow standard_normal so ~50% pass filter(x > 0).
    """
    rng = np.random.default_rng(seed)
    data: list[float | None] = rng.standard_normal(n).tolist()
    step = max(1, int(1 / null_rate))
    for i in range(0, n, step):
        data[i] = None
    return data


# ---------------------------------------------------------------------------
# Standard data sizes  (mirrors ZLinq's N parameter)
# ---------------------------------------------------------------------------

SIZES = {
    "xs":  100,
    "s":   1_000,
    "m":   10_000,
    "l":   100_000,
    "xl":  1_000_000,
    "xxl": 10_000_000,
}


# ---------------------------------------------------------------------------
# Structured records
# ---------------------------------------------------------------------------

@dataclass
class LogRecord:
    ts: int
    level: str
    status: int
    path: str
    latency_ms: float
    user_id: int | None
    bytes_sent: int


def log_records(n: int, seed: int = 42) -> list[LogRecord]:
    rng = random.Random(seed)
    paths   = ["/api/users", "/api/orders", "/api/items", "/health", "/auth"]
    levels  = ["INFO"] * 7 + ["WARN"] * 2 + ["ERROR"]
    return [
        LogRecord(
            ts          = 1_700_000_000 + i,
            level       = rng.choice(levels),
            status      = rng.choice([200, 200, 200, 201, 400, 429, 500]),
            path        = rng.choice(paths),
            latency_ms  = math.exp(rng.gauss(3.5, 1.2)),
            user_id     = rng.randint(1, 50_000) if rng.random() > 0.02 else None,
            bytes_sent  = rng.randint(200, 50_000),
        )
        for i in range(n)
    ]


def log_dicts(n: int, seed: int = 42) -> list[dict]:
    return [vars(r) for r in log_records(n, seed)]


@dataclass
class Product:
    product_id: str
    name: str
    price: float
    category: str
    stock: int
    rating: float


def products(n: int, seed: int = 42) -> list[Product]:
    rng = random.Random(seed)
    cats = ["electronics", "clothing", "food", "books", "toys"]
    return [
        Product(
            product_id = f"P{i:06d}",
            name       = f"Product {i}",
            price      = round(rng.uniform(0.5, 999.9), 2),
            category   = rng.choice(cats),
            stock      = rng.randint(0, 500),
            rating     = round(rng.uniform(1.0, 5.0), 1),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Embedding vectors  (AI use case)
# ---------------------------------------------------------------------------

def similarity_scores(n: int, seed: int = 42) -> list[float]:
    """Cosine similarity scores — Beta(2,5) distribution, skewed toward 0."""
    rng = np.random.default_rng(seed)
    return rng.beta(2, 5, n).tolist()


def embeddings(n: int, dim: int = 384, seed: int = 42) -> np.ndarray:
    """L2-normalized embedding vectors."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    return e / norms

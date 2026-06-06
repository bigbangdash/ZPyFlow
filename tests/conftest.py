"""Shared pytest fixtures for ZPyFlow test suite."""

import pytest

try:
    from zpyflow import Query, col, field, AggSpec, agg_count, agg_sum, agg_mean, agg_max, agg_min
    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False


@pytest.fixture
def products():
    return [
        {"name": "apple",  "price": 1.20, "qty": 50,  "active": True},
        {"name": "banana", "price": 0.50, "qty": 100, "active": True},
        {"name": "cherry", "price": 3.00, "qty": 20,  "active": False},
        {"name": "date",   "price": 5.00, "qty": 10,  "active": True},
        {"name": "elderberry", "price": 8.00, "qty": 5, "active": False},
    ]

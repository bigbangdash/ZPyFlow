"""
01_basic_numeric.py
-------------------
Getting started with ZPyFlow on plain Python lists.

Shows the two execution paths:
  - Expression DSL  →  Rust SIMD, GIL released
  - Python lambda   →  GIL held, but still fused (no intermediate lists)
"""

from zpyflow import Query, col

# ------------------------------------------------------------------
# Sample data
# ------------------------------------------------------------------
prices = [12.5, -3.0, 45.0, 0.0, 7.8, -1.2, 99.9, 22.1, 3.3, -0.5]

# ------------------------------------------------------------------
# Expression DSL  (recommended for float / int arrays)
# ------------------------------------------------------------------

# Filter → map → take in one fused pass, zero intermediate lists
discounted = (
    Query(prices)
        .filter(col > 0)          # keep positive prices
        .map(col * 0.9)           # apply 10% discount
        .take(5)                  # first 5 results
        .to_list()
)
print("discounted:", discounted)
# → [11.25, 40.5, 7.02, 89.91, 19.89]

# Aggregations
total   = Query(prices).filter(col > 0).sum()
count   = Query(prices).filter(col > 0).count()
avg     = total / count
maximum = Query(prices).max()
minimum = Query(prices).filter(col > 0).min()

print(f"total={total:.2f}  count={count}  avg={avg:.2f}")
print(f"max={maximum}  min_positive={minimum}")

# Chained DSL operators
result = (
    Query(prices)
        .filter(col.between(0, 50))   # keep values in [0, 50]
        .map(col ** 2)                # square
        .map(col.sqrt())              # back to original (demo)
        .to_list()
)
print("between + pow + sqrt:", [round(x, 4) for x in result])

# ------------------------------------------------------------------
# Python lambda  (use when logic can't be expressed as col DSL)
# ------------------------------------------------------------------

# Custom rounding rule: round to nearest 5
def round_to_5(x: float) -> float:
    return round(x / 5) * 5

rounded = (
    Query(prices)
        .filter(lambda x: x > 0)
        .map(round_to_5)
        .to_list()
)
print("rounded to 5:", rounded)

# any / all
has_negative = Query(prices).any(lambda x: x < 0)   # True
all_positive = Query(prices).all(lambda x: x > 0)   # False
print(f"has_negative={has_negative}  all_positive={all_positive}")

# reduce
product = Query([1.0, 2.0, 3.0, 4.0]).reduce(lambda acc, x: acc * x, initial=1.0)
print("product of [1,2,3,4]:", product)   # 24.0

# skip + take  (pagination pattern)
page_size = 3
page_2 = Query(prices).filter(col > 0).skip(page_size).take(page_size).to_list()
print("page 2:", page_2)

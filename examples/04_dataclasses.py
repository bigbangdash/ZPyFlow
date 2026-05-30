"""
04_dataclasses.py
-----------------
ZPyFlow with dataclasses and Pydantic models.

Shows the generic Python object path: lambda predicates, field projection,
reduce-based aggregation, and how to pull out numeric fields to switch to
the f64 fast path for heavy computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from zpyflow import Query, col, GroupBy

# ------------------------------------------------------------------
# Domain model
# ------------------------------------------------------------------

@dataclass
class Employee:
    id: int
    name: str
    department: str
    title: str
    salary: float
    years: int
    remote: bool
    start_date: date

employees: list[Employee] = [
    Employee(1,  "Alice",   "Engineering", "Staff Engineer",     145_000, 7,  True,  date(2017, 3, 1)),
    Employee(2,  "Bob",     "Marketing",   "Senior Manager",      92_000, 4,  False, date(2020, 6, 15)),
    Employee(3,  "Carol",   "Engineering", "Principal Engineer", 165_000, 10, True,  date(2014, 1, 10)),
    Employee(4,  "Dan",     "HR",          "Recruiter",           72_000, 2,  False, date(2022, 9, 1)),
    Employee(5,  "Eve",     "Engineering", "Senior Engineer",    125_000, 5,  True,  date(2019, 4, 20)),
    Employee(6,  "Frank",   "Marketing",   "Director",           115_000, 8,  False, date(2016, 7, 11)),
    Employee(7,  "Grace",   "HR",          "HR Manager",          88_000, 6,  True,  date(2018, 2, 28)),
    Employee(8,  "Hiro",    "Engineering", "Engineer",            98_000, 3,  True,  date(2021, 5, 5)),
    Employee(9,  "Ivy",     "Finance",     "Controller",         110_000, 9,  False, date(2015, 8, 22)),
    Employee(10, "Jake",    "Finance",     "Analyst",             78_000, 1,  True,  date(2023, 1, 3)),
]

# ------------------------------------------------------------------
# Case 1: Simple filter + project
# ------------------------------------------------------------------
senior_engineers = (
    Query(employees)
        .filter(lambda e: e.department == "Engineering" and e.years >= 5)
        .map(lambda e: {"name": e.name, "title": e.title, "years": e.years})
        .to_list()
)
print("Case 1 — Senior engineers (≥5 years):")
for r in senior_engineers:
    print(f"  {r}")

# ------------------------------------------------------------------
# Case 2: Multi-condition filter
# ------------------------------------------------------------------
remote_high_earners = (
    Query(employees)
        .filter(lambda e: e.remote and e.salary > 100_000)
        .map(lambda e: f"{e.name} ({e.department}) — ${e.salary:,.0f}")
        .to_list()
)
print("\nCase 2 — Remote employees earning > $100K:")
for s in remote_high_earners:
    print(f"  {s}")

# ------------------------------------------------------------------
# Case 3: Compute total payroll using reduce
# ------------------------------------------------------------------
total_payroll = (
    Query(employees)
        .reduce(lambda acc, e: acc + e.salary, initial=0.0)
)
print(f"\nCase 3 — Total payroll: ${total_payroll:,.0f}")

# Case 3b: Engineering payroll — pull out salaries as f64 for fast path
eng_salaries = [e.salary for e in employees if e.department == "Engineering"]
eng_total = Query(eng_salaries).sum()   # GIL released, SIMD
eng_avg   = eng_total / len(eng_salaries)
print(f"           Engineering payroll: ${eng_total:,.0f}  avg: ${eng_avg:,.0f}")

# ------------------------------------------------------------------
# Case 4: GroupBy aggregation
# ------------------------------------------------------------------
by_dept = GroupBy(employees, key_fn=lambda e: e.department)

dept_summary = by_dept.agg(
    headcount=lambda g: g.count(),
    total_salary=lambda g: Query([e.salary for e in g.to_list()]).sum(),
    remote_pct=lambda g: (
        Query(g.to_list()).filter(lambda e: e.remote).count()
        / g.count() * 100
    ),
)

print("\nCase 4 — Department summary:")
for row in sorted(dept_summary, key=lambda r: r["_key"]):
    print(f"  {row['_key']:15s}  n={row['headcount']}  "
          f"total=${row['total_salary']:>10,.0f}  "
          f"remote={row['remote_pct']:.0f}%")

# ------------------------------------------------------------------
# Case 5: Date-based filtering
# ------------------------------------------------------------------
cutoff = date(2020, 1, 1)
new_hires = (
    Query(employees)
        .filter(lambda e: e.start_date >= cutoff)
        .map(lambda e: (e.name, e.start_date.isoformat()))
        .to_list()
)
print(f"\nCase 5 — Hired since {cutoff}:")
for name, dt in new_hires:
    print(f"  {name} — {dt}")

# ------------------------------------------------------------------
# Case 6: first / last / any / all
# ------------------------------------------------------------------
highest_paid = Query(employees).max()                  # uses Python max → returns Employee
# Instead, get name of highest-paid employee:
highest_paid_name = (
    Query(employees)
        .reduce(lambda acc, e: e if e.salary > acc.salary else acc)
)
print(f"\nCase 6 — Highest paid: {highest_paid_name.name} (${highest_paid_name.salary:,.0f})")

anyone_fully_remote = Query(employees).any(lambda e: e.remote)
all_have_title      = Query(employees).all(lambda e: bool(e.title))
print(f"           Any remote: {anyone_fully_remote}  All have titles: {all_have_title}")

# ------------------------------------------------------------------
# Case 7: Pydantic model (same pattern, different class)
# ------------------------------------------------------------------
try:
    from pydantic import BaseModel

    class Order(BaseModel):
        order_id: str
        customer_id: int
        amount: float
        status: str
        items: int

    orders = [
        Order(order_id="A001", customer_id=101, amount=250.0,  status="shipped",  items=3),
        Order(order_id="A002", customer_id=102, amount=45.0,   status="pending",  items=1),
        Order(order_id="A003", customer_id=101, amount=1200.0, status="pending",  items=5),
        Order(order_id="A004", customer_id=103, amount=89.0,   status="cancelled",items=2),
        Order(order_id="A005", customer_id=102, amount=320.0,  status="shipped",  items=4),
    ]

    # Pending orders over $100 — extract dict for the response payload
    pending_large = (
        Query(orders)
            .filter(lambda o: o.status == "pending" and o.amount > 100)
            .map(lambda o: {"order_id": o.order_id, "amount": o.amount, "items": o.items})
            .to_list()
    )
    print("\nCase 7 — Pending orders > $100:")
    for o in pending_large:
        print(f"  {o}")

    # Revenue from shipped orders — fast f64 path
    shipped_revenue = (
        Query([o.amount for o in orders if o.status == "shipped"])
            .sum()
    )
    print(f"           Shipped revenue: ${shipped_revenue:.2f}")

except ImportError:
    print("\nCase 7 — Pydantic not installed, skipping")

"""
usecase_user_segmentation.py
----------------------------
Industry: SaaS / CRM / Marketing automation

ZPyFlow for customer segmentation and lifecycle analysis.

Operations on user records (dataclasses):
  - Segment users by plan tier, activity level, or churn risk
  - Map to enriched profile dicts for downstream systems (email, CRM)
  - Group by segment to compute cohort metrics
  - take() for sampling a segment without building the full list

Note: dataclass operations use the Python object path (GIL held).
Value is in ergonomics (chainable) and memory efficiency (no intermediate
lists when chaining filter → map → take for large user bases).
"""

from __future__ import annotations

import time
import random
import math
from dataclasses import dataclass
from zpyflow import Query

rng = random.Random(2024)

# ------------------------------------------------------------------
# User data model
# ------------------------------------------------------------------

@dataclass
class User:
    user_id:          int
    plan:             str    # free / starter / pro / enterprise
    monthly_revenue:  float
    days_since_login: int
    feature_usage:    float  # 0.0–1.0 breadth of feature adoption
    support_tickets:  int
    country:          str

PLANS    = ["free", "starter", "pro", "enterprise"]
P_PLANS  = [0.55, 0.25, 0.15, 0.05]
COUNTRIES = ["US", "GB", "DE", "JP", "FR", "CA", "AU", "BR", "IN", "SG"]

def make_users(n: int) -> list[User]:
    return [
        User(
            user_id          = i,
            plan             = rng.choices(PLANS, weights=P_PLANS)[0],
            monthly_revenue  = round(
                {"free": 0, "starter": 29, "pro": 99, "enterprise": 499}[
                    rng.choices(PLANS, weights=P_PLANS)[0]
                ] * rng.uniform(0.8, 1.2), 2
            ),
            days_since_login = int(math.exp(rng.gauss(1.5, 1.8))),
            feature_usage    = round(rng.betavariate(2, 3), 3),
            support_tickets  = max(0, int(rng.gauss(0.8, 1.2))),
            country          = rng.choice(COUNTRIES),
        )
        for i in range(n)
    ]

N = 300_000
print(f"Generating {N:,} user records...")
users = make_users(N)
print("Done.\n")

# ------------------------------------------------------------------
# Case 1: Identify at-risk users (churning behaviour signals)
# ------------------------------------------------------------------
CHURN_INACTIVE_DAYS = 30
CHURN_LOW_USAGE     = 0.15

t0 = time.perf_counter()
at_risk = (
    Query(users)
        .filter(lambda u: u.plan != "free"
                      and u.days_since_login > CHURN_INACTIVE_DAYS
                      and u.feature_usage < CHURN_LOW_USAGE)
        .map(lambda u: {
            "user_id":          u.user_id,
            "plan":             u.plan,
            "monthly_revenue":  u.monthly_revenue,
            "days_inactive":    u.days_since_login,
            "churn_risk":       "high",
        })
        .to_list()
)
ms = (time.perf_counter() - t0) * 1000

at_risk_revenue = sum(u["monthly_revenue"] for u in at_risk)
print(f"Case 1 — At-risk users (inactive >{CHURN_INACTIVE_DAYS}d, usage <{CHURN_LOW_USAGE:.0%}):")
print(f"  Count:  {len(at_risk):,}")
print(f"  MRR at risk: ${at_risk_revenue:,.0f}")
print(f"  Time: {ms:.1f}ms")

# ------------------------------------------------------------------
# Case 2: Expansion candidates (healthy paid users for upsell)
# ------------------------------------------------------------------
UPSELL_PLANS = {"starter", "pro"}
MIN_USAGE    = 0.6

t0 = time.perf_counter()
upsell_sample = (
    Query(users)
        .filter(lambda u: u.plan in UPSELL_PLANS
                      and u.feature_usage > MIN_USAGE
                      and u.days_since_login < 14)
        .map(lambda u: {
            "user_id":        u.user_id,
            "current_plan":   u.plan,
            "suggested_plan": "pro" if u.plan == "starter" else "enterprise",
            "feature_usage":  u.feature_usage,
        })
        .take(500)   # campaign batch cap
        .to_list()
)
ms = (time.perf_counter() - t0) * 1000

print(f"\nCase 2 — Upsell candidates (usage >{MIN_USAGE:.0%}, active):")
print(f"  Campaign batch: {len(upsell_sample)} users (capped at 500)")
print(f"  Time: {ms:.1f}ms")

# ------------------------------------------------------------------
# Case 3: Segment distribution — group by plan, compute cohort metrics
# ------------------------------------------------------------------
t0 = time.perf_counter()
by_plan = Query(users).group_by(lambda u: u.plan)
cohorts = by_plan.agg(
    count         = lambda g: g.count(),
    avg_revenue   = lambda g: round(g.map(lambda u: u.monthly_revenue).sum() / max(g.count(), 1), 2),
    avg_usage     = lambda g: round(g.map(lambda u: u.feature_usage).sum()   / max(g.count(), 1), 3),
    avg_inactive  = lambda g: round(g.map(lambda u: u.days_since_login).sum() / max(g.count(), 1), 1),
    avg_tickets   = lambda g: round(g.map(lambda u: u.support_tickets).sum() / max(g.count(), 1), 2),
)
ms = (time.perf_counter() - t0) * 1000

print(f"\nCase 3 — Cohort metrics by plan:")
print(f"  {'plan':12s}  {'count':>7}  {'avg_rev':>9}  {'usage':>7}  {'inactive_d':>11}  {'tickets':>8}")
for row in sorted(cohorts, key=lambda r: r["avg_revenue"], reverse=True):
    print(f"  {row['_key']:12s}  {row['count']:>7,}  "
          f"${row['avg_revenue']:>8.2f}  {row['avg_usage']:>7.3f}  "
          f"{row['avg_inactive']:>11.1f}  {row['avg_tickets']:>8.2f}")
print(f"  Time: {ms:.1f}ms")

# ------------------------------------------------------------------
# Case 4: High-support users — flag for CSM outreach
# ------------------------------------------------------------------
t0 = time.perf_counter()
high_support = (
    Query(users)
        .filter(lambda u: u.support_tickets >= 3 and u.plan in {"pro", "enterprise"})
        .map(lambda u: {
            "user_id": u.user_id,
            "plan":    u.plan,
            "tickets": u.support_tickets,
            "country": u.country,
        })
        .to_list()
)
ms = (time.perf_counter() - t0) * 1000

by_country = Query(high_support).group_by(lambda u: u["country"]).count_per_group()
print(f"\nCase 4 — High-support users (3+ tickets, paid):")
print(f"  Total: {len(high_support):,}")
print(f"  Top countries: {dict(sorted(by_country.items(), key=lambda x: -x[1])[:5])}")
print(f"  Time: {ms:.1f}ms")

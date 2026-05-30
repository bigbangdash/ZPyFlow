"""
usecase_game_simulation.py
--------------------------
Industry: Game backends / Physics simulations / Agent-based models

ZPyFlow for spatial filtering in game and simulation engines.

Every tick / frame, entities outside interaction range or below an
activity threshold are pruned from expensive update logic.
With 100K+ entities, a fused DSL pass replaces a Python loop with
no intermediate list created between filter and take.

Use cases:
  - Nearby entity detection (collision, AI aggro range)
  - Particle system culling (remove dead / out-of-bounds particles)
  - Agent-based simulation: filter active agents each step
"""

from __future__ import annotations

import time
import math
import random
from zpyflow import Query, col

rng = random.Random(1337)

# ------------------------------------------------------------------
# Scenario setup
# ------------------------------------------------------------------
N_ENTITIES          = 200_000   # active entities in the world
MAX_INTERACTION_RADIUS = 50.0   # units — only entities within this range matter
MAX_NEARBY          = 64        # engine hard cap: process at most 64 nearby entities
ACTIVITY_THRESHOLD  = 0.1       # entities with activity_score < 0.1 skip update

# Simulate entity distances from the player (exponential — most entities are far)
distances = [rng.expovariate(1 / 80) for _ in range(N_ENTITIES)]

# Simulate activity scores (beta-distributed — most entities are low-activity)
activity_scores = [rng.betavariate(1.5, 5) for _ in range(N_ENTITIES)]

# Simulate particle lifetimes remaining (uniform — particles die at random)
particle_lifetimes = [rng.uniform(0, 1) for _ in range(N_ENTITIES)]

# ------------------------------------------------------------------
# Case 1: Nearby entity detection (each game tick)
# ------------------------------------------------------------------
TICK_COUNT = 100   # simulate 100 ticks

t0 = time.perf_counter()
for _ in range(TICK_COUNT):
    nearby = (
        Query(distances)
            .filter(col < MAX_INTERACTION_RADIUS)  # within range
            .take(MAX_NEARBY)                       # engine cap
            .to_list()
    )
ms_per_tick = (time.perf_counter() - t0) * 1000 / TICK_COUNT

print(f"Case 1 — Nearby entity detection ({N_ENTITIES:,} entities):")
print(f"  Entities in range (last tick): {len(nearby)}")
print(f"  Average time per tick: {ms_per_tick:.3f}ms")
print(f"  Throughput: {1000 / ms_per_tick:.0f} ticks/sec")

# ------------------------------------------------------------------
# Case 2: Active agent filtering (skip sleeping agents)
# ------------------------------------------------------------------
t0 = time.perf_counter()
active_agents = (
    Query(activity_scores)
        .filter(col > ACTIVITY_THRESHOLD)
        .to_list()
)
ms = (time.perf_counter() - t0) * 1000

print(f"\nCase 2 — Active agent filtering (threshold={ACTIVITY_THRESHOLD}):")
print(f"  Active: {len(active_agents):,} of {N_ENTITIES:,} ({len(active_agents)/N_ENTITIES*100:.1f}%)")
print(f"  Time: {ms:.2f}ms")

# ------------------------------------------------------------------
# Case 3: Particle system culling (remove dead particles)
# ------------------------------------------------------------------
LIFETIME_MIN = 0.05   # particles with < 5% lifetime remaining are culled

t0 = time.perf_counter()
alive_count = Query(particle_lifetimes).filter(col > LIFETIME_MIN).count()
ms = (time.perf_counter() - t0) * 1000

print(f"\nCase 3 — Particle culling (lifetime > {LIFETIME_MIN}):")
print(f"  Alive particles: {alive_count:,} of {N_ENTITIES:,}")
print(f"  Culled: {N_ENTITIES - alive_count:,}")
print(f"  Time: {ms:.2f}ms")

# ------------------------------------------------------------------
# Case 4: Range sweep — interaction radius tuning
# ------------------------------------------------------------------
print(f"\nCase 4 — Interaction radius sensitivity:")
print(f"  {'radius':>8}  {'in_range':>10}  {'time_us':>10}")
for radius in [10, 25, 50, 100, 200]:
    t0 = time.perf_counter()
    n  = Query(distances).filter(col < radius).count()
    us = (time.perf_counter() - t0) * 1_000_000
    print(f"  {radius:>8}  {n:>10,}  {us:>9.0f}µs")

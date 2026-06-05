"""
09_sequence_tools.py
--------------------
Sequence manipulation tools added in ZPyFlow 0.1.x.

Covers:
  - cycle, step_by, interleave, sample
  - Query.iterate, Query.repeat, Query.repeatedly  (infinite factories)
  - flatten, dedupe, partition_by
"""

from __future__ import annotations

from zpyflow import Query, col

# ------------------------------------------------------------------
# Part 1: cycle / step_by / interleave / sample
# ------------------------------------------------------------------

print("Part 1 — cycle / step_by / interleave / sample\n")

# cycle: repeat a finite sequence n times (or infinitely with .take)
repeated = Query([1, 2, 3]).cycle(3).to_list()
print(f"  cycle(3):          {repeated}")

infinite_head = Query(["a", "b"]).cycle().take(7).to_list()
print(f"  cycle().take(7):   {infinite_head}")

# step_by: pick every n-th element
every_third = Query(range(15)).step_by(3).to_list()
print(f"  step_by(3):        {every_third}")

# interleave: zip two streams, stop at the shorter
merged = Query([1, 2, 3]).interleave(Query([10, 20, 30])).to_list()
print(f"  interleave:        {merged}")

short_merge = Query([1, 2]).interleave(Query([10, 20, 30])).to_list()
print(f"  interleave(short): {short_merge}  ← stops at shorter")

# sample: random without replacement
sample = Query(range(20)).sample(5, seed=42).to_list()
print(f"  sample(5):         {sorted(sample)}  (sorted for readability)")

# ------------------------------------------------------------------
# Part 2: Infinite sequence factories
# ------------------------------------------------------------------

print("\nPart 2 — Infinite sequence factories\n")

# iterate: Clojure-style lazy sequence  [seed, fn(seed), fn(fn(seed)), ...]
powers_of_2 = Query.iterate(lambda x: x * 2, 1).take(8).to_list()
print(f"  iterate(x*2, 1):          {powers_of_2}")

fibonacci = (
    Query.iterate(lambda pair: (pair[1], pair[0] + pair[1]), (0, 1))
        .map(lambda pair: pair[0])
        .take(10)
        .to_list()
)
print(f"  iterate (fibonacci):      {fibonacci}")

# repeat: same value n times, or infinitely
fives = Query.repeat(5, 4).to_list()
print(f"  repeat(5, 4):             {fives}")

padded = Query.repeat(0).take(3).to_list()
print(f"  repeat(0).take(3):        {padded}")

# repeatedly: call a function n times
counter = [0]
def increment():
    counter[0] += 1
    return counter[0]

ids = Query.repeatedly(increment, 5).to_list()
print(f"  repeatedly(increment, 5): {ids}")

# Practical: generate unique IDs
import uuid
sample_ids = Query.repeatedly(lambda: str(uuid.uuid4())[:8]).take(3).to_list()
print(f"  repeatedly(uuid, 3):      {sample_ids}")

# ------------------------------------------------------------------
# Part 3: flatten / dedupe / partition_by
# ------------------------------------------------------------------

print("\nPart 3 — flatten / dedupe / partition_by\n")

# flatten: expand one level of nesting
nested = [[1, 2, 3], [4, 5], [6]]
flat = Query(nested).flatten().to_list()
print(f"  flatten:         {flat}")

# flatten with filter after
words = [["hello", "world"], ["zpyflow", "is", "fast"]]
long_words = Query(words).flatten().filter(lambda w: len(w) > 4).to_list()
print(f"  flatten+filter:  {long_words}")

# dedupe: remove consecutive duplicates (like Unix `uniq`)
runs = [1, 1, 2, 3, 3, 3, 2, 2, 1]
deduped = Query(runs).dedupe().to_list()
print(f"  dedupe:          {deduped}  (non-consecutive 1/2 kept)")

# distinct: remove ALL duplicates
all_unique = Query(runs).distinct().to_list()
print(f"  distinct:        {all_unique}")

# partition_by: group consecutive elements sharing the same key (like run-length encoding)
events = [
    {"type": "click",  "ts": 1},
    {"type": "click",  "ts": 2},
    {"type": "scroll", "ts": 3},
    {"type": "scroll", "ts": 4},
    {"type": "scroll", "ts": 5},
    {"type": "click",  "ts": 6},
]
runs_by_type = (
    Query(events)
        .partition_by(lambda e: e["type"])
        .map(lambda group: {"type": group[0]["type"], "count": len(group)})
        .to_list()
)
print(f"\n  partition_by event type:")
for run in runs_by_type:
    print(f"    {run['type']:6s} × {run['count']}")

# ------------------------------------------------------------------
# Part 4: Combining tools
# ------------------------------------------------------------------

print("\nPart 4 — Combining tools\n")

# Generate an infinite sequence, step through it, take a sample
result = (
    Query.iterate(lambda x: x + 3, 0)   # 0, 3, 6, 9, ...
        .take(30)                         # materialise first 30
        .step_by(2)                       # every other: 0, 6, 12, ...
        .to_list()
)
print(f"  iterate(+3).take(30).step_by(2): {result}")

# Interleave two infinite streams (both limited with take first)
evens = Query.iterate(lambda x: x + 2, 0).take(5)
odds  = Query.iterate(lambda x: x + 2, 1).take(5)
zipped = evens.interleave(odds).to_list()
print(f"  interleave(evens, odds):         {zipped}")

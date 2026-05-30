"""
07_ai_embeddings.py
-------------------
ZPyFlow for AI / ML inference pipelines.

Covers:
  - Filtering similarity scores (common in vector search / RAG)
  - Batch inference result processing
  - Embedding norm validation
  - Feature preprocessing before model input
  - Token probability filtering (LLM logits)
"""

from __future__ import annotations

import random
import time
import math

import numpy as np
from zpyflow import Query, col, from_numpy

rng = np.random.default_rng(2024)

# ------------------------------------------------------------------
# Case 1: Vector search — filter candidates by cosine similarity
# ------------------------------------------------------------------
N_DOCS = 1_000_000

# Simulated cosine similarity scores for N_DOCS documents vs. a query
# (in a real system these come from an ANN index like FAISS or Qdrant)
similarity_scores = rng.beta(a=2, b=5, size=N_DOCS).astype(np.float64)  # skewed toward 0

THRESHOLD = 0.60
TOP_K     = 200

t0 = time.perf_counter()
# take(TOP_K) stops as soon as K results are collected — no full scan needed.
# Benchmarks show this is ~70x faster than numpy (which must scan everything first).
top_k_scores = from_numpy(similarity_scores).filter(col > THRESHOLD).take(TOP_K).to_list()
search_ms    = (time.perf_counter() - t0) * 1000

print(f"Case 1 — Vector search (threshold={THRESHOLD}):")
print(f"  {N_DOCS:,} documents → top-{TOP_K} in {search_ms:.1f}ms")
print(f"  Score range: [{min(top_k_scores):.4f}, {max(top_k_scores):.4f}]")

# ------------------------------------------------------------------
# Case 2: Batch inference score statistics
# ------------------------------------------------------------------
BATCH_SIZE  = 10_000
N_BATCHES   = 50

# Simulate N_BATCHES inference batches
batch_scores_all = [
    rng.uniform(0, 1, size=BATCH_SIZE).astype(np.float64)
    for _ in range(N_BATCHES)
]

t0 = time.perf_counter()

batch_stats = []
for batch in batch_scores_all:
    q = from_numpy(batch)
    n = q.count()
    batch_stats.append({
        "n":          n,
        "mean":       q.sum() / n,
        "high_conf":  q.filter(col > 0.9).count(),
        "low_conf":   q.filter(col < 0.3).count(),
        "max":        q.max(),
    })

stats_ms = (time.perf_counter() - t0) * 1000
total_items = N_BATCHES * BATCH_SIZE

print(f"\nCase 2 — Batch inference stats ({N_BATCHES} batches × {BATCH_SIZE:,}):")
print(f"  {total_items:,} scores processed in {stats_ms:.1f}ms")
print(f"  Throughput: {total_items / stats_ms * 1000:,.0f} items/sec")

# Aggregate across batches
all_highs  = sum(s["high_conf"] for s in batch_stats)
all_lows   = sum(s["low_conf"]  for s in batch_stats)
global_max = max(s["max"] for s in batch_stats)
print(f"  High-conf (>0.9): {all_highs:,}  Low-conf (<0.3): {all_lows:,}")
print(f"  Global max score: {global_max:.4f}")

# ------------------------------------------------------------------
# Case 3: Embedding norm validation
# ------------------------------------------------------------------
N_EMBS = 50_000
DIM    = 384   # common sentence-transformer dimension

# Simulate embeddings — some are un-normalized (norm != 1)
embeddings = rng.standard_normal((N_EMBS, DIM)).astype(np.float32)
# Normalize most but leave 5% un-normalized
normalize_mask = rng.random(N_EMBS) > 0.05
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings[normalize_mask] /= norms[normalize_mask]

# Compute norms for all embeddings
computed_norms = np.linalg.norm(embeddings, axis=1).astype(np.float64)

# Use ZPyFlow to find and report un-normalized embeddings
norm_query = from_numpy(computed_norms)

unnormalized_count = norm_query.filter(lambda n: abs(n - 1.0) > 0.01).count()
mean_norm          = norm_query.sum() / norm_query.count()
max_deviation      = from_numpy(computed_norms).map(lambda n: abs(n - 1.0)).max()

print(f"\nCase 3 — Embedding norm validation ({N_EMBS:,} embeddings, dim={DIM}):")
print(f"  Un-normalized (|norm-1| > 0.01): {unnormalized_count:,}")
print(f"  Mean norm:     {mean_norm:.4f}")
print(f"  Max deviation: {max_deviation:.4f}")

# ------------------------------------------------------------------
# Case 4: Feature preprocessing for a tabular model
# ------------------------------------------------------------------
# Raw features: age, income, credit_score, loan_amount, months_employed
N = 100_000
raw_features = {
    "age":              rng.integers(18, 80, size=N).astype(float),
    "income":           rng.lognormal(mean=10.5, sigma=0.8, size=N),
    "credit_score":     rng.integers(300, 850, size=N).astype(float),
    "loan_amount":      rng.lognormal(mean=9.0,  sigma=1.0, size=N),
    "months_employed":  rng.integers(0, 360, size=N).astype(float),
}

t0 = time.perf_counter()

# Validate and normalize each feature column via ZPyFlow
processed = {}
for feat, values in raw_features.items():
    q = from_numpy(values)
    vmin = q.min()
    vmax = q.max()
    # Min-max scale to [0, 1], clipping at ±3σ equivalent
    # PyExpr is single-op: chain two .map() calls for shift + scale.
    # map((col - vmin) / range) would silently lose the subtraction.
    processed[feat] = (
        from_numpy(values)
            .map(col - vmin)                   # shift to [0, range]
            .map(col / (vmax - vmin))          # scale to [0, 1]
            .filter(col.between(0.0, 1.0))
            .to_list()
    )

preproc_ms = (time.perf_counter() - t0) * 1000

print(f"\nCase 4 — Feature preprocessing ({N:,} rows, 5 features):")
print(f"  Time: {preproc_ms:.1f}ms")
for feat, values in processed.items():
    q = Query(values)
    print(f"  {feat:20s}  min={q.min():.4f}  max={q.max():.4f}  n={q.count():,}")

# ------------------------------------------------------------------
# Case 5: LLM token probability filtering (top-p / nucleus sampling)
# ------------------------------------------------------------------
VOCAB_SIZE = 32_000

# Simulate log-probability distribution over vocabulary
logprobs = rng.standard_normal(VOCAB_SIZE).astype(np.float64)
# Convert to probabilities via softmax (approximate)
logprobs -= logprobs.max()
probs = np.exp(logprobs)
probs /= probs.sum()

# Top-p (nucleus) sampling: keep tokens whose cumulative probability ≤ 0.9
NUCLEUS_P = 0.9

sorted_probs = np.sort(probs)[::-1]
cumulative   = np.cumsum(sorted_probs)

# Find the cutoff probability using ZPyFlow
cumulative_q = from_numpy(cumulative)
cutoff_idx   = cumulative_q.filter(col <= NUCLEUS_P).count()
cutoff_prob  = sorted_probs[cutoff_idx] if cutoff_idx < VOCAB_SIZE else 0.0

# Filter vocabulary to nucleus
nucleus_tokens = from_numpy(probs).filter(col >= cutoff_prob).count()

print(f"\nCase 5 — Top-p sampling (p={NUCLEUS_P}, vocab={VOCAB_SIZE:,}):")
print(f"  Nucleus size: {nucleus_tokens:,} tokens")
print(f"  Cutoff prob:  {cutoff_prob:.6f}")
print(f"  Compression:  {nucleus_tokens/VOCAB_SIZE*100:.1f}% of vocabulary")

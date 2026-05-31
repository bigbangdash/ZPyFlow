"""
usecase_langchain_langgraph.py
------------------------------
Industry: AI / LLM applications (RAG, agents, pipelines).

ZPyFlow slots into LangChain and LangGraph wherever a node processes
large numeric arrays — similarity scores, confidence values, logprobs, etc.

No special integration is needed: ZPyFlow is just called inside your
node functions or tools.  The examples below mock the LangChain/LangGraph
types so they run standalone without those libraries installed.

Patterns shown:
  1. RAG retriever  — threshold filter + early stopping on similarity scores
  2. LangGraph node — aggregate a large score array without list materialization
  3. LangChain tool — return pre-aggregated stats to the LLM
  4. Batch scoring  — per-batch stats over a streamed inference pipeline
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

import numpy as np
from zpyflow import Query, col, from_numpy

rng = random.Random(42)
np_rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Mock helpers (replace with real LangChain/LangGraph objects in production)
# ---------------------------------------------------------------------------

class MockDocument:
    def __init__(self, content: str, doc_id: int):
        self.page_content = content
        self.metadata = {"id": doc_id}

    def __repr__(self) -> str:
        return f"Document(id={self.metadata['id']}, content={self.page_content[:30]!r})"


def mock_embed(texts: list[str]) -> np.ndarray:
    """Returns random unit vectors — stand-in for a real embedding model."""
    vecs = np_rng.standard_normal((len(texts), 128)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def cosine_scores(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    return (doc_vecs @ query_vec).astype(np.float64)


# ---------------------------------------------------------------------------
# Case 1 — RAG retriever: threshold filter + early stopping
# ---------------------------------------------------------------------------

N_DOCS = 500_000
RETRIEVAL_THRESHOLD = 0.6
TOP_K = 20

print(f"Case 1 — RAG retriever ({N_DOCS:,} documents):")

docs = [MockDocument(f"doc content {i}", i) for i in range(N_DOCS)]

# Simulate realistic similarity score distribution:
# most docs are low-relevance (beta skewed toward 0), a few are highly relevant
scores = np_rng.beta(1.5, 8, N_DOCS).astype(np.float64)
# Inject ~50 highly relevant docs near the query
hot_idx = np_rng.choice(N_DOCS, size=50, replace=False)
scores[hot_idx] = np_rng.uniform(0.65, 0.95, size=50)

t0 = time.perf_counter()

# ZPyFlow: SIMD filter + early stopping — never scans beyond the K-th hit
top_indices = (
    from_numpy(scores)
    .filter(col > RETRIEVAL_THRESHOLD)
    .take(TOP_K)
    .to_list()
)

ms = (time.perf_counter() - t0) * 1000
retrieved = [docs[int(i)] for i in top_indices]

print(f"  Retrieved {len(retrieved)} docs above threshold={RETRIEVAL_THRESHOLD}")
print(f"  Time: {ms:.2f}ms  (early stopping — did not scan all {N_DOCS:,})")
print()

# ---------------------------------------------------------------------------
# Case 2 — LangGraph node: aggregate score array without materializing a list
# ---------------------------------------------------------------------------

print("Case 2 — LangGraph node (score aggregation):")


def score_filter_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node that receives a large score array and returns summary stats.
    All aggregations stay inside Rust — no Python list is created.

    In a real LangGraph graph:
        graph.add_node("score_filter", score_filter_node)
    """
    scores: list[float] = state["candidate_scores"]
    q = Query(scores)
    n = q.count()
    return {
        "total_candidates": n,
        "high_quality":     q.filter(col > 0.85).count(),
        "low_quality":      q.filter(col < 0.5).count(),
        "best_score":       round(q.max(), 4),
        "mean_score":       round(q.sum() / n, 4) if n else 0.0,
    }


N_CANDIDATES = 1_000_000
candidate_scores = np_rng.beta(2, 5, N_CANDIDATES).tolist()  # skewed toward 0

t0 = time.perf_counter()
result = score_filter_node({"candidate_scores": candidate_scores})
ms = (time.perf_counter() - t0) * 1000

print(f"  Input: {N_CANDIDATES:,} scores")
for k, v in result.items():
    print(f"  {k}: {v}")
print(f"  Time: {ms:.2f}ms")
print()

# ---------------------------------------------------------------------------
# Case 3 — LangChain tool: return pre-aggregated stats to the LLM
# ---------------------------------------------------------------------------

print("Case 3 — LangChain tool (stats returned to LLM):")


def analyze_search_results(scores: list[float]) -> dict[str, Any]:
    """
    LangChain @tool that aggregates similarity scores.
    The LLM receives a compact dict instead of a raw list.

    In a real LangChain app:
        @tool
        def analyze_search_results(scores: list[float]) -> dict: ...
        llm_with_tools = llm.bind_tools([analyze_search_results])
    """
    q = Query(scores)
    n = q.count()
    if n == 0:
        return {"error": "no scores provided"}
    return {
        "total":        n,
        "high_quality": q.filter(col > 0.85).count(),
        "low_quality":  q.filter(col < 0.5).count(),
        "best_score":   round(q.max(), 4),
        "mean_score":   round(q.sum() / n, 4),
    }


tool_input = np_rng.beta(2, 5, 200_000).tolist()

t0 = time.perf_counter()
tool_output = analyze_search_results(tool_input)
ms = (time.perf_counter() - t0) * 1000

print(f"  Tool input:  {len(tool_input):,} scores")
print(f"  Tool output: {tool_output}")
print(f"  Time: {ms:.2f}ms")
print()

# ---------------------------------------------------------------------------
# Case 4 — Batch scoring pipeline: per-batch stats over a streamed source
# ---------------------------------------------------------------------------

print("Case 4 — Batch scoring pipeline (streaming simulation):")

TOTAL = 2_000_000
BATCH_SIZE = 50_000


def score_stream(total: int, batch_size: int):
    """Simulate a stream of inference batches (e.g. from a model server)."""
    for start in range(0, total, batch_size):
        size = min(batch_size, total - start)
        yield np_rng.beta(2, 5, size).tolist()


t0 = time.perf_counter()
batch_stats = []
for batch in score_stream(TOTAL, BATCH_SIZE):
    q = Query(batch)
    n = q.count()
    batch_stats.append({
        "n":          n,
        "high_conf":  q.filter(col > 0.85).count(),
        "mean":       round(q.sum() / n, 4),
    })

ms = (time.perf_counter() - t0) * 1000
total_high = sum(b["high_conf"] for b in batch_stats)

print(f"  {len(batch_stats)} batches × {BATCH_SIZE:,} = {TOTAL:,} total scores")
print(f"  High-confidence (>0.85): {total_high:,} "
      f"({total_high/TOTAL*100:.1f}%)")
print(f"  Mean per batch: {sum(b['mean'] for b in batch_stats)/len(batch_stats):.4f}")
print(f"  Total time: {ms:.2f}ms")

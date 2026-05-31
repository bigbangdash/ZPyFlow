"""
usecase_langchain_langgraph.py
------------------------------
An AI agent that analyzes transaction data using ZPyFlow as tools,
backed by OpenAI / Anthropic / Google Gemini (whichever key is set).

ZPyFlow is used inside LangChain tools for fast numeric processing:
  - filter transactions by amount / risk score
  - compute aggregate stats without materializing full Python lists
  - early-stopping top-K retrieval

Setup — set at least one API key:
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."
    export GOOGLE_API_KEY="AIza..."

Run:
    python examples/usecase_langchain_langgraph.py

Dependencies:
    pip install langchain langgraph langchain-openai \
                langchain-anthropic langchain-google-genai numpy zpyflow
"""

from __future__ import annotations

import os
import random
import math
from typing import Annotated, Any

import numpy as np
from zpyflow import Query, col, from_numpy

# ── LangChain / LangGraph imports ──────────────────────────────────────────
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode

# ---------------------------------------------------------------------------
# 1. Pick LLM provider based on available API keys
# ---------------------------------------------------------------------------

def get_llm():
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        print("Using Claude (Anthropic)")
        return ChatAnthropic(model="claude-3-5-haiku-20241022", max_tokens=1024)

    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        print("Using GPT-4o-mini (OpenAI)")
        return ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)

    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("Using Gemini Flash (Google)")
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", max_tokens=1024)

    raise EnvironmentError(
        "No API key found. Set one of:\n"
        "  ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY"
    )

# ---------------------------------------------------------------------------
# 2. Generate mock transaction dataset (simulates a live data source)
# ---------------------------------------------------------------------------

rng = random.Random(42)
np_rng = np.random.default_rng(42)

N_TRANSACTIONS = 500_000

print(f"Generating {N_TRANSACTIONS:,} mock transactions...")

transactions = [
    {
        "id":         i,
        "amount":     round(math.exp(rng.gauss(4.5, 1.5)), 2),   # log-normal
        "risk_score": float(np_rng.beta(1.5, 8)),                 # mostly low
        "category":   rng.choice(["retail", "travel", "online", "atm", "p2p"]),
        "status":     rng.choice(["approved", "approved", "approved", "flagged", "declined"]),
    }
    for i in range(N_TRANSACTIONS)
]
# Inject some high-risk transactions
for i in range(0, N_TRANSACTIONS, 5000):
    transactions[i]["risk_score"] = round(rng.uniform(0.75, 0.99), 3)

amounts = np.array([t["amount"] for t in transactions], dtype=np.float64)
risk_scores = np.array([t["risk_score"] for t in transactions], dtype=np.float64)
print("Done.\n")

# ---------------------------------------------------------------------------
# 3. LangChain tools powered by ZPyFlow
# ---------------------------------------------------------------------------

@tool
def get_transaction_stats(threshold_amount: float = 0.0) -> dict[str, Any]:
    """
    Return summary statistics for transactions above a given amount threshold.
    Uses ZPyFlow SIMD aggregation — all ops stay in Rust, no Python list created.
    """
    q = from_numpy(amounts).filter(col > threshold_amount)
    n = q.count()
    if n == 0:
        return {"count": 0, "message": f"No transactions above {threshold_amount}"}
    return {
        "count":   n,
        "total":   round(q.sum(), 2),
        "mean":    round(q.sum() / n, 2),
        "max":     round(q.max(), 2),
        "min":     round(q.min(), 2),
        "pct_of_total": round(n / N_TRANSACTIONS * 100, 2),
    }


@tool
def get_high_risk_transactions(risk_threshold: float = 0.8, top_k: int = 10) -> dict[str, Any]:
    """
    Find transactions with risk score above the threshold.
    Uses ZPyFlow early stopping — stops scanning once top_k are found.
    Returns the top_k highest-risk transaction IDs and their scores.
    """
    # Count total flagged
    flagged_count = from_numpy(risk_scores).filter(col > risk_threshold).count()

    # Get top-K indices with early stopping
    top_indices = (
        from_numpy(risk_scores)
        .filter(col > risk_threshold)
        .take(top_k)
        .to_list()
    )
    top_txns = [
        {"id": transactions[int(i)]["id"], "risk_score": transactions[int(i)]["risk_score"]}
        for i in top_indices
    ]
    return {
        "risk_threshold":    risk_threshold,
        "total_flagged":     flagged_count,
        "pct_flagged":       round(flagged_count / N_TRANSACTIONS * 100, 3),
        "sample_top_k":      top_txns,
    }


@tool
def get_category_risk_summary() -> dict[str, Any]:
    """
    Compute average risk score and transaction count per category.
    Uses ZPyFlow lambda filters on dict records.
    """
    categories = ["retail", "travel", "online", "atm", "p2p"]
    summary = {}
    for cat in categories:
        q = Query(transactions).filter(lambda t, c=cat: t["category"] == c)
        n = q.count()
        if n == 0:
            continue
        risk_vals = [t["risk_score"] for t in transactions if t["category"] == cat]
        avg_risk = Query(risk_vals).sum() / n
        summary[cat] = {
            "count":    n,
            "avg_risk": round(avg_risk, 4),
        }
    return summary

# ---------------------------------------------------------------------------
# 4. LangGraph agent
# ---------------------------------------------------------------------------

tools = [get_transaction_stats, get_high_risk_transactions, get_category_risk_summary]
llm = get_llm()
llm_with_tools = llm.bind_tools(tools)


def call_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


graph = StateGraph(MessagesState)
graph.add_node("llm", call_llm)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("llm")
graph.add_conditional_edges("llm", should_continue)
graph.add_edge("tools", "llm")
app = graph.compile()

# ---------------------------------------------------------------------------
# 5. Run example queries
# ---------------------------------------------------------------------------

queries = [
    "How many transactions are above $1,000? What's the total value?",
    "Find high-risk transactions with a risk score above 0.85. How many are there?",
    "Which transaction category has the highest average risk score?",
]

for query in queries:
    print(f"Q: {query}")
    result = app.invoke({"messages": [HumanMessage(content=query)]})
    answer = result["messages"][-1].content
    print(f"A: {answer}")
    print()

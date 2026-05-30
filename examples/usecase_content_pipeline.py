"""
usecase_content_pipeline.py
---------------------------
Industry: Media / CMS / Newsletter platforms

ZPyFlow for content processing and editorial pipelines.

Operations on article/post records (Python dicts):
  - Filter by publication status, category, or quality score
  - Map to preview/feed format (excerpt + metadata, no full body)
  - take() for paginated feeds without scanning the full catalog
  - GroupBy for per-category analytics

Note: string/object operations use the Python path (GIL held).
ZPyFlow's value is the chainable API and avoiding intermediate lists
when building paginated feeds (filter → map → take in one pass).
"""

from __future__ import annotations

import math
import time
import random
from zpyflow import Query

rng = random.Random(77)

# ------------------------------------------------------------------
# Simulate content catalog
# ------------------------------------------------------------------

STATUSES    = ["draft", "review", "published", "archived"]
P_STATUS    = [0.20, 0.10, 0.60, 0.10]
CATEGORIES  = ["tech", "finance", "health", "sports", "culture", "science"]
TAGS_POOL   = ["ai", "python", "startup", "investing", "wellness",
               "climate", "product", "design", "data", "security"]

def make_articles(n: int) -> list[dict]:
    return [
        {
            "id":            f"ART-{i:06d}",
            "title":         f"Article {i}: {rng.choice(TAGS_POOL).title()} Trends",
            "body":          "Lorem ipsum " * rng.randint(50, 500),
            "category":      rng.choice(CATEGORIES),
            "status":        rng.choices(STATUSES, weights=P_STATUS)[0],
            "tags":          rng.sample(TAGS_POOL, k=rng.randint(1, 4)),
            "quality_score": round(rng.betavariate(3, 2), 3),  # skewed high
            "word_count":    rng.randint(300, 5000),
            "author_id":     rng.randint(1, 500),
            "published_at":  1_700_000_000 + i * 3600 if rng.random() > 0.4 else None,
            "views":         int(math.exp(rng.gauss(6, 1.5))),
        }
        for i in range(n)
    ]

N = 200_000
print(f"Generating {N:,} articles...")
articles = make_articles(N)
print("Done.\n")

# ------------------------------------------------------------------
# Case 1: Homepage feed — published articles, quality-filtered, paginated
# ------------------------------------------------------------------
MIN_QUALITY = 0.6
PAGE_SIZE   = 20
PAGE        = 0     # first page

t0 = time.perf_counter()
feed = (
    Query(articles)
        .filter(lambda a: a["status"] == "published" and a["quality_score"] >= MIN_QUALITY)
        .map(lambda a: {                          # strip body — feed needs preview only
            "id":            a["id"],
            "title":         a["title"],
            "category":      a["category"],
            "tags":          a["tags"],
            "excerpt":       a["body"][:200] + "…",
            "quality_score": a["quality_score"],
            "views":         a["views"],
        })
        .skip(PAGE * PAGE_SIZE)
        .take(PAGE_SIZE)
        .to_list()
)
ms = (time.perf_counter() - t0) * 1000

print(f"Case 1 — Homepage feed (quality ≥ {MIN_QUALITY}, page {PAGE+1}):")
print(f"  Articles returned: {len(feed)}")
if feed:
    print(f"  Top article: [{feed[0]['category']}] {feed[0]['title']}")
print(f"  Time: {ms:.1f}ms  (filter+map+take in one pass, body never materialized)")

# ------------------------------------------------------------------
# Case 2: Editorial queue — articles in review, ordered for editor
# ------------------------------------------------------------------
t0 = time.perf_counter()
review_queue = (
    Query(articles)
        .filter(lambda a: a["status"] == "review")
        .map(lambda a: {
            "id":         a["id"],
            "title":      a["title"],
            "author_id":  a["author_id"],
            "word_count": a["word_count"],
            "category":   a["category"],
            "priority":   "long-read" if a["word_count"] > 2000 else "short",
        })
        .to_list()
)
ms = (time.perf_counter() - t0) * 1000

long_reads = sum(1 for a in review_queue if a["priority"] == "long-read")
print(f"\nCase 2 — Editorial review queue:")
print(f"  {len(review_queue):,} articles awaiting review")
print(f"  Long-reads (>2000 words): {long_reads} | Short: {len(review_queue)-long_reads}")
print(f"  Time: {ms:.1f}ms")

# ------------------------------------------------------------------
# Case 3: Category analytics — published articles grouped by category
# ------------------------------------------------------------------
t0 = time.perf_counter()
published = Query(articles).filter(lambda a: a["status"] == "published").to_list()
by_cat    = Query(published).group_by(lambda a: a["category"])
cat_stats = by_cat.agg(
    count      = lambda g: g.count(),
    avg_views  = lambda g: round(g.map(lambda a: a["views"]).sum()         / max(g.count(), 1)),
    avg_quality= lambda g: round(g.map(lambda a: a["quality_score"]).sum() / max(g.count(), 1), 3),
    total_views= lambda g: g.map(lambda a: a["views"]).sum(),
)
ms = (time.perf_counter() - t0) * 1000

print(f"\nCase 3 — Published articles by category:")
print(f"  {'category':10s}  {'count':>6}  {'avg_views':>10}  {'quality':>8}  {'total_views':>12}")
for row in sorted(cat_stats, key=lambda r: r["total_views"], reverse=True):
    print(f"  {row['_key']:10s}  {row['count']:>6,}  "
          f"{row['avg_views']:>10,}  {row['avg_quality']:>8.3f}  {row['total_views']:>12,}")
print(f"  Time: {ms:.1f}ms")

# ------------------------------------------------------------------
# Case 4: Trending content — high views, recently published, tag-filtered
# ------------------------------------------------------------------
TARGET_TAG  = "ai"
VIEW_THRESH = 10_000

t0 = time.perf_counter()
trending = (
    Query(articles)
        .filter(lambda a: a["status"] == "published"
                      and TARGET_TAG in a["tags"]
                      and a["views"] > VIEW_THRESH)
        .map(lambda a: {"id": a["id"], "title": a["title"], "views": a["views"]})
        .take(10)
        .to_list()
)
ms = (time.perf_counter() - t0) * 1000

print(f"\nCase 4 — Trending '{TARGET_TAG}' articles (>{VIEW_THRESH:,} views, top 10):")
for a in trending:
    print(f"  [{a['views']:>7,} views] {a['title']}")
print(f"  Time: {ms:.1f}ms")

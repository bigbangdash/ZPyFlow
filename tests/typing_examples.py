from __future__ import annotations

from typing import TypedDict

from zpyflow import Query, agg_count, agg_sum, col, field, from_numpy


class Product(TypedDict):
    category: str
    price: float
    active: bool


products: list[Product] = [
    {"category": "books", "price": 12.5, "active": True},
    {"category": "games", "price": 59.0, "active": False},
]

prices: list[float] = Query([1.0, 2.0, 3.0]).filter(col > 1).map(col * 2).to_list()
active: list[Product] = Query(products).filter(field("active") == True).to_list()

groups: list[dict[str, object]] = Query(products).group_agg(
    field("category"),
    count=agg_count(),
)

revenue: list[dict[str, object]] = Query(products).group_agg(
    lambda p: p["category"],
    count=agg_count(),
    total=agg_sum(lambda p: p["price"]),
)

maybe_first: Product | None = Query(products).first()


def accepts_numpy(arr: object) -> None:
    from_numpy(arr).filter(col > 0).count()

"""
Convert simple comparison / arithmetic lambdas to ZPyFlow DSL Expr objects.

Handles patterns like:
  filter: lambda x: x > 5     →  col > 5.0
  filter: lambda x: x >= 0.5  →  col >= 0.5
  map:    lambda x: x * 2     →  col * 2.0
  map:    lambda x: x + 1.0   →  col + 1.0

Returns None when the lambda is too complex; caller falls back to the Python
callable path.  Never raises — all exceptions are swallowed.
"""
from __future__ import annotations

import ast
import inspect
import re
from typing import Any, Callable, Optional


def try_lambda_to_expr(fn: Callable[..., Any]) -> Optional[Any]:
    """
    Parse fn's source as a single-arg lambda and return a DSL Expr, or None.

    Only succeeds for:
      - compare:  lambda x: x OP literal   (OP ∈ >, >=, <, <=, ==, !=)
      - binop:    lambda x: x OP literal   (OP ∈ *, +, -, /)
    where literal is an int or float constant.
    """
    try:
        src = _extract_lambda_source(fn)
        if src is None:
            return None
        tree = ast.parse(src, mode="eval")
        return _ast_to_expr(tree)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _extract_lambda_source(fn: Callable[..., Any]) -> Optional[str]:
    try:
        raw = inspect.getsource(fn).strip()
    except Exception:
        return None

    # inspect may return the whole line; find the lambda substring
    m = re.search(r"lambda\s+\w+\s*:[^\n,)\]]+", raw)
    if m:
        src = m.group(0).strip()
        # strip trailing partial expressions (e.g. ".filter(lambda x: x > 0)")
        src = src.rstrip(" ,)]\n")
        return src
    return None


def _ast_to_expr(tree: ast.AST) -> Optional[Any]:
    from zpyflow._zpyflow import col  # lazy import to avoid circular dep

    if not isinstance(tree, ast.Expression):
        return None
    node = tree.body
    if not isinstance(node, ast.Lambda):
        return None

    if len(node.args.args) != 1:
        return None
    arg_name = node.args.args[0].arg
    body = node.body

    if isinstance(body, ast.Compare):
        return _compare_to_expr(body, arg_name, col)
    if isinstance(body, ast.BinOp):
        return _binop_to_expr(body, arg_name, col)
    return None


def _const_value(node: ast.expr) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    # Python <3.8 ast.Num compat
    if isinstance(node, ast.Num):
        return float(node.n)  # type: ignore[arg-type]
    return None


def _compare_to_expr(node: ast.Compare, arg_name: str, col: Any) -> Optional[Any]:
    if not isinstance(node.left, ast.Name) or node.left.id != arg_name:
        return None
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return None

    value = _const_value(node.comparators[0])
    if value is None:
        return None

    op = node.ops[0]
    if isinstance(op, ast.Gt):    return col > value
    if isinstance(op, ast.GtE):   return col >= value
    if isinstance(op, ast.Lt):    return col < value
    if isinstance(op, ast.LtE):   return col <= value
    if isinstance(op, ast.Eq):    return col == value
    if isinstance(op, ast.NotEq): return col != value
    return None


def _binop_to_expr(node: ast.BinOp, arg_name: str, col: Any) -> Optional[Any]:
    if not isinstance(node.left, ast.Name) or node.left.id != arg_name:
        return None

    value = _const_value(node.right)
    if value is None:
        return None

    op = node.op
    if isinstance(op, ast.Mult): return col * value
    if isinstance(op, ast.Add):  return col + value
    if isinstance(op, ast.Sub):  return col - value
    if isinstance(op, ast.Div):  return col / value
    return None

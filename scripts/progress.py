"""Progress against the 'good enough' lines in metrics/targets.json.

Reads metrics/current.json (written by `make metrics`) and prints a scoreboard:
which cases are DONE, which are not, and how far each open case is from its
line (largest gap first = what to work on next).

It also detects polishing: if, since the previous run, the only cases that
improved are ones already DONE while open cases remain, it exits 3.

Usage:
    python scripts/progress.py          # scoreboard; exit 3 on polishing
    python scripts/progress.py --no-state   # do not read/update the previous snapshot
    python scripts/progress.py --targets X  # alternative targets file (for testing the gate)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MDIR = ROOT / "metrics"
PREV = MDIR / ".progress_prev.json"
DET = ("rust_allocs", "rust_bytes", "rust_peak", "py_peak")


def output_counts(names: list[str]) -> dict[str, int]:
    """Run the to_list cases once (same definitions as the judge) to get len(output)."""
    spec = importlib.util.spec_from_file_location("_metrics_cases", ROOT / "scripts" / "metrics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # read-only use of the judge's case table
    cases = mod.build_cases()
    return {n: len(cases[n]()) for n in names}


def evaluate(cur: dict, targets: dict) -> dict[str, dict]:
    """name -> {done, gap, why}. gap: bytes over the line (0 when done)."""
    res = {}
    agg = targets["aggregate"]
    for n in agg["cases"]:
        m = cur[n]
        done = m["rust_allocs"] <= agg["rust_allocs"]
        res[n] = {"done": done, "gap": 0 if done else m["rust_bytes"],
                  "why": f"allocs {m['rust_allocs']} (line {agg['rust_allocs']}), bytes {m['rust_bytes']}"}
    tl = targets["to_list"]
    counts = output_counts(list(tl["cases"]))
    for n, per in tl["cases"].items():
        m = cur[n]
        line = int(counts[n] * per * tl["slack"])
        done = m["rust_bytes"] <= line
        res[n] = {"done": done, "gap": max(0, m["rust_bytes"] - line),
                  "why": f"bytes {m['rust_bytes']} (line {line} = {counts[n]} items x {per}B x {tl['slack']})"}
    return res


def improved(cur: dict, prev: dict) -> list[str]:
    return [n for n, c in cur.items() if n in prev and any(c[k] < prev[n][k] for k in DET)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-state", action="store_true")
    ap.add_argument("--targets", default=str(MDIR / "targets.json"))
    a = ap.parse_args()

    cur = json.loads((MDIR / "current.json").read_text())
    targets = json.loads(Path(a.targets).read_text())
    ev = evaluate(cur, targets)
    done = [n for n, e in ev.items() if e["done"]]
    open_ = sorted((n for n, e in ev.items() if not e["done"]), key=lambda n: -ev[n]["gap"])

    print(f"PROGRESS: {len(done)}/{len(ev)} cases meet their line")
    for n in open_:
        print(f"  OPEN  {n:24} gap {ev[n]['gap']:>10} B   {ev[n]['why']}")
    for n in done:
        print(f"  DONE  {n:24} {ev[n]['why']}")
    if open_:
        print(f"NEXT: {open_[0]} (largest gap)")

    status = 0
    if not a.no_state and PREV.exists():
        prev = json.loads(PREV.read_text())
        imp = improved(cur, prev)
        if imp and open_ and all(ev[n]["done"] for n in imp):
            print(f"\nPOLISHING: this change only improved cases already DONE ({', '.join(imp)}) "
                  f"while {len(open_)} case(s) are still OPEN. Work on {open_[0]} instead, "
                  "or report to the user why the DONE case had to be touched.")
            status = 3
    if not a.no_state:
        PREV.write_text(json.dumps(cur, indent=1))
    return status


if __name__ == "__main__":
    sys.exit(main())

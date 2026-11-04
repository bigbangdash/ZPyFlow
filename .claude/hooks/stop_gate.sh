#!/bin/sh
# Stop gate: if library code changed, rebuild with the alloc counter and run
# scripts/metrics.py --check. Regression -> exit 2 (Claude must fix or revert).
cd "$(dirname "$0")/../.." || exit 0
input=$(cat)
if [ -z "$(git status --porcelain -- src zpyflow Cargo.toml Cargo.lock)" ]; then
  exit 0  # no library change, nothing to measure
fi
out=$( { .venv/bin/maturin develop --profile dev-release \
          --features "alloc-stats pyo3/extension-module" >/dev/null 2>&1 \
        || echo "BUILD FAILED"; } ; .venv/bin/python scripts/metrics.py --check 2>&1 )
status=$?
case "$out" in *"BUILD FAILED"*) status=1 ;; esac
[ $status -eq 0 ] && exit 0
# Second stop attempt in a row: let it stop, but the regression must be reported.
if printf '%s' "$input" | grep -q '"stop_hook_active": *true'; then
  echo "metrics gate still failing — report this regression to the user as-is." >&2
  printf '%s\n' "$out" | tail -20 >&2
  exit 0
fi
echo "metrics gate failed (see CLAUDE.md constraints). Fix or revert before stopping:" >&2
printf '%s\n' "$out" | tail -20 >&2
exit 2

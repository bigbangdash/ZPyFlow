# bench_fastapi.py — ZPyFlow in a FastAPI sync endpoint under concurrent load
#
# Two server configurations compared side-by-side:
#   - 1 worker  (uvicorn, single process) — isolates GIL-release effect
#   - N workers (uvicorn --workers N)     — production-like multi-process setup
#
# FastAPI sync endpoints run in a thread pool per worker process.
# ZPyFlow numeric DSL releases the GIL → threads don't serialize.
# Object pipelines using Python lambdas hold the GIL (shown for contrast).
#
# ── AWS ECS deployment notes ────────────────────────────────────────────────
#
# Typical ECS task definition (Fargate):
#
#   Container command:
#     uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 4
#
#   Worker count rule of thumb:
#     CPU-bound sync endpoints → workers = vCPU count (e.g. 4 for 4 vCPU task)
#     Mix of async + sync      → workers = 2–4, rely on async concurrency
#
#   ECS scaling layers:
#     1. Thread pool (per worker, intra-process)  ← this benchmark's "1w" column
#     2. Workers     (per task, multi-process)    ← this benchmark's "Nw" column
#     3. ECS tasks   (horizontal, ALB-distributed) ← linear with task count
#
#   ZPyFlow benefit per layer:
#     Layer 1: GIL release → threads don't serialize → measured as "×" column
#     Layer 2: faster per-process → measured as "scale" column staying near Nx
#     Layer 3: same as Python (stateless request handling)
#
#   Dockerfile snippet:
#     CMD ["uvicorn", "app.main:app",
#          "--host", "0.0.0.0", "--port", "8080",
#          "--workers", "4", "--log-level", "info"]
#
#   For task sizing: a 2 vCPU / 4 GB task with --workers 2 is a common starting
#   point; scale out ECS tasks rather than up (keeps memory footprint predictable).
#
# ────────────────────────────────────────────────────────────────────────────
#
# Setup:
#   pip install fastapi uvicorn httpx
#
# Run:
#   python sandbox/benchmark/benchmarks/bench_fastapi.py
#   python sandbox/benchmark/run.py --suite fastapi

# ---------------------------------------------------------------------------
# App definition (module-level — imported by uvicorn worker processes)
# ---------------------------------------------------------------------------

import fastapi

try:
    from zpyflow import Query, col
except ImportError:
    raise RuntimeError("zpyflow not built — run: maturin develop --profile dev-release")

N_FLOATS = 200_000
N_DICTS  = 100_000

floats = [float(i % 1000) for i in range(N_FLOATS)]
dicts  = [{"status": "active" if i % 3 != 0 else "inactive",
           "value":  float(i % 1000),
           "cat":    str(i % 10)}
          for i in range(N_DICTS)]

app = fastapi.FastAPI()


@app.get("/python/filter_count")
def ep_py_filter_count():
    return {"result": sum(1 for x in floats if x > 500)}

@app.get("/zpyflow/filter_count")
def ep_zp_filter_count():
    return {"result": Query(floats).filter(col > 500).count()}


@app.get("/python/filter_sum")
def ep_py_filter_sum():
    return {"result": sum(x for x in floats if x > 500)}

@app.get("/zpyflow/filter_sum")
def ep_zp_filter_sum():
    return {"result": Query(floats).filter(col > 500).sum()}


@app.get("/python/stats")
def ep_py_stats():
    xs   = [x for x in floats if x > 0]
    n    = len(xs)
    s    = sum(xs)
    mean = s / n
    var  = sum((x - mean) ** 2 for x in xs) / n
    return {"n": n, "sum": s, "mean": mean, "min": min(xs), "max": max(xs), "var": var}

@app.get("/zpyflow/stats")
def ep_zp_stats():
    return Query(floats).filter(col > 0).stats()


@app.get("/python/object_filter")
def ep_py_object_filter():
    result = [r for r in dicts if r["status"] == "active" and r["value"] > 300]
    return {"count": len(result)}

@app.get("/zpyflow/object_filter")
def ep_zp_object_filter():
    result = Query(dicts).filter(lambda r: r["status"] == "active" and r["value"] > 300).to_list()
    return {"count": len(result)}


@app.get("/python/chained")
def ep_py_chained():
    return {"result": sum(x * 2 for x in floats if x > 500)}

@app.get("/zpyflow/chained")
def ep_zp_chained():
    return {"result": Query(floats).filter(col > 500).map(col * 2).sum()}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time
    import subprocess
    import concurrent.futures
    import httpx
    from pathlib import Path

    HOST       = "127.0.0.1"
    PORT_1W    = 18765   # 1-worker server
    PORT_NW    = 18766   # N-worker server
    N_WORKERS  = 4
    # Client load is scaled with worker count so each worker sees the same
    # concurrent pressure. 1-worker baseline: 8 concurrent / 48 requests.
    N_CONCURRENT_1W = 8
    N_REQUESTS_1W   = 48
    N_CONCURRENT_NW = N_CONCURRENT_1W * N_WORKERS   # 32
    N_REQUESTS_NW   = N_REQUESTS_1W   * N_WORKERS   # 192

    BENCH_ROOT = Path(__file__).parent.parent  # sandbox/benchmark/

    ENDPOINT_PAIRS = [
        ("filter+count (numeric)", "/python/filter_count",  "/zpyflow/filter_count"),
        ("filter+sum   (numeric)", "/python/filter_sum",    "/zpyflow/filter_sum"),
        ("stats()      (numeric)", "/python/stats",         "/zpyflow/stats"),
        ("object filter  (dict) ", "/python/object_filter", "/zpyflow/object_filter"),
        ("chained f→map→sum     ", "/python/chained",       "/zpyflow/chained"),
    ]

    def start_uvicorn(port, workers=1):
        cmd = [
            sys.executable, "-m", "uvicorn",
            "benchmarks.bench_fastapi:app",
            "--host", HOST,
            "--port", str(port),
            "--log-level", "error",
        ]
        if workers > 1:
            cmd += ["--workers", str(workers)]
        return subprocess.Popen(cmd, cwd=str(BENCH_ROOT),
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)

    def bench_endpoint(host, port, path, n_concurrent, n_total):
        url = f"http://{host}:{port}{path}"
        def one():
            t0 = time.perf_counter()
            httpx.get(url, timeout=10)
            return time.perf_counter() - t0
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as ex:
            t0 = time.perf_counter()
            latencies = list(ex.map(lambda _: one(), range(n_total)))
            wall = time.perf_counter() - t0
        return n_total / wall, sum(latencies) / len(latencies) * 1000

    # ── Start both servers ─────────────────────────────────────────────────
    print(f"Starting 1-worker  uvicorn on :{PORT_1W} ...")
    p1w = start_uvicorn(PORT_1W, workers=1)
    print(f"Starting {N_WORKERS}-worker uvicorn on :{PORT_NW} ...")
    pnw = start_uvicorn(PORT_NW, workers=N_WORKERS)
    time.sleep(2.5)  # wait for both to bind

    try:
        # ── Warmup ────────────────────────────────────────────────────────
        for _, py_path, zp_path in ENDPOINT_PAIRS:
            bench_endpoint(HOST, PORT_1W, py_path, n_concurrent=1, n_total=2)
            bench_endpoint(HOST, PORT_1W, zp_path, n_concurrent=1, n_total=2)
            bench_endpoint(HOST, PORT_NW, py_path, n_concurrent=2, n_total=4)
            bench_endpoint(HOST, PORT_NW, zp_path, n_concurrent=2, n_total=4)

        # ── Collect all results first ──────────────────────────────────────
        results = []
        for label, py_path, zp_path in ENDPOINT_PAIRS:
            r1_py, _ = bench_endpoint(HOST, PORT_1W, py_path,
                                      n_concurrent=N_CONCURRENT_1W, n_total=N_REQUESTS_1W)
            r1_zp, _ = bench_endpoint(HOST, PORT_1W, zp_path,
                                      n_concurrent=N_CONCURRENT_1W, n_total=N_REQUESTS_1W)
            rn_py, _ = bench_endpoint(HOST, PORT_NW, py_path,
                                      n_concurrent=N_CONCURRENT_NW, n_total=N_REQUESTS_NW)
            rn_zp, _ = bench_endpoint(HOST, PORT_NW, zp_path,
                                      n_concurrent=N_CONCURRENT_NW, n_total=N_REQUESTS_NW)
            results.append((label, r1_py, r1_zp, rn_py, rn_zp))

        # ── Print ─────────────────────────────────────────────────────────
        W = 64

        print(f"\nN_floats={N_FLOATS:,}  N_dicts={N_DICTS:,}  "
              f"1w: concurrent={N_CONCURRENT_1W}  "
              f"{N_WORKERS}w: concurrent={N_CONCURRENT_NW}\n")

        # Table 1: per-worker GIL speedup (1 worker)
        print(f"── Q1: Is ZPyFlow faster per worker? (1 worker, concurrent={N_CONCURRENT_1W}) {'─'*12}")
        print(f"  {'scenario':<28s}  {'Python':>7s}  {'ZPyFlow':>8s}  {'speedup':>8s}")
        print(f"  {'─'*28}  {'─'*7}  {'─'*8}  {'─'*8}")
        for label, r1_py, r1_zp, rn_py, rn_zp in results:
            note = "  ← lambda, GIL held" if "object" in label else ""
            print(f"  {label:<28s}  {r1_py:>6.0f}/s  {r1_zp:>7.0f}/s  {r1_zp/r1_py:>7.2f}x{note}")

        # Table 2: multi-worker scaling
        print(f"\n── Q2: Does it scale with workers? (1w → {N_WORKERS}w, concurrent ×{N_WORKERS}) {'─'*8}")
        print(f"  {'scenario':<28s}  {'Py 1w→Nw':>14s}  {'ZPy 1w→Nw':>15s}  {'Py scale':>9s}  {'ZPy scale':>9s}")
        print(f"  {'─'*28}  {'─'*14}  {'─'*15}  {'─'*9}  {'─'*9}")
        for label, r1_py, r1_zp, rn_py, rn_zp in results:
            sc_py = rn_py / r1_py
            sc_zp = rn_zp / r1_zp
            warn_py = " ⚠" if sc_py < 2.0 else ""
            warn_zp = " ⚠" if sc_zp < 2.0 else ""
            print(f"  {label:<28s}  "
                  f"{r1_py:>5.0f}→{rn_py:<6.0f}/s  "
                  f"{r1_zp:>6.0f}→{rn_zp:<6.0f}/s  "
                  f"{sc_py:>8.2f}x{warn_py:<2s}  "
                  f"{sc_zp:>8.2f}x{warn_zp}")

        print(f"\n  ⚠ = scales less than 2x (ideal = {N_WORKERS}.0x)")
        print("  ZPyFlow SIMD ops saturate CPU per worker → adding workers")
        print("  has diminishing returns; scale out ECS tasks instead.")

    finally:
        p1w.terminate()
        pnw.terminate()
        p1w.wait()
        pnw.wait()

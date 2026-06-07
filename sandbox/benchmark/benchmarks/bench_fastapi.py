# bench_fastapi.py — ZPyFlow in a FastAPI sync endpoint under concurrent load
#
# FastAPI runs `def` (sync) endpoints in a thread pool.
# ZPyFlow numeric DSL releases the GIL → threads don't serialize → higher RPS.
# Object pipelines using Python lambdas hold the GIL (shown for comparison).
#
# Setup:
#   pip install fastapi uvicorn httpx
#
# Run:
#   python sandbox/benchmark/benchmarks/bench_fastapi.py
#   python sandbox/benchmark/run.py --suite fastapi

import time
import threading
import concurrent.futures

try:
    import fastapi
    import uvicorn
    import httpx
except ImportError:
    print("Missing dependencies: pip install fastapi uvicorn httpx")
    raise SystemExit(1)

try:
    from zpyflow import Query, col, agg_count, agg_sum
except ImportError:
    print("zpyflow not built — run: maturin develop --profile dev-release")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Shared data (loaded once; endpoints read-only)
# ---------------------------------------------------------------------------

N_FLOATS = 200_000
N_DICTS  = 100_000

floats = [float(i % 1000) for i in range(N_FLOATS)]
dicts  = [{"status": "active" if i % 3 != 0 else "inactive",
           "value":  float(i % 1000),
           "cat":    str(i % 10)}
          for i in range(N_DICTS)]

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = fastapi.FastAPI()


# 1. filter + count (numeric)
@app.get("/python/filter_count")
def ep_py_filter_count():
    return {"result": sum(1 for x in floats if x > 500)}

@app.get("/zpyflow/filter_count")
def ep_zp_filter_count():
    return {"result": Query(floats).filter(col > 500).count()}


# 2. filter + sum (numeric)
@app.get("/python/filter_sum")
def ep_py_filter_sum():
    return {"result": sum(x for x in floats if x > 500)}

@app.get("/zpyflow/filter_sum")
def ep_zp_filter_sum():
    return {"result": Query(floats).filter(col > 500).sum()}


# 3. stats() — 7-stat aggregation in one pass
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


# 4. object filter (dict, lambda — GIL held)
@app.get("/python/object_filter")
def ep_py_object_filter():
    result = [r for r in dicts if r["status"] == "active" and r["value"] > 300]
    return {"count": len(result)}

@app.get("/zpyflow/object_filter")
def ep_zp_object_filter():
    result = Query(dicts).filter(lambda r: r["status"] == "active" and r["value"] > 300).to_list()
    return {"count": len(result)}


# 5. chained: filter → map → sum (numeric)
@app.get("/python/chained")
def ep_py_chained():
    return {"result": sum(x * 2 for x in floats if x > 500)}

@app.get("/zpyflow/chained")
def ep_zp_chained():
    return {"result": Query(floats).filter(col > 500).map(col * 2).sum()}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

HOST = "127.0.0.1"
PORT = 18765


def _start_server():
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="error")
    uvicorn.Server(config).run()


server_thread = threading.Thread(target=_start_server, daemon=True)
server_thread.start()
time.sleep(1.2)  # wait for uvicorn to bind

# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

N_CONCURRENT = 8
N_REQUESTS   = 48  # multiple of N_CONCURRENT


def bench_endpoint(path, n_concurrent=N_CONCURRENT, n_total=N_REQUESTS):
    url = f"http://{HOST}:{PORT}{path}"

    def one():
        t0 = time.perf_counter()
        httpx.get(url, timeout=10)
        return time.perf_counter() - t0

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as ex:
        t0 = time.perf_counter()
        latencies = list(ex.map(lambda _: one(), range(n_total)))
        wall = time.perf_counter() - t0

    return n_total / wall, sum(latencies) / len(latencies) * 1000  # rps, avg_ms


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

ENDPOINT_PAIRS = [
    ("filter+count (numeric)", "/python/filter_count", "/zpyflow/filter_count"),
    ("filter+sum   (numeric)", "/python/filter_sum",   "/zpyflow/filter_sum"),
    ("stats()      (numeric)", "/python/stats",        "/zpyflow/stats"),
    ("object filter  (dict) ", "/python/object_filter","/zpyflow/object_filter"),
    ("chained f→map→sum     ", "/python/chained",      "/zpyflow/chained"),
]

print(f"N_floats={N_FLOATS:,}  N_dicts={N_DICTS:,}  "
      f"concurrent={N_CONCURRENT}  total={N_REQUESTS} req/scenario\n")

# warmup
for _, py_path, zp_path in ENDPOINT_PAIRS:
    bench_endpoint(py_path, n_concurrent=1, n_total=2)
    bench_endpoint(zp_path, n_concurrent=1, n_total=2)

print(f"{'scenario':<28s}  {'Python RPS':>10s}  {'ZPyFlow RPS':>11s}  {'speedup':>8s}  {'py lat':>8s}  {'zp lat':>8s}")
print("─" * 90)

for label, py_path, zp_path in ENDPOINT_PAIRS:
    rps_py, lat_py = bench_endpoint(py_path)
    rps_zp, lat_zp = bench_endpoint(zp_path)
    speedup = rps_zp / rps_py
    print(f"  {label:<26s}  {rps_py:>10.1f}  {rps_zp:>11.1f}  {speedup:>7.2f}x  "
          f"{lat_py:>6.1f}ms  {lat_zp:>6.1f}ms")

print()
print("Note: numeric DSL endpoints release the GIL → ZPyFlow RPS > Python RPS.")
print("      object filter uses Python lambda → GIL held → speedup ≈ 1.0.")

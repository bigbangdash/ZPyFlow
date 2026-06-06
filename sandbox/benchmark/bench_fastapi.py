# bench_fastapi.py — ZPyFlow in a FastAPI sync endpoint under concurrent load
#
# FastAPI runs `def` (sync) endpoints in a thread pool.
# This means multiple requests execute in parallel threads.
# ZPyFlow releases the GIL during Rust execution → threads don't serialize.
#
# Setup:
#   pip install fastapi uvicorn httpx
#
# Run:
#   python sandbox/benchmark/bench_fastapi.py

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
    from zpyflow import Query, col
except ImportError:
    print("zpyflow not built — run: maturin develop --profile dev-release")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

DATA_SIZE = 200_000
data = [float(i % 1000) for i in range(DATA_SIZE)]

app = fastapi.FastAPI()


@app.get("/python")
def endpoint_python():
    """Sync endpoint — generator sum over 200 k floats, one Python object per element."""
    result = sum(1 for x in data if x > 500)
    return {"count": result}


@app.get("/zpyflow")
def endpoint_zpyflow():
    """Sync endpoint — SIMD Rust kernel, no Python objects constructed per element."""
    result = Query(data).filter(col > 500).count()
    return {"count": result}


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------

HOST = "127.0.0.1"
PORT = 18765


def start_server():
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="error")
    server = uvicorn.Server(config)
    server.run()


server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()
time.sleep(1.0)  # wait for uvicorn to bind


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

N_CONCURRENT = 8
N_REQUESTS = 40


def send_requests(path, n_concurrent, n_total):
    url = f"http://{HOST}:{PORT}{path}"
    times = []

    def one():
        t0 = time.perf_counter()
        httpx.get(url, timeout=10)
        return time.perf_counter() - t0

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as ex:
        t_wall_0 = time.perf_counter()
        latencies = list(ex.map(lambda _: one(), range(n_total)))
        wall = time.perf_counter() - t_wall_0

    rps = n_total / wall
    avg_ms = sum(latencies) / len(latencies) * 1000
    return rps, avg_ms


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

print(f"DATA_SIZE={DATA_SIZE:,}  concurrent={N_CONCURRENT}  total={N_REQUESTS}\n")

# warmup
send_requests("/python", 1, 2)
send_requests("/zpyflow", 1, 2)

rps_py, lat_py = send_requests("/python", N_CONCURRENT, N_REQUESTS)
rps_zp, lat_zp = send_requests("/zpyflow", N_CONCURRENT, N_REQUESTS)

print(f"{'endpoint':20s}  {'RPS':>8s}  {'avg latency':>12s}")
print("-" * 46)
print(f"{'GET /python':20s}  {rps_py:>8.1f}  {lat_py:>10.1f}ms")
print(f"{'GET /zpyflow':20s}  {rps_zp:>8.1f}  {lat_zp:>10.1f}ms")
print(f"\nZPyFlow RPS advantage: {rps_zp/rps_py:.2f}x")
print()
print("Expected: ZPyFlow higher RPS because the Rust kernel processes floats without")
print("constructing Python objects per element (SIMD speed, not GIL release).")

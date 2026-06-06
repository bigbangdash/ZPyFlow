# bench_multiprocess.py — scaling with ProcessPoolExecutor
#
# Each process has its own GIL, so both Python and ZPyFlow scale.
# Key difference: ZPyFlow is faster per-process (SIMD), so absolute
# throughput is higher even at the same process count.
#
# Also shows the data-pickling overhead that multiprocessing adds
# compared to threading — use threads when data is already in-process.
#
# Run:
#   python sandbox/benchmark/bench_multiprocess.py

import time
import concurrent.futures

try:
    from zpyflow import Query, col
    HAS_ZPYFLOW = True
except ImportError:
    print("zpyflow not built — run: maturin develop --profile dev-release")
    raise SystemExit(1)

DATA_SIZE = 500_000
N_WORKERS = 4
data = [float(i % 1000) for i in range(DATA_SIZE)]


# Top-level functions required for pickling
def python_task(d):
    return sum(1 for x in d if x > 500)


def zpyflow_task(d):
    from zpyflow import Query, col  # re-import inside worker process
    return Query(d).filter(col > 500).count()


def run_parallel(fn, n_workers, repeat=5):
    times = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
        for _ in range(repeat):
            t0 = time.perf_counter()
            list(ex.map(fn, [data] * n_workers))
            times.append(time.perf_counter() - t0)
    return min(times)


def run_single(fn, repeat=5):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(data)
        times.append(time.perf_counter() - t0)
    return min(times)


if __name__ == "__main__":  # required for multiprocessing on macOS/Windows
    print(f"DATA_SIZE={DATA_SIZE:,}  N_WORKERS={N_WORKERS}  (best of 5 runs)\n")

    t_py_1 = run_single(python_task)
    t_zp_1 = run_single(zpyflow_task)
    t_py_n = run_parallel(python_task, N_WORKERS)
    t_zp_n = run_parallel(zpyflow_task, N_WORKERS)

    print(f"{'':30s} {'1 process':>10s}  {f'{N_WORKERS} processes':>12s}  {'speedup':>8s}")
    print("-" * 68)
    print(f"{'Python list comp':30s} {t_py_1*1000:>9.1f}ms  {t_py_n*1000:>11.1f}ms  {t_py_1/t_py_n:>7.2f}x")
    print(f"{'ZPyFlow DSL':30s} {t_zp_1*1000:>9.1f}ms  {t_zp_n*1000:>11.1f}ms  {t_zp_1/t_zp_n:>7.2f}x")
    print()
    print("Note: multiprocess times include pickle overhead for data transfer.")
    print("Compare with bench_threading.py to see the pickling cost.")

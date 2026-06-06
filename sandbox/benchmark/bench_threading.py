# bench_threading.py — GIL release under thread contention
#
# Point: Python list comp holds the GIL → N threads serialize.
#        ZPyFlow DSL releases the GIL → N threads run in parallel.
#
# Run:
#   python sandbox/benchmark/bench_threading.py

import time
import concurrent.futures

try:
    from zpyflow import Query, col
    HAS_ZPYFLOW = True
except ImportError:
    HAS_ZPYFLOW = False
    print("zpyflow not built — run: maturin develop --profile dev-release")
    raise SystemExit(1)

DATA_SIZE = 500_000
N_THREADS = 4
data = [float(i % 1000) for i in range(DATA_SIZE)]


def python_task(d):
    return sum(1 for x in d if x > 500)


def zpyflow_task(d):
    return Query(d).filter(col > 500).count()


def run_parallel(fn, n_threads, repeat=5):
    times = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
        for _ in range(repeat):
            t0 = time.perf_counter()
            list(ex.map(fn, [data] * n_threads))
            times.append(time.perf_counter() - t0)
    return min(times)  # best of N (same as pytest-benchmark default)


def run_single(fn, repeat=5):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(data)
        times.append(time.perf_counter() - t0)
    return min(times)


print(f"DATA_SIZE={DATA_SIZE:,}  N_THREADS={N_THREADS}  (best of 5 runs)\n")

t_py_1 = run_single(python_task)
t_zp_1 = run_single(zpyflow_task)
t_py_n = run_parallel(python_task, N_THREADS)
t_zp_n = run_parallel(zpyflow_task, N_THREADS)

print(f"{'':30s} {'1 thread':>10s}  {f'{N_THREADS} threads':>10s}  {'speedup':>8s}")
print("-" * 65)
print(f"{'Python list comp':30s} {t_py_1*1000:>9.1f}ms  {t_py_n*1000:>9.1f}ms  {t_py_1/t_py_n:>7.2f}x")
print(f"{'ZPyFlow DSL':30s} {t_zp_1*1000:>9.1f}ms  {t_zp_n*1000:>9.1f}ms  {t_zp_1/t_zp_n:>7.2f}x")
print()
print("Expected:")
print(f"  Python:  speedup ≈ 1.0  (GIL serializes threads)")
print(f"  ZPyFlow: speedup ≈ {N_THREADS}.0  (GIL released → true parallelism)")

//! Benchmark suite for ZPyFlow's Rust core.
//!
//! Run with:
//!   cargo bench --bench pipeline -- --output-format html
//!
//! Or for a specific benchmark group:
//!   cargo bench --bench pipeline filter
//!
//! Baseline comparison strategy:
//!   - Rust std iterator (theoretical minimum allocation baseline)
//!   - ZPyFlow NumericPipeline (our zero-allocation path)
//!   - ZPyFlow NumericPipeline + SIMD
//!   - ZPyFlow parallel (rayon)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use _zpyflow::pipeline::{
    numeric::{IntOp, IntPipeline, NumericOp, NumericPipeline},
    sources::SliceStream,
    traits::ZStream,
};
use _zpyflow::simd::{simd_dot_product_f64, simd_filter_gt, simd_map_mul_inplace, simd_sum_f64};

const SIZES: &[usize] = &[1_000, 10_000, 100_000, 1_000_000, 10_000_000];

// ---------------------------------------------------------------------------
// Helper: generate reproducible test data
// ---------------------------------------------------------------------------

fn generate_f64(n: usize) -> Vec<f64> {
    // Simple LCG to avoid dependency on rand
    let mut x: u64 = 0x123456789ABCDEF0;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x as f64) / u64::MAX as f64 * 200.0 - 100.0 // range [-100, 100]
        })
        .collect()
}

fn generate_i64(n: usize) -> Vec<i64> {
    let mut x: u64 = 0xFEDCBA9876543210;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x as i64) % 1_000_000
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 1. Filter benchmark — compare pipelines
// ---------------------------------------------------------------------------

fn bench_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_f64");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        let data = generate_f64(size);
        let threshold = 0.0f64;

        // Baseline: std iterator (the compiler's own fusion)
        group.bench_with_input(BenchmarkId::new("std_iterator", size), &size, |b, _| {
            b.iter(|| {
                let d = data.clone();
                d.into_iter().filter(|&x| x > threshold).collect::<Vec<_>>()
            });
        });

        // ZPyFlow NumericPipeline (scalar fused path)
        group.bench_with_input(BenchmarkId::new("zpyflow_scalar", size), &size, |b, _| {
            b.iter(|| {
                NumericPipeline::new(data.clone())
                    .push_op(NumericOp::FilterGt(threshold))
                    .execute()
            });
        });

        // ZPyFlow SIMD path
        group.bench_with_input(BenchmarkId::new("zpyflow_simd", size), &size, |b, _| {
            b.iter(|| simd_filter_gt(black_box(&data), black_box(threshold)));
        });

        // Slice-based ZStream (monomorphized, stack-allocated pipeline state)
        group.bench_with_input(BenchmarkId::new("zstream_slice", size), &size, |b, _| {
            b.iter(|| {
                SliceStream::new(&data)
                    .zfilter(|x| *x > threshold)
                    .zcollect_vec()
            });
        });

        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("zpyflow_parallel", size), &size, |b, _| {
            b.iter(|| {
                _zpyflow::parallel::engine::par_execute_f64(
                    data.clone(),
                    &[NumericOp::FilterGt(threshold)],
                )
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. Chained pipeline: filter + map + take
// ---------------------------------------------------------------------------

fn bench_chained(c: &mut Criterion) {
    let mut group = c.benchmark_group("chained_filter_map_take");

    for &size in &[100_000, 1_000_000, 10_000_000] {
        group.throughput(Throughput::Elements(size as u64));
        let data = generate_f64(size);

        // std iterator baseline
        group.bench_with_input(BenchmarkId::new("std_iterator", size), &size, |b, _| {
            b.iter(|| {
                data.iter()
                    .copied()
                    .filter(|&x| x > 0.0)
                    .map(|x| x * 2.0)
                    .take(1000)
                    .collect::<Vec<_>>()
            });
        });

        // ZPyFlow fused (3 ops in 1 pass, no intermediate allocation)
        group.bench_with_input(BenchmarkId::new("zpyflow_fused", size), &size, |b, _| {
            b.iter(|| {
                NumericPipeline::new(data.clone())
                    .push_op(NumericOp::FilterGt(0.0))
                    .push_op(NumericOp::MapMulScalar(2.0))
                    .push_op(NumericOp::Take(1000))
                    .execute()
            });
        });

        // ZStream monomorphized (fully inlined state machine)
        group.bench_with_input(BenchmarkId::new("zstream_fused", size), &size, |b, _| {
            b.iter(|| {
                SliceStream::new(&data)
                    .zfilter(|&x| x > 0.0)
                    .zmap(|x| x * 2.0)
                    .ztake(1000)
                    .zcollect_vec()
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. SIMD arithmetic throughput
// ---------------------------------------------------------------------------

fn bench_simd_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_arithmetic");

    for &size in &[10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(size as u64));
        let mut data = generate_f64(size);

        group.bench_with_input(BenchmarkId::new("simd_mul_inplace", size), &size, |b, _| {
            b.iter(|| simd_map_mul_inplace(black_box(&mut data), black_box(2.0)));
        });

        group.bench_with_input(
            BenchmarkId::new("scalar_mul_inplace", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let d = black_box(&mut data);
                    for x in d.iter_mut() {
                        *x *= 2.0;
                    }
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("simd_sum", size), &size, |b, _| {
            b.iter(|| simd_sum_f64(black_box(&data)));
        });

        group.bench_with_input(BenchmarkId::new("scalar_sum", size), &size, |b, _| {
            b.iter(|| black_box(&data).iter().copied().sum::<f64>());
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Dot product (AI/ML use case — embedding similarity)
// ---------------------------------------------------------------------------

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");
    let dims = &[128, 256, 512, 1536, 3072]; // common embedding dimensions

    for &dim in dims {
        group.throughput(Throughput::Elements(dim as u64));
        let a = generate_f64(dim);
        let b = generate_f64(dim);

        group.bench_with_input(BenchmarkId::new("simd", dim), &dim, |b_fn, _| {
            b_fn.iter(|| simd_dot_product_f64(black_box(&a), black_box(&b)));
        });

        group.bench_with_input(BenchmarkId::new("scalar_zip", dim), &dim, |b_fn, _| {
            b_fn.iter(|| {
                black_box(&a)
                    .iter()
                    .zip(black_box(&b).iter())
                    .map(|(x, y)| x * y)
                    .sum::<f64>()
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 5. Integer pipeline — common in ETL/event processing
// ---------------------------------------------------------------------------

fn bench_integer_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("integer_filter_map");

    for &size in &[100_000, 1_000_000] {
        group.throughput(Throughput::Elements(size as u64));
        let data = generate_i64(size);

        group.bench_with_input(BenchmarkId::new("zpyflow_i64", size), &size, |b, _| {
            b.iter(|| {
                IntPipeline::new(data.clone())
                    .push_op(IntOp::FilterGt(0))
                    .push_op(IntOp::MapMulScalar(2))
                    .execute()
            });
        });

        group.bench_with_input(BenchmarkId::new("std_iterator", size), &size, |b, _| {
            b.iter(|| {
                data.iter()
                    .copied()
                    .filter(|&x| x > 0)
                    .map(|x| x.wrapping_mul(2))
                    .collect::<Vec<_>>()
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 6. Allocation stress test — how much does clone cost?
// ---------------------------------------------------------------------------

fn bench_allocation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_overhead");
    let size = 1_000_000;
    let data = generate_f64(size);

    // Measure clone cost (paid per NumericPipeline.execute() currently)
    group.bench_function("vec_clone_1m", |b| {
        b.iter(|| black_box(data.clone()));
    });

    // Measure collect cost (paid at terminal operations)
    group.bench_function("iter_collect_1m", |b| {
        b.iter(|| black_box(data.iter().copied().collect::<Vec<_>>()));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_filter,
    bench_chained,
    bench_simd_arithmetic,
    bench_dot_product,
    bench_integer_pipeline,
    bench_allocation_overhead,
);
criterion_main!(benches);

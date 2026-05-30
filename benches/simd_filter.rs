//! Dedicated SIMD filter benchmark — measures lane utilization.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use _zpyflow::simd::{simd_filter_gt, simd_filter_gt_f32};

fn generate_uniform(n: usize, lo: f64, hi: f64) -> Vec<f64> {
    let mut x: u64 = 0xDEADBEEFCAFEBABE;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) as f64 / (u32::MAX as f64) * (hi - lo) + lo
        })
        .collect()
}

fn bench_filter_selectivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_selectivity_1M");
    let size = 1_000_000usize;
    group.throughput(Throughput::Elements(size as u64));

    // Test different selectivities: what % of data passes the filter
    // Low selectivity = most filtered = more branch mispredictions
    for selectivity_pct in [10, 25, 50, 75, 90] {
        let lo = -100.0f64;
        let hi = 100.0f64;
        // threshold such that `selectivity_pct`% of uniform[-100,100] passes
        let threshold = lo + (hi - lo) * (1.0 - selectivity_pct as f64 / 100.0);
        let data = generate_uniform(size, lo, hi);

        group.bench_with_input(
            BenchmarkId::new("simd_gt", format!("{}%", selectivity_pct)),
            &selectivity_pct,
            |b, _| {
                b.iter(|| simd_filter_gt(black_box(&data), black_box(threshold)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar_gt", format!("{}%", selectivity_pct)),
            &selectivity_pct,
            |b, _| {
                b.iter(|| {
                    black_box(&data)
                        .iter()
                        .copied()
                        .filter(|&x| x > threshold)
                        .collect::<Vec<_>>()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_filter_selectivity);
criterion_main!(benches);

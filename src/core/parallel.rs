//! Parallel execution utilities — bench path only.
//!
//! ## Responsibility split
//!
//! **Production path** (query.rs → Python API):
//!   `NumericPipeline::execute_parallel_with_skip_take` in `pipeline/numeric.rs`.
//!   Activated via `.parallel()` on a `PyQuery`.  Takes ownership of the
//!   pipeline, extracts skip/take bounds, then fans out via Rayon.
//!
//! **Bench path** (benches/pipeline.rs):
//!   `engine::parallel_execute_f64_pipeline` / `parallel_execute_i64_pipeline` below.
//!   Simpler entry points that take raw `Vec<T>` + ops — useful for
//!   microbenchmarks that want to drive the parallel kernel without going
//!   through the full Python wrapper stack.

#[cfg(feature = "parallel")]
pub mod engine {
    use crate::core::numeric::pipeline::{apply_int_op, apply_scalar_op, IntOp, NumericOp, ScalarResult};
    use rayon::prelude::*;

    /// Bench utility: filter/map a f64 vec in parallel.
    /// Production code uses `NumericPipeline::execute_parallel_with_skip_take` instead.
    pub fn parallel_execute_f64_pipeline(data: Vec<f64>, ops: &[NumericOp]) -> Vec<f64> {
        let ops_arc = std::sync::Arc::new(ops.to_vec());
        data.into_par_iter()
            .filter_map(move |mut val| {
                for op in ops_arc.iter() {
                    match apply_scalar_op(val, op) {
                        ScalarResult::Value(v) => val = v,
                        ScalarResult::Filtered => return None,
                    }
                }
                Some(val)
            })
            .collect()
    }

    /// Bench utility: filter/map an i64 vec in parallel.
    /// Production code uses `IntPipeline::execute_parallel_with_skip_take` instead.
    pub fn parallel_execute_i64_pipeline(data: Vec<i64>, ops: &[IntOp]) -> Vec<i64> {
        let ops_arc = std::sync::Arc::new(ops.to_vec());
        data.into_par_iter()
            .filter_map(move |mut val| {
                for op in ops_arc.iter() {
                    match apply_int_op(val, op) {
                        Some(v) => val = v,
                        None => return None,
                    }
                }
                Some(val)
            })
            .collect()
    }
}

// Stub when parallel feature disabled
#[cfg(not(feature = "parallel"))]
pub mod engine {
    use crate::core::numeric::pipeline::NumericOp;

    pub fn parallel_execute_f64_pipeline(data: Vec<f64>, _ops: &[NumericOp]) -> Vec<f64> {
        crate::core::numeric::pipeline::NumericPipeline::new(data).execute()
    }
}

pub mod numeric;
pub mod obj;
pub mod parallel;
pub mod sources;
pub mod traits;

pub use numeric::pipeline::{
    collapse_ops, count_fused_f32_bounded, count_fused_f64_bounded, count_fused_i64_bounded,
    eval_filter_f64, eval_filter_i64, execute_fused_f32_bounded, execute_fused_f64,
    execute_fused_f64_bounded, execute_fused_i64_bounded, filter_max_fused_f64,
    filter_mean_fused_f64, filter_min_fused_f64, filter_multi_stat_f64, filter_sum_fused_f64,
    filter_var_fused_f64, max_fused_f64_bounded, mean_fused_f64_bounded, min_fused_f64_bounded,
    stats_fused_f64_bounded, sum_fused_f64_bounded, var_fused_f64_bounded, IntOp, IntPipeline,
    NumericOp, NumericPipeline,
};
pub use obj::pipeline::{
    count_obj_pipeline, execute_obj_pipeline, row_passes, sum_field_obj_pipeline, ObjOp, RustRow,
    RustValue,
};
pub use sources::*;
pub use traits::*;
// simd functions used by the Python layer
pub use numeric::simd::{simd_max_f32, simd_sum_f32};

use crate::{
    CompilerStateId,
    error::{NATIVE_ENGINE_ERROR_CODE, bounded_message, internal_error},
};
use graphs_compiler::engine::{
    CompilerEngineError as NativeCompilerEngineError, CompilerMeasurement,
};
use gz_engine::{
    EngineResult, ErrorCode, GraphHash, LatencyStats, MeasureConfigHash, MeasureFailure,
    MeasureMetadata, MeasureResult,
};
use std::time::Duration;

pub const ACCURACY_FAILURE_REWARD: f32 = -10_000.0;

pub const COMPILE_FAILURE_CODE: ErrorCode = ErrorCode::new(1_001);
pub const ACCURACY_EVALUATION_FAILURE_CODE: ErrorCode = ErrorCode::new(1_002);
pub const PERFORMANCE_EVALUATION_FAILURE_CODE: ErrorCode = ErrorCode::new(1_003);
pub const COMPILER_MEASUREMENT_FAILURE_CODE: ErrorCode = ErrorCode::new(1_004);

pub(crate) fn project_measurement(
    graph: CompilerStateId,
    graph_hash: GraphHash,
    config_hash: MeasureConfigHash,
    measurement: Result<CompilerMeasurement, NativeCompilerEngineError>,
) -> EngineResult<MeasureResult<CompilerStateId>> {
    match measurement {
        Ok(CompilerMeasurement::Passed { performance, .. }) => {
            let execution = &performance.execution;
            let samples_ms = execution.samples.iter().copied().map(duration_ms).collect();
            let latency = LatencyStats::new(
                duration_ms(execution.mean),
                duration_ms(execution.median),
                duration_ms(execution.p95),
                samples_ms,
            )
            .map_err(|error| {
                internal_error(
                    NATIVE_ENGINE_ERROR_CODE,
                    format!("compiler returned invalid latency: {error}"),
                )
            })?;

            Ok(MeasureResult {
                graph,
                graph_hash,
                config_hash,
                measured: true,
                valid: true,
                scalar_reward: Some(-latency.median_ms),
                latency: Some(latency),
                failure: None,
                metadata: MeasureMetadata::default(),
            })
        }
        Ok(CompilerMeasurement::AccuracyFailed { .. }) => Ok(MeasureResult {
            graph,
            graph_hash,
            config_hash,
            measured: true,
            valid: true,
            latency: None,
            scalar_reward: Some(ACCURACY_FAILURE_REWARD),
            failure: None,
            metadata: MeasureMetadata::default(),
        }),
        Err(error) => Ok(invalid_measurement(graph, graph_hash, config_hash, error)),
    }
}

fn invalid_measurement(
    graph: CompilerStateId,
    graph_hash: GraphHash,
    config_hash: MeasureConfigHash,
    error: NativeCompilerEngineError,
) -> MeasureResult<CompilerStateId> {
    let code = match error {
        NativeCompilerEngineError::Compile(_) => COMPILE_FAILURE_CODE,
        NativeCompilerEngineError::AccuracyEvaluation(_) => ACCURACY_EVALUATION_FAILURE_CODE,
        NativeCompilerEngineError::PerformanceEvaluation(_) => PERFORMANCE_EVALUATION_FAILURE_CODE,
        _ => COMPILER_MEASUREMENT_FAILURE_CODE,
    };

    MeasureResult {
        graph,
        graph_hash,
        config_hash,
        measured: true,
        valid: false,
        latency: None,
        scalar_reward: None,
        failure: Some(MeasureFailure {
            code,
            message: bounded_message(error.to_string()),
        }),
        metadata: MeasureMetadata::default(),
    }
}

fn duration_ms(duration: Duration) -> f32 {
    (duration.as_secs_f64() * 1_000.0) as f32
}

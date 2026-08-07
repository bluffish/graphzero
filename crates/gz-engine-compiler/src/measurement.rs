use crate::{
    CompilerStateId,
    error::{NATIVE_ENGINE_ERROR_CODE, bounded_message, error_chain, internal_error},
};
use graphs_compiler::engine::{
    CompilerEngineError as NativeCompilerEngineError, CompilerMeasurement,
};
use graphs_compiler::evaluators::PerformanceReport;
use gz_engine::{
    EngineResult, ErrorCode, GraphHash, LatencyStats, MeasureConfigHash, MeasureFailure,
    MeasureMetadata, MeasureResult,
};
use std::time::Duration;

const EVIDENCE_MAGIC: &[u8; 4] = b"GCE1";
const EVIDENCE_HEADER_BYTES: usize = 12;
const EVIDENCE_SAMPLE_BYTES: usize = 16;
const EVIDENCE_PASSED: u8 = 1;
const EVIDENCE_ACCURACY_FAILED: u8 = 2;

pub const ACCURACY_FAILURE_REWARD: f32 = -10_000.0;

pub const COMPILE_FAILURE_CODE: ErrorCode = ErrorCode::new(1_001);
pub const ACCURACY_EVALUATION_FAILURE_CODE: ErrorCode = ErrorCode::new(1_002);
pub const PERFORMANCE_EVALUATION_FAILURE_CODE: ErrorCode = ErrorCode::new(1_003);
pub const COMPILER_MEASUREMENT_FAILURE_CODE: ErrorCode = ErrorCode::new(1_004);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompilerMeasurementEvidence {
    pub execution_ns: Vec<u64>,
    pub end_to_end_ns: Vec<u64>,
}

impl CompilerMeasurementEvidence {
    pub fn encode(&self) -> EngineResult<MeasureMetadata> {
        if self.execution_ns.len() != self.end_to_end_ns.len() {
            return Err(internal_error(
                NATIVE_ENGINE_ERROR_CODE,
                "compiler execution and end-to-end sample counts differ",
            ));
        }
        let count = u32::try_from(self.execution_ns.len()).map_err(|_| {
            internal_error(
                NATIVE_ENGINE_ERROR_CODE,
                "compiler measurement sample count exceeds u32",
            )
        })?;
        let capacity = 12_usize
            .checked_add(
                self.execution_ns
                    .len()
                    .checked_mul(EVIDENCE_SAMPLE_BYTES)
                    .ok_or_else(|| {
                        internal_error(
                            NATIVE_ENGINE_ERROR_CODE,
                            "compiler measurement evidence size overflows",
                        )
                    })?,
            )
            .ok_or_else(|| {
                internal_error(
                    NATIVE_ENGINE_ERROR_CODE,
                    "compiler measurement evidence size overflows",
                )
            })?;
        let mut bytes = Vec::with_capacity(capacity);
        bytes.extend_from_slice(EVIDENCE_MAGIC);
        bytes.push(EVIDENCE_PASSED);
        bytes.extend_from_slice(&[0, 0, 0]);
        bytes.extend_from_slice(&count.to_le_bytes());
        for (execution, end_to_end) in self.execution_ns.iter().zip(&self.end_to_end_ns) {
            bytes.extend_from_slice(&execution.to_le_bytes());
            bytes.extend_from_slice(&end_to_end.to_le_bytes());
        }
        Ok(MeasureMetadata { bytes })
    }
}

#[must_use]
pub fn accuracy_failure_evidence() -> MeasureMetadata {
    let mut bytes = Vec::with_capacity(12);
    bytes.extend_from_slice(EVIDENCE_MAGIC);
    bytes.push(EVIDENCE_ACCURACY_FAILED);
    bytes.extend_from_slice(&[0, 0, 0]);
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    MeasureMetadata { bytes }
}

pub fn decode_measurement_evidence(
    metadata: &MeasureMetadata,
) -> EngineResult<Option<CompilerMeasurementEvidence>> {
    let bytes = &metadata.bytes;
    if bytes.len() < EVIDENCE_HEADER_BYTES || &bytes[..4] != EVIDENCE_MAGIC {
        return Err(internal_error(
            NATIVE_ENGINE_ERROR_CODE,
            "compiler measurement evidence is malformed",
        ));
    }
    let kind = bytes[4];
    if bytes[5..8] != [0, 0, 0] {
        return Err(internal_error(
            NATIVE_ENGINE_ERROR_CODE,
            "compiler measurement evidence reserved bytes are nonzero",
        ));
    }
    let count = usize::try_from(u32::from_le_bytes(bytes[8..12].try_into().map_err(
        |_| {
            internal_error(
                NATIVE_ENGINE_ERROR_CODE,
                "compiler measurement evidence is truncated",
            )
        },
    )?))
    .map_err(|_| {
        internal_error(
            NATIVE_ENGINE_ERROR_CODE,
            "compiler measurement evidence sample count is too large",
        )
    })?;
    let expected = 12_usize
        .checked_add(count.checked_mul(EVIDENCE_SAMPLE_BYTES).ok_or_else(|| {
            internal_error(
                NATIVE_ENGINE_ERROR_CODE,
                "compiler measurement evidence size overflows",
            )
        })?)
        .ok_or_else(|| {
            internal_error(
                NATIVE_ENGINE_ERROR_CODE,
                "compiler measurement evidence size overflows",
            )
        })?;
    if bytes.len() != expected {
        return Err(internal_error(
            NATIVE_ENGINE_ERROR_CODE,
            "compiler measurement evidence has the wrong length",
        ));
    }
    match kind {
        EVIDENCE_ACCURACY_FAILED if count == 0 => Ok(None),
        EVIDENCE_PASSED => {
            let mut execution_ns = Vec::with_capacity(count);
            let mut end_to_end_ns = Vec::with_capacity(count);
            for sample in bytes[12..].chunks_exact(EVIDENCE_SAMPLE_BYTES) {
                execution_ns.push(u64::from_le_bytes(sample[..8].try_into().unwrap()));
                end_to_end_ns.push(u64::from_le_bytes(sample[8..].try_into().unwrap()));
            }
            Ok(Some(CompilerMeasurementEvidence {
                execution_ns,
                end_to_end_ns,
            }))
        }
        _ => Err(internal_error(
            NATIVE_ENGINE_ERROR_CODE,
            "compiler measurement evidence kind is invalid",
        )),
    }
}

pub(crate) fn project_measurement(
    graph: CompilerStateId,
    graph_hash: GraphHash,
    config_hash: MeasureConfigHash,
    measurement: Result<CompilerMeasurement, NativeCompilerEngineError>,
) -> EngineResult<MeasureResult<CompilerStateId>> {
    match measurement {
        Ok(CompilerMeasurement::Passed { performance, .. }) => {
            project_performance(graph, graph_hash, config_hash, *performance)
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
            metadata: accuracy_failure_evidence(),
        }),
        Err(error) => Ok(invalid_measurement(graph, graph_hash, config_hash, error)),
    }
}

pub(crate) fn project_performance_only_measurement(
    graph: CompilerStateId,
    graph_hash: GraphHash,
    config_hash: MeasureConfigHash,
    measurement: Result<PerformanceReport, NativeCompilerEngineError>,
) -> EngineResult<MeasureResult<CompilerStateId>> {
    match measurement {
        Ok(performance) => project_performance(graph, graph_hash, config_hash, performance),
        Err(error) => Ok(invalid_measurement(graph, graph_hash, config_hash, error)),
    }
}

fn project_performance(
    graph: CompilerStateId,
    graph_hash: GraphHash,
    config_hash: MeasureConfigHash,
    performance: PerformanceReport,
) -> EngineResult<MeasureResult<CompilerStateId>> {
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
        metadata: passed_evidence(
            &performance.execution.samples,
            &performance.observed.samples,
        )?,
    })
}

fn invalid_measurement(
    graph: CompilerStateId,
    graph_hash: GraphHash,
    config_hash: MeasureConfigHash,
    error: NativeCompilerEngineError,
) -> MeasureResult<CompilerStateId> {
    let code = match error {
        NativeCompilerEngineError::NvidiaCompile(_)
        | NativeCompilerEngineError::IntelCompile(_) => COMPILE_FAILURE_CODE,
        NativeCompilerEngineError::AccuracyEvaluation(_) => ACCURACY_EVALUATION_FAILURE_CODE,
        NativeCompilerEngineError::PerformanceEvaluation(_) => PERFORMANCE_EVALUATION_FAILURE_CODE,
        _ => COMPILER_MEASUREMENT_FAILURE_CODE,
    };

    let message = error_chain(&error);
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
            message: bounded_message(message),
        }),
        metadata: MeasureMetadata::default(),
    }
}

fn duration_ms(duration: Duration) -> f32 {
    (duration.as_secs_f64() * 1_000.0) as f32
}

fn passed_evidence(
    execution: &[Duration],
    end_to_end: &[Duration],
) -> EngineResult<MeasureMetadata> {
    if execution.len() != end_to_end.len() {
        return Err(internal_error(
            NATIVE_ENGINE_ERROR_CODE,
            "compiler execution and end-to-end sample counts differ",
        ));
    }
    CompilerMeasurementEvidence {
        execution_ns: execution
            .iter()
            .copied()
            .map(duration_ns)
            .collect::<EngineResult<Vec<_>>>()?,
        end_to_end_ns: end_to_end
            .iter()
            .copied()
            .map(duration_ns)
            .collect::<EngineResult<Vec<_>>>()?,
    }
    .encode()
}

fn duration_ns(duration: Duration) -> EngineResult<u64> {
    u64::try_from(duration.as_nanos()).map_err(|_| {
        internal_error(
            NATIVE_ENGINE_ERROR_CODE,
            "compiler measurement duration exceeds u64 nanoseconds",
        )
    })
}

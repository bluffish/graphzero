use gz_engine::{GraphEngine, MeasureOptions};
use gz_engine_whittle::WhittleEngine;
use gz_measure_service::{
    Coordinator, CoordinatorConfig, DeviceId, DeviceProfileHash, EncodedMeasureConfig, Enrollment,
    MeasureSubmission, RequestNonce, ServiceError, artifact_descriptor, job_id, measurement_key,
};

#[test]
fn job_identity_reuses_nonce_but_separates_logical_calls() {
    let engine = WhittleEngine::default();
    let graph = engine.root();
    let submission = submission(&engine, graph);
    let descriptor = artifact_descriptor(&submission.artifact).unwrap();
    let key = measurement_key(&submission, descriptor);
    let nonce = RequestNonce::from_bytes([7; 16]);

    assert_eq!(job_id(key, nonce), job_id(key, nonce));
    assert_ne!(
        job_id(key, nonce),
        job_id(key, RequestNonce::from_bytes([8; 16]))
    );
}

#[test]
fn bounded_queue_rejects_without_growing_artifact_storage() {
    let device_id = DeviceId::from_bytes([3; 16]);
    let config = CoordinatorConfig {
        enrollment: Enrollment::one_insecure_test_device(device_id),
        queue_capacity: 1,
        ..CoordinatorConfig::default()
    };
    let coordinator = Coordinator::new(config).unwrap();
    let handle = coordinator.handle();
    let engine = WhittleEngine::default();
    let graph = engine.root();

    let _pending = handle
        .enqueue_with_nonce(
            submission(&engine, graph),
            RequestNonce::from_bytes([1; 16]),
        )
        .unwrap();
    let error = handle
        .enqueue_with_nonce(
            submission(&engine, graph),
            RequestNonce::from_bytes([2; 16]),
        )
        .unwrap_err();

    assert!(matches!(error, ServiceError::Capacity(_)));
    let snapshot = handle.snapshot();
    assert_eq!(snapshot.queued_jobs, 1);
    assert_eq!(snapshot.artifact_items, 1);
}

fn submission(
    engine: &WhittleEngine,
    graph: <WhittleEngine as GraphEngine>::Graph,
) -> MeasureSubmission {
    let base = engine.measure_options();
    let options = MeasureOptions::new(base.config_hash, 1, None, true).unwrap();
    MeasureSubmission::from_engine(
        engine,
        graph,
        options,
        EncodedMeasureConfig {
            encoding: 1,
            payload: vec![1],
        },
        DeviceProfileHash::from_bytes([4; 32]),
    )
    .unwrap()
}

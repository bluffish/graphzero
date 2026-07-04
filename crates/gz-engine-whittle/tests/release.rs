use gz_engine::{CandidateOptions, EngineError, GraphEngine};
use gz_engine_whittle::{WhittleEngine, WhittleEngineConfig, WhittleRoot};

const NO_NODE: u32 = u32::MAX;

fn and_idempotent_artifact() -> Vec<u8> {
    wav1(1, 16, 2, &[(0, 0, NO_NODE), (2, 0, 0), (5, 1, NO_NODE)])
}

fn wav1(arity: u16, capacity: u16, output_node: u32, nodes: &[(i8, u32, u32)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"WAV1");
    bytes.extend_from_slice(&arity.to_le_bytes());
    bytes.extend_from_slice(&capacity.to_le_bytes());
    bytes.extend_from_slice(&(nodes.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&output_node.to_le_bytes());

    for (op, arg0, arg1) in nodes {
        bytes.push(*op as u8);
        bytes.extend_from_slice(&arg0.to_le_bytes());
        bytes.extend_from_slice(&arg1.to_le_bytes());
    }

    bytes
}

fn and_engine() -> WhittleEngine {
    WhittleEngine::new(WhittleEngineConfig {
        root: WhittleRoot::Artifact(and_idempotent_artifact()),
        ..WhittleEngineConfig::default()
    })
    .unwrap()
}

#[test]
fn release_reuses_graph_and_candidate_slots() {
    let mut engine = and_engine();
    let root = engine.root();
    let mut root_candidates = Vec::new();
    engine
        .candidates(root, CandidateOptions::default(), &mut root_candidates)
        .unwrap();
    let first = root_candidates[0];
    let mut graph_slots = Vec::new();
    let mut candidate_slots = Vec::new();

    for _ in 0..8 {
        let applied = engine.apply(root, first).unwrap();
        let mut child_candidates = Vec::new();
        engine
            .candidates(
                applied.after,
                CandidateOptions::default(),
                &mut child_candidates,
            )
            .unwrap();
        graph_slots.push(applied.after.raw());
        candidate_slots.extend(child_candidates.iter().map(|candidate| candidate.raw()));
        engine.release(&[applied.after], &child_candidates).unwrap();
    }

    assert_eq!(graph_slots, vec![1; 8]);
    assert!(candidate_slots.iter().copied().all(|slot| slot < 64));
    assert_eq!(
        engine.candidate_info(root, first).unwrap().graph_hash,
        engine.hash(root).unwrap()
    );
}

#[test]
fn release_invalidates_transition_cache_to_released_graph() {
    let mut engine = and_engine();
    let root = engine.root();
    let mut root_candidates = Vec::new();
    engine
        .candidates(root, CandidateOptions::default(), &mut root_candidates)
        .unwrap();
    let first = root_candidates[0];
    let applied = engine.apply(root, first).unwrap();
    let released_slot = applied.after.raw();

    engine.release(&[applied.after], &[]).unwrap();

    let applied_again = engine.apply(root, first).unwrap();
    assert_eq!(applied_again.after.raw(), released_slot);
    assert_eq!(
        engine.hash(applied_again.after).unwrap(),
        applied.after_hash
    );
}

#[test]
fn releasing_root_errors() {
    let mut engine = and_engine();
    let error = engine.release(&[engine.root()], &[]).unwrap_err();

    assert!(matches!(error, EngineError::Internal { .. }));
    assert!(engine.hash(engine.root()).is_ok());
}

#[cfg(debug_assertions)]
#[test]
fn released_graph_handle_panics_on_dereference_in_debug() {
    let mut engine = and_engine();
    let root = engine.root();
    let mut candidates = Vec::new();
    engine
        .candidates(root, CandidateOptions::default(), &mut candidates)
        .unwrap();
    let applied = engine.apply(root, candidates[0]).unwrap();

    engine.release(&[applied.after], &[]).unwrap();

    let stale = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = engine.hash(applied.after);
    }));
    assert!(stale.is_err());
}

#[cfg(debug_assertions)]
#[test]
fn released_candidate_handle_panics_on_dereference_in_debug() {
    let mut engine = and_engine();
    let root = engine.root();
    let mut candidates = Vec::new();
    engine
        .candidates(root, CandidateOptions::default(), &mut candidates)
        .unwrap();
    let candidate = candidates[0];

    engine.release(&[], &[candidate]).unwrap();

    let stale = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = engine.candidate_info(root, candidate);
    }));
    assert!(stale.is_err());
}

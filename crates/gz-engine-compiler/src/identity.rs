use blake3::Hasher;
use graphs_compiler::{
    actions::{CompilerActionHash, CompilerActionSetHash, CompilerDomainHash, CompilerStateHash},
    engine::COMPILER_ENGINE_SCHEMA_VERSION,
};
use gz_engine::{ActionSetHash, CandidateHash, EngineId, EngineVersion, GraphHash};

use crate::CompilerCandidateMode;

pub(crate) fn compiler_engine_id() -> EngineId {
    let digest = domain_hash(b"gz-engine-compiler-id-v1", &[]);
    EngineId::from_bytes(
        digest[..EngineId::BYTE_LEN]
            .try_into()
            .expect("fixed-size hash"),
    )
}

pub(crate) fn compiler_engine_version(
    domain: CompilerDomainHash,
    candidate_mode: CompilerCandidateMode,
) -> EngineVersion {
    let schema = COMPILER_ENGINE_SCHEMA_VERSION.to_le_bytes();
    let digest = match candidate_mode {
        CompilerCandidateMode::Sitewise => domain_hash(
            b"gz-engine-compiler-version-v1",
            &[&schema, domain.as_ref()],
        ),
        CompilerCandidateMode::HybridV1
        | CompilerCandidateMode::HybridProviderV1
        | CompilerCandidateMode::HybridNvidiaV1
        | CompilerCandidateMode::HybridNvidiaV2
        | CompilerCandidateMode::EndToEndV1
        | CompilerCandidateMode::EndToEndNvidiaV1
        | CompilerCandidateMode::EndToEndNvidiaV2
        | CompilerCandidateMode::EndToEndNvidiaV3
        | CompilerCandidateMode::EndToEndNvidiaV4
        | CompilerCandidateMode::EndToEndNvidiaV5 => domain_hash(
            b"gz-engine-compiler-version-v2",
            &[&schema, domain.as_ref(), &[candidate_mode.identity_tag()]],
        ),
    };
    EngineVersion::from_bytes(
        digest[..EngineVersion::BYTE_LEN]
            .try_into()
            .expect("fixed-size hash"),
    )
}

fn domain_hash(domain: &[u8], chunks: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(&(domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    for chunk in chunks {
        hasher.update(&(chunk.len() as u64).to_le_bytes());
        hasher.update(chunk);
    }
    *hasher.finalize().as_bytes()
}

pub(crate) fn graph_hash(hash: CompilerStateHash) -> GraphHash {
    GraphHash::from_bytes(*hash.as_ref())
}

pub(crate) fn candidate_hash(hash: CompilerActionHash) -> CandidateHash {
    CandidateHash::from_bytes(*hash.as_ref())
}

pub(crate) fn action_set_hash(
    hash: CompilerActionSetHash,
    candidate_mode: CompilerCandidateMode,
) -> ActionSetHash {
    match candidate_mode {
        CompilerCandidateMode::Sitewise => ActionSetHash::from_bytes(*hash.as_ref()),
        CompilerCandidateMode::HybridV1
        | CompilerCandidateMode::HybridProviderV1
        | CompilerCandidateMode::HybridNvidiaV1
        | CompilerCandidateMode::HybridNvidiaV2
        | CompilerCandidateMode::EndToEndV1
        | CompilerCandidateMode::EndToEndNvidiaV1
        | CompilerCandidateMode::EndToEndNvidiaV2
        | CompilerCandidateMode::EndToEndNvidiaV3
        | CompilerCandidateMode::EndToEndNvidiaV4
        | CompilerCandidateMode::EndToEndNvidiaV5 => ActionSetHash::from_bytes(domain_hash(
            b"gz-engine-compiler-action-set-v2",
            &[hash.as_ref(), &[candidate_mode.identity_tag()]],
        )),
    }
}

pub(crate) fn batch_candidate_hash(
    parent_hash: GraphHash,
    action_set_hash: ActionSetHash,
    member_hashes: &[CandidateHash],
) -> CandidateHash {
    let mut hasher = Hasher::new();
    hasher.update(b"gz-engine-compiler-candidate-batch-v1\0");
    hasher.update(parent_hash.as_bytes());
    hasher.update(action_set_hash.as_bytes());
    hasher.update(&(member_hashes.len() as u64).to_le_bytes());
    for hash in member_hashes {
        hasher.update(hash.as_bytes());
    }
    CandidateHash::from_bytes(*hasher.finalize().as_bytes())
}

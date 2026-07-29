use blake3::Hasher;
use graphs_compiler::{
    actions::{CompilerActionHash, CompilerActionSetHash, CompilerDomainHash, CompilerStateHash},
    engine::COMPILER_ENGINE_SCHEMA_VERSION,
};
use gz_engine::{ActionSetHash, CandidateHash, EngineId, EngineVersion, GraphHash};

pub(crate) fn compiler_engine_id() -> EngineId {
    let digest = domain_hash(b"gz-engine-compiler-id-v1", &[]);
    EngineId::from_bytes(
        digest[..EngineId::BYTE_LEN]
            .try_into()
            .expect("fixed-size hash"),
    )
}

pub(crate) fn compiler_engine_version(domain: CompilerDomainHash) -> EngineVersion {
    let schema = COMPILER_ENGINE_SCHEMA_VERSION.to_le_bytes();
    let digest = domain_hash(
        b"gz-engine-compiler-version-v1",
        &[&schema, domain.as_ref()],
    );
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

pub(crate) fn action_set_hash(hash: CompilerActionSetHash) -> ActionSetHash {
    ActionSetHash::from_bytes(*hash.as_ref())
}

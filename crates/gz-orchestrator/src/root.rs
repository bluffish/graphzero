use gz_engine::{EngineResult, GraphEngine};

pub trait RootSource<E: GraphEngine> {
    fn next_root(&mut self, engine: &mut E) -> EngineResult<Option<E::Graph>>;

    /// The single root every episode shares, for sources that have one
    /// (fixed-root mode). Opponent rollouts replay from it without
    /// consuming the episode budget. None for per-episode roots.
    fn fixed_root(&mut self, engine: &mut E) -> EngineResult<Option<E::Graph>> {
        let _ = engine;
        Ok(None)
    }
}

impl<E, F> RootSource<E> for F
where
    E: GraphEngine,
    F: FnMut(&mut E) -> EngineResult<Option<E::Graph>>,
{
    fn next_root(&mut self, engine: &mut E) -> EngineResult<Option<E::Graph>> {
        self(engine)
    }
}

pub struct CountedRoots<F> {
    remaining: u64,
    factory: F,
}

impl<F> CountedRoots<F> {
    #[must_use]
    pub const fn new(count: u64, factory: F) -> Self {
        Self {
            remaining: count,
            factory,
        }
    }
}

impl<E, F> RootSource<E> for CountedRoots<F>
where
    E: GraphEngine,
    F: FnMut(&mut E) -> EngineResult<E::Graph>,
{
    fn next_root(&mut self, engine: &mut E) -> EngineResult<Option<E::Graph>> {
        if self.remaining == 0 {
            return Ok(None);
        }

        self.remaining -= 1;
        (self.factory)(engine).map(Some)
    }
}

/// Derives an episode's Gumbel noise seed from its id (which encodes lane
/// and per-lane order), so episodes sharing a root explore differently
/// while staying deterministic across drivers.
#[must_use]
pub fn episode_noise_seed(episode_id: u64) -> u64 {
    // splitmix64 finalizer.
    let mut z = episode_id.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

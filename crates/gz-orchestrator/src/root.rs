use gz_engine::{EngineResult, GraphEngine};

pub trait RootSource<E: GraphEngine> {
    fn next_root(&mut self, engine: &mut E) -> EngineResult<Option<E::Graph>>;
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

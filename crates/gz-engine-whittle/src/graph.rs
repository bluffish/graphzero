use gz_engine::GraphHash;
use std::fmt;

pub const NO_NODE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum OpCode {
    Input = 0,
    Const = 1,
    And = 2,
    Or = 3,
    Not = 4,
    Output = 5,
}

impl OpCode {
    pub(crate) fn from_i8(value: i8) -> Result<Self, GraphError> {
        match value {
            0 => Ok(Self::Input),
            1 => Ok(Self::Const),
            2 => Ok(Self::And),
            3 => Ok(Self::Or),
            4 => Ok(Self::Not),
            5 => Ok(Self::Output),
            _ => Err(GraphError::BadOp(value)),
        }
    }

    pub(crate) const fn as_i8(self) -> i8 {
        self as i8
    }
}

#[cfg(debug_assertions)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct WhittleGraphId {
    index: u32,
    generation: u32,
}

#[cfg(not(debug_assertions))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct WhittleGraphId(u32);

impl WhittleGraphId {
    #[must_use]
    pub const fn from_raw(value: u32) -> Self {
        #[cfg(debug_assertions)]
        {
            Self {
                index: value,
                generation: 0,
            }
        }
        #[cfg(not(debug_assertions))]
        {
            Self(value)
        }
    }

    #[must_use]
    pub(crate) const fn from_slot(index: u32, generation: u32) -> Self {
        #[cfg(debug_assertions)]
        {
            Self { index, generation }
        }
        #[cfg(not(debug_assertions))]
        {
            let _ = generation;
            Self(index)
        }
    }

    #[must_use]
    pub(crate) const fn generation(self) -> u32 {
        #[cfg(debug_assertions)]
        {
            self.generation
        }
        #[cfg(not(debug_assertions))]
        {
            0
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        #[cfg(debug_assertions)]
        {
            self.index
        }
        #[cfg(not(debug_assertions))]
        {
            self.0
        }
    }
}

#[cfg(debug_assertions)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct WhittleCandidateId {
    index: u32,
    generation: u32,
}

#[cfg(not(debug_assertions))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct WhittleCandidateId(u32);

impl WhittleCandidateId {
    #[must_use]
    pub const fn from_raw(value: u32) -> Self {
        #[cfg(debug_assertions)]
        {
            Self {
                index: value,
                generation: 0,
            }
        }
        #[cfg(not(debug_assertions))]
        {
            Self(value)
        }
    }

    #[must_use]
    pub(crate) const fn from_slot(index: u32, generation: u32) -> Self {
        #[cfg(debug_assertions)]
        {
            Self { index, generation }
        }
        #[cfg(not(debug_assertions))]
        {
            let _ = generation;
            Self(index)
        }
    }

    #[must_use]
    pub(crate) const fn generation(self) -> u32 {
        #[cfg(debug_assertions)]
        {
            self.generation
        }
        #[cfg(not(debug_assertions))]
        {
            0
        }
    }

    #[must_use]
    pub const fn raw(self) -> u32 {
        #[cfg(debug_assertions)]
        {
            self.index
        }
        #[cfg(not(debug_assertions))]
        {
            self.0
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WhittleGraph {
    pub arity: u16,
    pub capacity: u16,
    pub output_node: u32,
    pub op: Box<[OpCode]>,
    pub arg0: Box<[u32]>,
    pub arg1: Box<[u32]>,
    pub canonical: Box<[u8]>,
    pub hash: GraphHash,
}

impl WhittleGraph {
    #[must_use]
    pub fn cost(&self) -> u32 {
        self.op.len() as u32
    }

    pub(crate) fn body(&self) -> GraphBody {
        GraphBody {
            arity: self.arity,
            capacity: self.capacity,
            output_node: self.output_node,
            op: self.op.to_vec(),
            arg0: self.arg0.to_vec(),
            arg1: self.arg1.to_vec(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GraphBody {
    pub arity: u16,
    pub capacity: u16,
    pub output_node: u32,
    pub op: Vec<OpCode>,
    pub arg0: Vec<u32>,
    pub arg1: Vec<u32>,
}

impl GraphBody {
    pub(crate) fn input(arity: u16, capacity: u16, input_index: u16) -> Result<Self, GraphError> {
        if arity == 0 {
            return Err(GraphError::InvalidInput("arity must be greater than zero"));
        }
        if input_index >= arity {
            return Err(GraphError::InvalidInput("input_index out of range"));
        }
        if capacity < arity + 1 {
            return Err(GraphError::InvalidInput("capacity too small"));
        }

        let mut op = vec![OpCode::Input; arity as usize];
        let mut arg0: Vec<_> = (0..u32::from(arity)).collect();
        let mut arg1 = vec![NO_NODE; arity as usize];
        op.push(OpCode::Output);
        arg0.push(u32::from(input_index));
        arg1.push(NO_NODE);

        Ok(Self {
            arity,
            capacity,
            output_node: arity.into(),
            op,
            arg0,
            arg1,
        })
    }

    pub(crate) fn new(
        arity: u16,
        capacity: u16,
        output_node: u32,
        op: Vec<OpCode>,
        arg0: Vec<u32>,
        arg1: Vec<u32>,
    ) -> Result<Self, GraphError> {
        let body = Self {
            arity,
            capacity,
            output_node,
            op,
            arg0,
            arg1,
        };
        body.validate()?;
        Ok(body)
    }

    pub(crate) fn validate(&self) -> Result<(), GraphError> {
        let n = self.op.len();

        if n == 0 {
            return Err(GraphError::InvalidInput("graph must contain nodes"));
        }
        if self.arg0.len() != n || self.arg1.len() != n {
            return Err(GraphError::InvalidInput("graph arrays differ in length"));
        }
        if usize::from(self.capacity) < n {
            return Err(GraphError::InvalidInput("capacity below node count"));
        }
        if self.output_node as usize >= n {
            return Err(GraphError::InvalidInput("output_node out of range"));
        }
        if usize::from(self.arity) > n {
            return Err(GraphError::InvalidInput("arity exceeds node count"));
        }

        for i in 0..usize::from(self.arity) {
            if self.op[i] != OpCode::Input || self.arg0[i] != i as u32 {
                return Err(GraphError::InvalidInput(
                    "input nodes must occupy ids 0..arity-1",
                ));
            }
        }

        Ok(())
    }
}

pub(crate) fn compact_graph(graph: &GraphBody) -> Result<GraphBody, GraphError> {
    graph.validate()?;

    let mut reachable = vec![false; graph.op.len()];
    let mut stack = vec![graph.output_node];

    while let Some(node) = stack.pop() {
        let index = node as usize;
        if index >= graph.op.len() {
            return Err(GraphError::InvalidInput("bad node reference"));
        }
        if reachable[index] {
            continue;
        }
        reachable[index] = true;
        push_children(graph, node, &mut stack)?;
    }

    let mut keep = Vec::with_capacity(graph.op.len());
    let mut identity = true;

    for (i, is_reachable) in reachable.iter().copied().enumerate() {
        if i < usize::from(graph.arity) || is_reachable {
            if i != keep.len() {
                identity = false;
            }
            keep.push(i as u32);
        }
    }

    if identity && keep.len() == graph.op.len() {
        return Ok(graph.clone());
    }

    let mut remap = vec![NO_NODE; graph.op.len()];
    for (new, old) in keep.iter().copied().enumerate() {
        remap[old as usize] = new as u32;
    }

    let mut op = Vec::with_capacity(keep.len());
    let mut arg0 = Vec::with_capacity(keep.len());
    let mut arg1 = Vec::with_capacity(keep.len());

    for old in keep {
        let index = old as usize;
        let code = graph.op[index];
        op.push(code);

        match code {
            OpCode::Input => {
                arg0.push(graph.arg0[index]);
                arg1.push(NO_NODE);
            }
            OpCode::Const => {
                arg0.push(u32::from(graph.arg0[index] != 0));
                arg1.push(NO_NODE);
            }
            OpCode::Not | OpCode::Output => {
                arg0.push(remapped(&remap, graph.arg0[index])?);
                arg1.push(NO_NODE);
            }
            OpCode::And | OpCode::Or => {
                arg0.push(remapped(&remap, graph.arg0[index])?);
                arg1.push(remapped(&remap, graph.arg1[index])?);
            }
        }
    }

    GraphBody::new(
        graph.arity,
        graph.capacity,
        remapped(&remap, graph.output_node)?,
        op,
        arg0,
        arg1,
    )
}

fn remapped(remap: &[u32], node: u32) -> Result<u32, GraphError> {
    let mapped = remap
        .get(node as usize)
        .copied()
        .ok_or(GraphError::InvalidInput("bad node reference"))?;

    if mapped == NO_NODE {
        Err(GraphError::InvalidInput("unreachable child reference"))
    } else {
        Ok(mapped)
    }
}

pub(crate) fn children(
    graph: &GraphBody,
    node: u32,
) -> Result<impl Iterator<Item = u32>, GraphError> {
    let index = node as usize;
    let code = *graph
        .op
        .get(index)
        .ok_or(GraphError::InvalidInput("node out of range"))?;

    let mut out = [NO_NODE; 2];
    let len = match code {
        OpCode::Input | OpCode::Const => 0,
        OpCode::Not | OpCode::Output => {
            out[0] = graph.arg0[index];
            1
        }
        OpCode::And | OpCode::Or => {
            out[0] = graph.arg0[index];
            out[1] = graph.arg1[index];
            2
        }
    };

    Ok(out.into_iter().take(len))
}

fn push_children(graph: &GraphBody, node: u32, out: &mut Vec<u32>) -> Result<(), GraphError> {
    out.extend(children(graph, node)?);
    Ok(())
}

pub(crate) fn serialize_wav1(graph: &GraphBody) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + graph.op.len() * 9);
    out.extend_from_slice(b"WAV1");
    out.extend_from_slice(&graph.arity.to_le_bytes());
    out.extend_from_slice(&graph.capacity.to_le_bytes());
    out.extend_from_slice(&(graph.op.len() as u32).to_le_bytes());
    out.extend_from_slice(&graph.output_node.to_le_bytes());

    for i in 0..graph.op.len() {
        out.push(graph.op[i].as_i8() as u8);
        out.extend_from_slice(&graph.arg0[i].to_le_bytes());
        out.extend_from_slice(&graph.arg1[i].to_le_bytes());
    }

    out
}

pub(crate) fn deserialize_wav1(bytes: &[u8]) -> Result<GraphBody, GraphError> {
    if bytes.len() < 16 || &bytes[..4] != b"WAV1" {
        return Err(GraphError::InvalidInput("bad WAV1 graph artifact"));
    }

    let arity = u16::from_le_bytes([bytes[4], bytes[5]]);
    let capacity = u16::from_le_bytes([bytes[6], bytes[7]]);
    let node_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let output_node = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    let expected_len = 16 + node_count * 9;

    if bytes.len() != expected_len {
        return Err(GraphError::InvalidInput("wrong WAV1 graph artifact length"));
    }

    let mut op = Vec::with_capacity(node_count);
    let mut arg0 = Vec::with_capacity(node_count);
    let mut arg1 = Vec::with_capacity(node_count);
    let mut offset = 16;

    for _ in 0..node_count {
        op.push(OpCode::from_i8(bytes[offset] as i8)?);
        offset += 1;
        arg0.push(u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]));
        offset += 4;
        arg1.push(u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]));
        offset += 4;
    }

    GraphBody::new(arity, capacity, output_node, op, arg0, arg1)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GraphError {
    InvalidInput(&'static str),
    BadOp(i8),
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => f.write_str(message),
            Self::BadOp(op) => write!(f, "bad op {op}"),
        }
    }
}

impl std::error::Error for GraphError {}

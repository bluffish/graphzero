use crate::wire::{bf16_bits_to_f32, f32_to_bf16_bits};
use crate::{
    ActionFeature, ENCODING_VERSION, FeatureEdge, FeatureError, FeatureResult, FeatureRow,
    FeatureSchema, FeatureSchemaConfig, FeatureSchemaHash, OpponentStateFeatures, PositionFeatures,
};

const ROW_MAGIC: &[u8; 4] = b"GZFR";
const TARGET_MAGIC: &[u8; 4] = b"GZFT";
const ROW_HEADER_LEN: usize = 40;
const TARGET_HEADER_LEN: usize = 20;

#[derive(Clone, Debug, PartialEq)]
pub struct RowTargets {
    pub policy: Vec<f32>,
    pub value: Option<f32>,
    pub horizon_value: Option<[f32; 2]>,
    pub reward: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingTargetsView {
    pub capacity: u32,
    pub row_count: u32,
    pub max_actions: u32,
    pub policy: Vec<f32>,
    pub value: Vec<f32>,
    pub value_valid: Vec<u8>,
    pub horizon_value: Vec<f32>,
    pub horizon_value_valid: Vec<u8>,
    pub reward: Vec<f32>,
}

impl TrainingTargetsView {
    pub fn parse(bytes: &[u8]) -> FeatureResult<Self> {
        let header = TargetHeader::parse(bytes)?;
        let layout = TargetLayout::new(header.capacity as usize, header.max_actions as usize)?;
        if bytes.len() != layout.total_len {
            return Err(FeatureError::InvalidEncoding("bad target length"));
        }

        Ok(Self {
            capacity: header.capacity,
            row_count: header.row_count,
            max_actions: header.max_actions,
            policy: read_f32_vec(bytes, layout.policy, layout.b * layout.a)?,
            value: read_f32_vec(bytes, layout.value, layout.b)?,
            value_valid: bytes[layout.value_valid..layout.value_valid + layout.b].to_vec(),
            horizon_value: read_f32_vec(bytes, layout.horizon_value, layout.b * 2)?,
            horizon_value_valid: bytes
                [layout.horizon_value_valid..layout.horizon_value_valid + layout.b]
                .to_vec(),
            reward: read_f32_vec(bytes, layout.reward, layout.b)?,
        })
    }
}

pub fn encode_feature_row(
    row: &FeatureRow,
    schema: &FeatureSchema,
    out: &mut Vec<u8>,
) -> FeatureResult<()> {
    row.validate(schema)?;
    out.clear();
    out.extend_from_slice(ROW_MAGIC);
    write_u32(out, ENCODING_VERSION);
    out.extend_from_slice(schema.hash().as_bytes());
    write_u32(out, row.node_count);

    write_state_body(
        row.node_count,
        &row.node_tokens,
        &row.node_attrs,
        &row.edges,
        out,
    )?;

    write_u32(out, row.actions.len() as u32);
    for action in &row.actions {
        write_u16(out, narrow_index(action.kind_token, "action kind token")?);
        write_bf16(out, action.static_prior);
        let subject_count = u8::try_from(action.subjects.len())
            .map_err(|_| FeatureError::InvalidEncoding("subject count exceeds u8"))?;
        out.push(subject_count);
        for subject in &action.subjects {
            write_u32(out, *subject);
        }
    }

    write_position(out, row.position);
    match &row.opponent {
        Some(opponent) => {
            out.push(1);
            write_u32(out, opponent.node_count);
            write_state_body(
                opponent.node_count,
                &opponent.node_tokens,
                &opponent.node_attrs,
                &opponent.edges,
                out,
            )?;
            write_position(out, opponent.position);
        }
        None => out.push(0),
    }
    Ok(())
}

pub fn decode_feature_row(bytes: &[u8]) -> FeatureResult<FeatureRow> {
    parse_row_header(bytes)?;
    let mut reader = Reader::new(&bytes[ROW_HEADER_LEN..]);

    let node_count = reader.u32()?;
    let (node_tokens, node_attrs, edges) = read_state_body(&mut reader, node_count)?;

    let action_count = reader.len()?;
    let mut actions = Vec::with_capacity(action_count);
    for _ in 0..action_count {
        let kind_token = u32::from(reader.u16()?);
        let static_prior = reader.bf16()?;
        if !static_prior.is_finite() {
            return Err(FeatureError::InvalidEncoding("non-finite action prior"));
        }
        let subject_count = usize::from(reader.u8()?);
        let mut subjects = Vec::with_capacity(subject_count);
        for _ in 0..subject_count {
            let subject = reader.u32()?;
            if subject >= node_count {
                return Err(FeatureError::InvalidEncoding("action subject out of range"));
            }
            subjects.push(subject);
        }
        actions.push(ActionFeature {
            kind_token,
            static_prior,
            subjects,
        });
    }

    let position = read_position(&mut reader)?;
    let opponent = if reader.u8()? != 0 {
        let node_count = reader.u32()?;
        let (node_tokens, node_attrs, edges) = read_state_body(&mut reader, node_count)?;
        let position = read_position(&mut reader)?;
        Some(OpponentStateFeatures {
            node_count,
            node_tokens,
            node_attrs,
            edges,
            position,
        })
    } else {
        None
    };
    if !reader.is_empty() {
        return Err(FeatureError::InvalidEncoding("trailing row bytes"));
    }

    Ok(FeatureRow {
        node_count,
        node_tokens,
        node_attrs,
        edges,
        actions,
        position,
        opponent,
    })
}

pub fn validate_feature_row_header(
    bytes: &[u8],
    expected: &FeatureSchemaHash,
) -> FeatureResult<()> {
    let schema_hash = parse_row_header(bytes)?;
    if &schema_hash != expected {
        return Err(FeatureError::InvalidEncoding(
            "feature schema hash mismatch",
        ));
    }
    Ok(())
}

fn write_state_body(
    node_count: u32,
    node_tokens: &[u16],
    node_attrs: &[f32],
    edges: &[FeatureEdge],
    out: &mut Vec<u8>,
) -> FeatureResult<()> {
    if node_tokens.len() != node_count as usize {
        return Err(FeatureError::InvalidEncoding("node token length mismatch"));
    }
    write_u32(out, node_tokens.len() as u32);
    for token in node_tokens {
        write_u16(out, *token);
    }

    write_u32(out, node_attrs.len() as u32);
    for value in node_attrs {
        write_bf16(out, *value);
    }

    write_u32(out, edges.len() as u32);
    for edge in edges {
        write_u32(out, edge.src);
        write_u32(out, edge.dst);
        out.push(edge.edge_type);
    }
    Ok(())
}

fn read_state_body(
    reader: &mut Reader<'_>,
    node_count: u32,
) -> FeatureResult<(Vec<u16>, Vec<f32>, Vec<FeatureEdge>)> {
    if node_count == 0 {
        return Err(FeatureError::InvalidEncoding("zero node count"));
    }

    let node_tokens = reader.u16_vec()?;
    if node_tokens.len() != node_count as usize {
        return Err(FeatureError::InvalidEncoding("node token length mismatch"));
    }

    let node_attrs = reader.bf16_vec()?;
    if node_attrs.iter().any(|value| !value.is_finite()) {
        return Err(FeatureError::InvalidEncoding("non-finite node attr"));
    }

    let edge_count = reader.len()?;
    let mut edges = Vec::with_capacity(edge_count);
    for _ in 0..edge_count {
        let edge = FeatureEdge {
            src: reader.u32()?,
            dst: reader.u32()?,
            edge_type: reader.u8()?,
        };
        if edge.src >= node_count || edge.dst >= node_count {
            return Err(FeatureError::InvalidEncoding("edge endpoint out of range"));
        }
        edges.push(edge);
    }

    Ok((node_tokens, node_attrs, edges))
}

fn write_position(out: &mut Vec<u8>, position: PositionFeatures) {
    write_u32(out, position.root_step);
    write_u32(out, position.leaf_depth);
    write_bf16(out, position.budget_fraction);
    write_bf16(out, position.budget_step);
    write_bf16(out, position.opponent_reward);
    out.push(u8::from(position.opponent_present));
}

fn read_position(reader: &mut Reader<'_>) -> FeatureResult<PositionFeatures> {
    let position = PositionFeatures {
        root_step: reader.u32()?,
        leaf_depth: reader.u32()?,
        budget_fraction: reader.bf16()?,
        budget_step: reader.bf16()?,
        opponent_reward: reader.bf16()?,
        opponent_present: reader.u8()? != 0,
    };
    if !position.budget_fraction.is_finite()
        || !position.budget_step.is_finite()
        || !position.opponent_reward.is_finite()
    {
        return Err(FeatureError::InvalidEncoding("non-finite position feature"));
    }
    if !position.opponent_present && position.opponent_reward != 0.0 {
        return Err(FeatureError::InvalidEncoding(
            "absent opponent must have zero reward",
        ));
    }
    Ok(position)
}

pub fn encode_feature_schema_config(
    config: &FeatureSchemaConfig,
    out: &mut Vec<u8>,
) -> FeatureResult<()> {
    FeatureSchema::new(config.clone())?;
    let name = config.name.as_bytes();
    let name_len =
        u16::try_from(name.len()).map_err(|_| FeatureError::InvalidSchema("name too long"))?;

    out.clear();
    write_u16(out, name_len);
    out.extend_from_slice(name);
    write_u16(out, config.node_vocab_size);
    write_u16(out, config.node_attr_dim);
    out.push(config.edge_type_count);
    write_u32(out, config.action_kind_vocab_size);
    write_u32(out, config.max_nodes);
    write_u32(out, config.max_edges);
    write_u32(out, config.max_actions);
    write_u32(out, config.max_subjects);
    write_f32(out, config.opponent_reward_scale);
    out.push(config.expander_degree);
    write_u64(out, config.expander_seed);
    Ok(())
}

pub fn decode_feature_schema_config(bytes: &[u8]) -> FeatureResult<FeatureSchemaConfig> {
    let mut reader = Reader::new(bytes);
    let name_len = reader.u16()? as usize;
    let name = reader.bytes(name_len)?;
    let name = std::str::from_utf8(name)
        .map_err(|_| FeatureError::InvalidEncoding("invalid schema name utf8"))?
        .to_owned();
    let config = FeatureSchemaConfig {
        name,
        node_vocab_size: reader.u16()?,
        node_attr_dim: reader.u16()?,
        edge_type_count: reader.u8()?,
        action_kind_vocab_size: reader.u32()?,
        max_nodes: reader.u32()?,
        max_edges: reader.u32()?,
        max_actions: reader.u32()?,
        max_subjects: reader.u32()?,
        opponent_reward_scale: reader.f32()?,
        expander_degree: reader.u8()?,
        expander_seed: reader.u64()?,
    };
    if !reader.is_empty() {
        return Err(FeatureError::InvalidEncoding(
            "trailing schema config bytes",
        ));
    }
    FeatureSchema::new(config.clone())?;
    Ok(config)
}

pub fn encode_training_targets(
    targets: &[RowTargets],
    capacity: usize,
    max_actions: usize,
    out: &mut Vec<u8>,
) -> FeatureResult<()> {
    if targets.is_empty() {
        return Err(FeatureError::EmptyBatch);
    }
    if capacity == 0 || max_actions == 0 {
        return Err(FeatureError::InvalidEncoding("zero target dimension"));
    }
    if targets.len() > capacity {
        return Err(FeatureError::BatchOverflow {
            capacity,
            actual: targets.len(),
        });
    }
    for target in targets {
        if target.policy.len() > max_actions {
            return Err(FeatureError::ActionOverflow {
                max: max_actions as u32,
                actual: target.policy.len(),
            });
        }
        if target.policy.iter().any(|value| !value.is_finite()) {
            return Err(FeatureError::InvalidEncoding("non-finite policy target"));
        }
        if let Some(value) = target.value
            && !value.is_finite()
        {
            return Err(FeatureError::InvalidEncoding("invalid value target"));
        }
        if let Some(values) = target.horizon_value
            && values
                .iter()
                .any(|value| !value.is_finite() || !(-1.0..=1.0).contains(value))
        {
            return Err(FeatureError::InvalidEncoding(
                "invalid horizon value target",
            ));
        }
        if !target.reward.is_finite() {
            return Err(FeatureError::InvalidEncoding("non-finite reward target"));
        }
    }

    let layout = TargetLayout::new(capacity, max_actions)?;
    out.clear();
    out.resize(layout.total_len, 0);
    out[0..4].copy_from_slice(TARGET_MAGIC);
    write_u32_at(out, 4, ENCODING_VERSION);
    write_u32_at(out, 8, capacity as u32);
    write_u32_at(out, 12, targets.len() as u32);
    write_u32_at(out, 16, max_actions as u32);

    for (row_index, target) in targets.iter().enumerate() {
        for (action_index, value) in target.policy.iter().copied().enumerate() {
            write_f32_at(
                out,
                layout.policy + (row_index * max_actions + action_index) * 4,
                value,
            );
        }
        if let Some(value) = target.value {
            write_f32_at(out, layout.value + row_index * 4, value);
            out[layout.value_valid + row_index] = 1;
        }
        if let Some(values) = target.horizon_value {
            for (head, value) in values.into_iter().enumerate() {
                write_f32_at(
                    out,
                    layout.horizon_value + (row_index * 2 + head) * 4,
                    value,
                );
            }
            out[layout.horizon_value_valid + row_index] = 1;
        }
        write_f32_at(out, layout.reward + row_index * 4, target.reward);
    }

    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct TargetHeader {
    capacity: u32,
    row_count: u32,
    max_actions: u32,
}

impl TargetHeader {
    fn parse(bytes: &[u8]) -> FeatureResult<Self> {
        if bytes.len() < TARGET_HEADER_LEN {
            return Err(FeatureError::InvalidEncoding("target header truncated"));
        }
        if &bytes[0..4] != TARGET_MAGIC {
            return Err(FeatureError::InvalidEncoding("bad target magic"));
        }
        let version = read_u32_at(bytes, 4)?;
        if version != ENCODING_VERSION {
            return Err(FeatureError::InvalidEncoding("unsupported target version"));
        }
        let header = Self {
            capacity: read_u32_at(bytes, 8)?,
            row_count: read_u32_at(bytes, 12)?,
            max_actions: read_u32_at(bytes, 16)?,
        };
        if header.capacity == 0 {
            return Err(FeatureError::InvalidEncoding("zero target capacity"));
        }
        if header.max_actions == 0 {
            return Err(FeatureError::InvalidEncoding("zero target actions"));
        }
        if header.row_count > header.capacity {
            return Err(FeatureError::InvalidEncoding(
                "target row count exceeds capacity",
            ));
        }
        Ok(header)
    }
}

#[derive(Clone, Copy, Debug)]
struct TargetLayout {
    b: usize,
    a: usize,
    policy: usize,
    value: usize,
    value_valid: usize,
    horizon_value: usize,
    horizon_value_valid: usize,
    reward: usize,
    total_len: usize,
}

impl TargetLayout {
    fn new(capacity: usize, max_actions: usize) -> FeatureResult<Self> {
        let policy_len = capacity
            .checked_mul(max_actions)
            .and_then(|count| count.checked_mul(4))
            .ok_or(FeatureError::InvalidEncoding("target length overflow"))?;
        let value_len = capacity
            .checked_mul(4)
            .ok_or(FeatureError::InvalidEncoding("target length overflow"))?;
        let horizon_value_len = capacity
            .checked_mul(2)
            .and_then(|count| count.checked_mul(4))
            .ok_or(FeatureError::InvalidEncoding("target length overflow"))?;
        let mut cursor = TARGET_HEADER_LEN;
        let policy = section(&mut cursor, policy_len);
        let value = section(&mut cursor, value_len);
        let value_valid = section(&mut cursor, capacity);
        let horizon_value = section(&mut cursor, horizon_value_len);
        let horizon_value_valid = section(&mut cursor, capacity);
        let reward = section(&mut cursor, value_len);
        let total_len = align4(cursor);
        Ok(Self {
            b: capacity,
            a: max_actions,
            policy,
            value,
            value_valid,
            horizon_value,
            horizon_value_valid,
            reward,
            total_len,
        })
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    const fn is_empty(&self) -> bool {
        self.cursor == self.bytes.len()
    }

    fn u8(&mut self) -> FeatureResult<u8> {
        let byte = *self
            .bytes
            .get(self.cursor)
            .ok_or(FeatureError::InvalidEncoding("u8 truncated"))?;
        self.cursor += 1;
        Ok(byte)
    }

    fn u16(&mut self) -> FeatureResult<u16> {
        let bytes = self.take(2, "u16 truncated")?;
        Ok(u16::from_le_bytes(
            bytes.try_into().expect("length checked"),
        ))
    }

    fn u32(&mut self) -> FeatureResult<u32> {
        let bytes = self.take(4, "u32 truncated")?;
        Ok(u32::from_le_bytes(
            bytes.try_into().expect("length checked"),
        ))
    }

    fn u64(&mut self) -> FeatureResult<u64> {
        let bytes = self.take(8, "u64 truncated")?;
        Ok(u64::from_le_bytes(
            bytes.try_into().expect("length checked"),
        ))
    }

    fn f32(&mut self) -> FeatureResult<f32> {
        let bytes = self.take(4, "f32 truncated")?;
        Ok(f32::from_le_bytes(
            bytes.try_into().expect("length checked"),
        ))
    }

    fn bf16(&mut self) -> FeatureResult<f32> {
        let bytes = self.take(2, "bf16 truncated")?;
        Ok(bf16_bits_to_f32(u16::from_le_bytes(
            bytes.try_into().expect("length checked"),
        )))
    }

    fn bf16_vec(&mut self) -> FeatureResult<Vec<f32>> {
        let count = self.len()?;
        self.ensure_count(count, 2)?;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(self.bf16()?);
        }
        Ok(out)
    }

    fn len(&mut self) -> FeatureResult<usize> {
        usize::try_from(self.u32()?).map_err(|_| FeatureError::InvalidEncoding("length overflow"))
    }

    fn u16_vec(&mut self) -> FeatureResult<Vec<u16>> {
        let count = self.len()?;
        self.ensure_count(count, 2)?;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(self.u16()?);
        }
        Ok(out)
    }

    fn ensure_count(&self, count: usize, width: usize) -> FeatureResult<()> {
        let len = count
            .checked_mul(width)
            .ok_or(FeatureError::InvalidEncoding("length overflow"))?;
        self.bytes
            .get(self.cursor..self.cursor + len)
            .ok_or(FeatureError::InvalidEncoding("section truncated"))?;
        Ok(())
    }

    fn take(&mut self, len: usize, message: &'static str) -> FeatureResult<&'a [u8]> {
        let bytes = self
            .bytes
            .get(self.cursor..self.cursor + len)
            .ok_or(FeatureError::InvalidEncoding(message))?;
        self.cursor += len;
        Ok(bytes)
    }

    fn bytes(&mut self, len: usize) -> FeatureResult<&'a [u8]> {
        self.take(len, "bytes truncated")
    }
}

fn parse_row_header(bytes: &[u8]) -> FeatureResult<FeatureSchemaHash> {
    if bytes.len() < ROW_HEADER_LEN {
        return Err(FeatureError::InvalidEncoding("row header truncated"));
    }
    if &bytes[0..4] != ROW_MAGIC {
        return Err(FeatureError::InvalidEncoding("bad row magic"));
    }
    let version = read_u32_at(bytes, 4)?;
    if version != ENCODING_VERSION {
        return Err(FeatureError::InvalidEncoding("unsupported row version"));
    }
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&bytes[8..40]);
    Ok(FeatureSchemaHash::from_bytes(hash))
}

fn write_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_f32(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_bf16(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&f32_to_bf16_bits(value).to_le_bytes());
}

fn narrow_index(value: u32, what: &'static str) -> FeatureResult<u16> {
    u16::try_from(value).map_err(|_| FeatureError::InvalidEncoding(what))
}

fn section(cursor: &mut usize, len: usize) -> usize {
    *cursor = align4(*cursor);
    let offset = *cursor;
    *cursor += len;
    offset
}

const fn align4(value: usize) -> usize {
    (value + 3) & !3
}

fn write_u32_at(out: &mut [u8], offset: usize, value: u32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_f32_at(out: &mut [u8], offset: usize, value: f32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn read_u32_at(bytes: &[u8], offset: usize) -> FeatureResult<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or(FeatureError::InvalidEncoding("u32 truncated"))?;
    Ok(u32::from_le_bytes(
        slice.try_into().expect("length checked"),
    ))
}

fn read_f32_at(bytes: &[u8], offset: usize) -> FeatureResult<f32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or(FeatureError::InvalidEncoding("f32 truncated"))?;
    Ok(f32::from_le_bytes(
        slice.try_into().expect("length checked"),
    ))
}

fn read_f32_vec(bytes: &[u8], offset: usize, count: usize) -> FeatureResult<Vec<f32>> {
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        out.push(read_f32_at(bytes, offset + index * 4)?);
    }
    Ok(out)
}

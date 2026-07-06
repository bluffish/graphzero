use crate::wire::{bf16_bits_to_f32, f32_to_bf16_bits};
use crate::{
    BATCH_ENCODING_VERSION, FeatureError, FeatureResult, FeatureRow, FeatureSchema,
    FeatureSchemaHash,
};
use std::num::NonZeroUsize;

const BATCH_MAGIC: &[u8; 4] = b"GZFB";
const OUTPUT_MAGIC: &[u8; 4] = b"GZFO";
const BATCH_HEADER_LEN: usize = 68;
const OUTPUT_HEADER_LEN: usize = 16;

#[derive(Clone, Debug)]
pub struct FeatureCollator {
    schema: FeatureSchema,
    batch_capacity: NonZeroUsize,
}

impl FeatureCollator {
    #[must_use]
    pub const fn new(schema: FeatureSchema, batch_capacity: NonZeroUsize) -> Self {
        Self {
            schema,
            batch_capacity,
        }
    }

    #[must_use]
    pub const fn schema(&self) -> &FeatureSchema {
        &self.schema
    }

    #[must_use]
    pub const fn batch_capacity(&self) -> NonZeroUsize {
        self.batch_capacity
    }

    pub fn collate_into(&mut self, rows: &[FeatureRow], out: &mut Vec<u8>) -> FeatureResult<()> {
        if rows.is_empty() {
            return Err(FeatureError::EmptyBatch);
        }
        let capacity = self.batch_capacity.get();
        if rows.len() > capacity {
            return Err(FeatureError::BatchOverflow {
                capacity,
                actual: rows.len(),
            });
        }
        for row in rows {
            row.validate(&self.schema)?;
        }
        // v2 packs node indexes and kind tokens as u16 on the wire; the
        // subject padding sentinel is 0xffff, so valid indexes must stay
        // strictly below it.
        if self.schema.config().max_nodes > u32::from(u16::MAX) {
            return Err(FeatureError::InvalidEncoding(
                "max_nodes exceeds u16 wire width",
            ));
        }

        let layout = BatchLayout::new(&self.schema, capacity);
        out.clear();
        out.resize(layout.total_len, 0);
        fill_subject_padding(out, &layout);
        write_batch_header(out, &self.schema, capacity, rows.len());

        for (row_index, row) in rows.iter().enumerate() {
            write_u32_at(out, layout.node_count + row_index * 4, row.node_count);
            for (node_index, token) in row.node_tokens.iter().copied().enumerate() {
                let offset = layout.node_tokens + (row_index * layout.n + node_index) * 2;
                write_u16_at(out, offset, token);
            }
            for (attr_index, value) in row.node_attrs.iter().copied().enumerate() {
                let offset = layout.node_attrs + (row_index * layout.n * layout.d + attr_index) * 2;
                write_bf16_at(out, offset, value);
            }

            write_u32_at(
                out,
                layout.edge_count + row_index * 4,
                row.edges.len() as u32,
            );
            for (edge_index, edge) in row.edges.iter().copied().enumerate() {
                write_u16_at(
                    out,
                    layout.edge_src + (row_index * layout.e + edge_index) * 2,
                    narrow_index(edge.src, "edge src")?,
                );
                write_u16_at(
                    out,
                    layout.edge_dst + (row_index * layout.e + edge_index) * 2,
                    narrow_index(edge.dst, "edge dst")?,
                );
                out[layout.edge_type + row_index * layout.e + edge_index] = edge.edge_type;
            }

            write_u32_at(
                out,
                layout.action_count + row_index * 4,
                row.actions.len() as u32,
            );
            for (action_index, action) in row.actions.iter().enumerate() {
                write_u16_at(
                    out,
                    layout.action_kind + (row_index * layout.a + action_index) * 2,
                    narrow_index(action.kind_token, "action kind token")?,
                );
                write_bf16_at(
                    out,
                    layout.action_prior + (row_index * layout.a + action_index) * 2,
                    action.static_prior,
                );
                out[layout.subject_count + row_index * layout.a + action_index] =
                    action.subjects.len() as u8;
                for (subject_index, subject) in action.subjects.iter().copied().enumerate() {
                    write_u16_at(
                        out,
                        layout.action_subjects
                            + ((row_index * layout.a + action_index) * layout.s + subject_index)
                                * 2,
                        narrow_index(subject, "action subject")?,
                    );
                }
            }

            let position = layout.position + row_index * 8;
            write_bf16_at(out, position, row.position.root_step as f32);
            write_bf16_at(out, position + 2, row.position.leaf_depth as f32);
            write_bf16_at(out, position + 4, row.position.budget_fraction);
            write_bf16_at(out, position + 6, row.position.budget_step);
            write_bf16_at(
                out,
                layout.opponent_reward + row_index * 2,
                row.position.opponent_reward,
            );
            out[layout.opponent_present + row_index] = u8::from(row.position.opponent_present);
        }

        Ok(())
    }

    pub fn decode_outputs(
        &self,
        bytes: &[u8],
        action_counts: &[u32],
    ) -> FeatureResult<Vec<RowOutput>> {
        let max_actions = output_max_actions(bytes)?;
        if max_actions != self.schema.config().max_actions as usize {
            return Err(FeatureError::InvalidEncoding(
                "output action width mismatch",
            ));
        }
        let capacity = output_capacity(bytes, action_counts.len(), max_actions)?;
        if capacity != self.batch_capacity.get() {
            return Err(FeatureError::InvalidEncoding("bad output length"));
        }
        decode_outputs(bytes, action_counts)
    }
}

pub fn decode_outputs(bytes: &[u8], action_counts: &[u32]) -> FeatureResult<Vec<RowOutput>> {
    if bytes.len() < OUTPUT_HEADER_LEN {
        return Err(FeatureError::InvalidEncoding("output header truncated"));
    }
    if &bytes[0..4] != OUTPUT_MAGIC {
        return Err(FeatureError::InvalidEncoding("bad output magic"));
    }
    let version = read_u32_at(bytes, 4)?;
    if version != BATCH_ENCODING_VERSION {
        return Err(FeatureError::InvalidEncoding("unsupported output version"));
    }
    let row_count = read_u32_at(bytes, 8)? as usize;
    if row_count != action_counts.len() {
        return Err(FeatureError::InvalidEncoding("output row count mismatch"));
    }
    let max_actions = read_u32_at(bytes, 12)? as usize;
    let capacity = output_capacity(bytes, row_count, max_actions)?;
    let expected_len = OUTPUT_HEADER_LEN + capacity * 4 + capacity * max_actions * 4;
    if bytes.len() != expected_len {
        return Err(FeatureError::InvalidEncoding("bad output length"));
    }

    let values = OUTPUT_HEADER_LEN;
    let policy = values + capacity * 4;
    let mut rows = Vec::with_capacity(row_count);
    for (row_index, &action_count) in action_counts.iter().enumerate() {
        let action_count = action_count as usize;
        if action_count > max_actions {
            return Err(FeatureError::ActionOverflow {
                max: max_actions as u32,
                actual: action_count,
            });
        }
        let value = read_f32_at(bytes, values + row_index * 4)?;
        let mut policy_logits = Vec::with_capacity(action_count);
        for action_index in 0..action_count {
            policy_logits.push(read_f32_at(
                bytes,
                policy + (row_index * max_actions + action_index) * 4,
            )?);
        }
        rows.push(RowOutput {
            policy_logits,
            value,
        });
    }
    Ok(rows)
}

/// Validates per-row action counts against an encoded batch without copying
/// batch sections.
pub fn validate_batch_action_counts(bytes: &[u8], expected: &[u32]) -> FeatureResult<()> {
    let header = BatchHeader::parse(bytes)?;
    if expected.len() != header.row_count as usize {
        return Err(FeatureError::InvalidEncoding(
            "action count length mismatch",
        ));
    }

    let layout = BatchLayout::from_dims(
        header.batch_capacity as usize,
        header.max_nodes as usize,
        header.max_edges as usize,
        header.max_actions as usize,
        header.max_subjects as usize,
        header.node_attr_dim as usize,
    );
    if bytes.len() != layout.total_len {
        return Err(FeatureError::InvalidEncoding("bad batch length"));
    }

    for (row_index, &expected) in expected.iter().enumerate() {
        if expected > header.max_actions {
            return Err(FeatureError::ActionOverflow {
                max: header.max_actions,
                actual: expected as usize,
            });
        }
        let actual = read_u32_at(bytes, layout.action_count + row_index * 4)?;
        if actual != expected {
            return Err(FeatureError::InvalidEncoding("action count mismatch"));
        }
    }

    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct RowOutput {
    pub policy_logits: Vec<f32>,
    pub value: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FeatureBatchView {
    pub schema_hash: FeatureSchemaHash,
    pub batch_capacity: u32,
    pub row_count: u32,
    pub max_nodes: u32,
    pub max_edges: u32,
    pub max_actions: u32,
    pub max_subjects: u32,
    pub node_attr_dim: u32,
    pub node_count: Vec<u32>,
    pub node_tokens: Vec<u16>,
    pub node_attrs: Vec<f32>,
    pub edge_count: Vec<u32>,
    pub edge_src: Vec<u32>,
    pub edge_dst: Vec<u32>,
    pub edge_type: Vec<u8>,
    pub action_count: Vec<u32>,
    pub action_kind: Vec<u32>,
    pub action_prior: Vec<f32>,
    pub subject_count: Vec<u8>,
    pub action_subjects: Vec<u32>,
    pub position: Vec<[f32; 4]>,
    pub opponent_reward: Vec<f32>,
    pub opponent_present: Vec<u8>,
}

impl FeatureBatchView {
    pub fn parse(bytes: &[u8]) -> FeatureResult<Self> {
        let header = BatchHeader::parse(bytes)?;

        let layout = BatchLayout::from_dims(
            header.batch_capacity as usize,
            header.max_nodes as usize,
            header.max_edges as usize,
            header.max_actions as usize,
            header.max_subjects as usize,
            header.node_attr_dim as usize,
        );
        if bytes.len() != layout.total_len {
            return Err(FeatureError::InvalidEncoding("bad batch length"));
        }

        Ok(Self {
            schema_hash: header.schema_hash,
            batch_capacity: header.batch_capacity,
            row_count: header.row_count,
            max_nodes: header.max_nodes,
            max_edges: header.max_edges,
            max_actions: header.max_actions,
            max_subjects: header.max_subjects,
            node_attr_dim: header.node_attr_dim,
            node_count: read_u32_vec(bytes, layout.node_count, layout.b)?,
            node_tokens: read_u16_vec(bytes, layout.node_tokens, layout.b * layout.n)?,
            node_attrs: read_bf16_vec(bytes, layout.node_attrs, layout.b * layout.n * layout.d)?,
            edge_count: read_u32_vec(bytes, layout.edge_count, layout.b)?,
            edge_src: read_u16_widened_vec(bytes, layout.edge_src, layout.b * layout.e)?,
            edge_dst: read_u16_widened_vec(bytes, layout.edge_dst, layout.b * layout.e)?,
            edge_type: bytes[layout.edge_type..layout.edge_type + layout.b * layout.e].to_vec(),
            action_count: read_u32_vec(bytes, layout.action_count, layout.b)?,
            action_kind: read_u16_widened_vec(bytes, layout.action_kind, layout.b * layout.a)?,
            action_prior: read_bf16_vec(bytes, layout.action_prior, layout.b * layout.a)?,
            subject_count: bytes[layout.subject_count..layout.subject_count + layout.b * layout.a]
                .to_vec(),
            action_subjects: read_u16_widened_vec(
                bytes,
                layout.action_subjects,
                layout.b * layout.a * layout.s,
            )?,
            position: read_position_vec(bytes, layout.position, layout.b)?,
            opponent_reward: read_bf16_vec(bytes, layout.opponent_reward, layout.b)?,
            opponent_present: bytes[layout.opponent_present..layout.opponent_present + layout.b]
                .to_vec(),
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct BatchHeader {
    schema_hash: FeatureSchemaHash,
    batch_capacity: u32,
    row_count: u32,
    max_nodes: u32,
    max_edges: u32,
    max_actions: u32,
    max_subjects: u32,
    node_attr_dim: u32,
}

impl BatchHeader {
    fn parse(bytes: &[u8]) -> FeatureResult<Self> {
        if bytes.len() < BATCH_HEADER_LEN {
            return Err(FeatureError::InvalidEncoding("batch header truncated"));
        }
        if &bytes[0..4] != BATCH_MAGIC {
            return Err(FeatureError::InvalidEncoding("bad batch magic"));
        }
        let version = read_u32_at(bytes, 4)?;
        if version != BATCH_ENCODING_VERSION {
            return Err(FeatureError::InvalidEncoding("unsupported batch version"));
        }
        let header = Self {
            schema_hash: read_hash_at(bytes, 8)?,
            batch_capacity: read_u32_at(bytes, 40)?,
            row_count: read_u32_at(bytes, 44)?,
            max_nodes: read_u32_at(bytes, 48)?,
            max_edges: read_u32_at(bytes, 52)?,
            max_actions: read_u32_at(bytes, 56)?,
            max_subjects: read_u32_at(bytes, 60)?,
            node_attr_dim: read_u32_at(bytes, 64)?,
        };
        if header.batch_capacity == 0 {
            return Err(FeatureError::InvalidEncoding("zero batch capacity"));
        }
        if header.row_count > header.batch_capacity {
            return Err(FeatureError::InvalidEncoding("row count exceeds capacity"));
        }
        Ok(header)
    }
}

#[derive(Clone, Copy, Debug)]
struct BatchLayout {
    b: usize,
    n: usize,
    e: usize,
    a: usize,
    s: usize,
    d: usize,
    node_count: usize,
    node_tokens: usize,
    node_attrs: usize,
    edge_count: usize,
    edge_src: usize,
    edge_dst: usize,
    edge_type: usize,
    action_count: usize,
    action_kind: usize,
    action_prior: usize,
    subject_count: usize,
    action_subjects: usize,
    position: usize,
    opponent_reward: usize,
    opponent_present: usize,
    total_len: usize,
}

impl BatchLayout {
    fn new(schema: &FeatureSchema, batch_capacity: usize) -> Self {
        let config = schema.config();
        Self::from_dims(
            batch_capacity,
            config.max_nodes as usize,
            config.max_edges as usize,
            config.max_actions as usize,
            config.max_subjects as usize,
            config.node_attr_dim as usize,
        )
    }

    fn from_dims(b: usize, n: usize, e: usize, a: usize, s: usize, d: usize) -> Self {
        let mut cursor = BATCH_HEADER_LEN;
        let node_count = section(&mut cursor, b * 4);
        let node_tokens = section(&mut cursor, b * n * 2);
        let node_attrs = section(&mut cursor, b * n * d * 2);
        let edge_count = section(&mut cursor, b * 4);
        let edge_src = section(&mut cursor, b * e * 2);
        let edge_dst = section(&mut cursor, b * e * 2);
        let edge_type = section(&mut cursor, b * e);
        let action_count = section(&mut cursor, b * 4);
        let action_kind = section(&mut cursor, b * a * 2);
        let action_prior = section(&mut cursor, b * a * 2);
        let subject_count = section(&mut cursor, b * a);
        let action_subjects = section(&mut cursor, b * a * s * 2);
        let position = section(&mut cursor, b * 4 * 2);
        let opponent_reward = section(&mut cursor, b * 2);
        let opponent_present = section(&mut cursor, b);
        let total_len = align4(cursor);

        Self {
            b,
            n,
            e,
            a,
            s,
            d,
            node_count,
            node_tokens,
            node_attrs,
            edge_count,
            edge_src,
            edge_dst,
            edge_type,
            action_count,
            action_kind,
            action_prior,
            subject_count,
            action_subjects,
            position,
            opponent_reward,
            opponent_present,
            total_len,
        }
    }
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

fn fill_subject_padding(out: &mut [u8], layout: &BatchLayout) {
    let start = layout.action_subjects;
    let end = start + layout.b * layout.a * layout.s * 2;
    out[start..end].fill(0xff);
}

fn write_batch_header(out: &mut [u8], schema: &FeatureSchema, capacity: usize, row_count: usize) {
    out[0..4].copy_from_slice(BATCH_MAGIC);
    write_u32_at(out, 4, BATCH_ENCODING_VERSION);
    out[8..40].copy_from_slice(schema.hash().as_bytes());
    write_u32_at(out, 40, capacity as u32);
    write_u32_at(out, 44, row_count as u32);
    write_u32_at(out, 48, schema.config().max_nodes);
    write_u32_at(out, 52, schema.config().max_edges);
    write_u32_at(out, 56, schema.config().max_actions);
    write_u32_at(out, 60, schema.config().max_subjects);
    write_u32_at(out, 64, schema.config().node_attr_dim.into());
}

fn write_u16_at(out: &mut [u8], offset: usize, value: u16) {
    out[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u32_at(out: &mut [u8], offset: usize, value: u32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_bf16_at(out: &mut [u8], offset: usize, value: f32) {
    out[offset..offset + 2].copy_from_slice(&f32_to_bf16_bits(value).to_le_bytes());
}

fn narrow_index(value: u32, what: &'static str) -> FeatureResult<u16> {
    u16::try_from(value).map_err(|_| FeatureError::InvalidEncoding(what))
}

fn read_hash_at(bytes: &[u8], offset: usize) -> FeatureResult<FeatureSchemaHash> {
    let slice = bytes
        .get(offset..offset + 32)
        .ok_or(FeatureError::InvalidEncoding("hash truncated"))?;
    let mut out = [0u8; 32];
    out.copy_from_slice(slice);
    Ok(FeatureSchemaHash::from_bytes(out))
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

fn read_u16_vec(bytes: &[u8], offset: usize, count: usize) -> FeatureResult<Vec<u16>> {
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        let start = offset + index * 2;
        let slice = bytes
            .get(start..start + 2)
            .ok_or(FeatureError::InvalidEncoding("u16 section truncated"))?;
        out.push(u16::from_le_bytes(
            slice.try_into().expect("length checked"),
        ));
    }
    Ok(out)
}

fn read_u32_vec(bytes: &[u8], offset: usize, count: usize) -> FeatureResult<Vec<u32>> {
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        out.push(read_u32_at(bytes, offset + index * 4)?);
    }
    Ok(out)
}

fn read_bf16_at(bytes: &[u8], offset: usize) -> FeatureResult<f32> {
    let slice = bytes
        .get(offset..offset + 2)
        .ok_or(FeatureError::InvalidEncoding("bf16 truncated"))?;
    Ok(bf16_bits_to_f32(u16::from_le_bytes(
        slice.try_into().expect("length checked"),
    )))
}

fn read_bf16_vec(bytes: &[u8], offset: usize, count: usize) -> FeatureResult<Vec<f32>> {
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        out.push(read_bf16_at(bytes, offset + index * 2)?);
    }
    Ok(out)
}

fn read_u16_widened_vec(bytes: &[u8], offset: usize, count: usize) -> FeatureResult<Vec<u32>> {
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        let start = offset + index * 2;
        let slice = bytes
            .get(start..start + 2)
            .ok_or(FeatureError::InvalidEncoding("u16 section truncated"))?;
        out.push(u32::from(u16::from_le_bytes(
            slice.try_into().expect("length checked"),
        )));
    }
    Ok(out)
}

fn read_position_vec(bytes: &[u8], offset: usize, count: usize) -> FeatureResult<Vec<[f32; 4]>> {
    let mut out = Vec::with_capacity(count);
    for row in 0..count {
        let start = offset + row * 8;
        out.push([
            read_bf16_at(bytes, start)?,
            read_bf16_at(bytes, start + 2)?,
            read_bf16_at(bytes, start + 4)?,
            read_bf16_at(bytes, start + 6)?,
        ]);
    }
    Ok(out)
}

fn output_max_actions(bytes: &[u8]) -> FeatureResult<usize> {
    read_u32_at(bytes, 12).map(|value| value as usize)
}

fn output_capacity(bytes: &[u8], row_count: usize, max_actions: usize) -> FeatureResult<usize> {
    let body_len = bytes
        .len()
        .checked_sub(OUTPUT_HEADER_LEN)
        .ok_or(FeatureError::InvalidEncoding("output header truncated"))?;
    let row_width = (1 + max_actions)
        .checked_mul(4)
        .ok_or(FeatureError::InvalidEncoding("output length overflow"))?;
    if max_actions == 0 || row_width == 0 || body_len % row_width != 0 {
        return Err(FeatureError::InvalidEncoding("bad output length"));
    }
    let capacity = body_len / row_width;
    if row_count > capacity {
        return Err(FeatureError::InvalidEncoding(
            "output row count exceeds capacity",
        ));
    }
    Ok(capacity)
}

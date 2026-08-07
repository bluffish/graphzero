use graphs_compiler::{
    actions::CompilerStateView,
    ir::{OperandSlot, ResultSlot, ScopeId},
};
use gz_features::{FeatureEdge, FeatureError, FeatureResult, FeatureSchemaConfig};
use std::collections::HashMap;

use super::{
    state::{
        BODY_TOKEN, COMPILER_NODE_ATTR_DIM, COMPONENT_TOKEN, EntityIndexes, MODULE_TOKEN,
        OP_TOKEN_BASE, StateFeatures, append_node, checked_sum, count_feature, index_feature,
    },
    value::{value_attrs, value_token},
};

const SLOT_BUCKETS: u8 = 16;
const MODULE_TO_COMPONENT: u8 = 0;
const SCOPE_TO_OPERATION: u8 = 1;
const OPERATION_TO_BODY: u8 = 2;
const CONTROL: u8 = 3;
const MODULE_TENSOR_BASE: u8 = 4;
const COMPONENT_INPUT_BASE: u8 = MODULE_TENSOR_BASE + SLOT_BUCKETS;
const COMPONENT_OUTPUT_BASE: u8 = COMPONENT_INPUT_BASE + SLOT_BUCKETS;
const BODY_ARGUMENT_BASE: u8 = COMPONENT_OUTPUT_BASE + SLOT_BUCKETS;
const OPERAND_BASE: u8 = BODY_ARGUMENT_BASE + SLOT_BUCKETS;
const RESULT_BASE: u8 = OPERAND_BASE + SLOT_BUCKETS;
pub(super) const COMPACT_EDGE_TYPES: u8 = RESULT_BASE + SLOT_BUCKETS;

pub(super) fn state_features(
    view: CompilerStateView<'_>,
    config: &FeatureSchemaConfig,
) -> FeatureResult<StateFeatures> {
    if view.node_attribute_fingerprints.len() != view.canonical_node_order.len()
        || view.value_type_fingerprints.len() != view.canonical_value_order.len()
    {
        return Err(FeatureError::InvalidRow(
            "compiler semantic fingerprint order is inconsistent",
        ));
    }
    let module = view.module;
    let module_view = module.view();
    let graph = module_view.graph;
    let node_count = checked_sum(
        1,
        module_view.component_ids.len(),
        view.canonical_body_order.len(),
        view.canonical_node_order.len(),
    )?
    .checked_add(view.canonical_value_order.len())
    .ok_or(FeatureError::InvalidRow(
        "compiler compact feature node count overflow",
    ))?;
    if node_count > config.max_nodes as usize {
        return Err(FeatureError::NodeOverflow {
            max: config.max_nodes,
            actual: u32::try_from(node_count).unwrap_or(u32::MAX),
        });
    }

    let component_offset = 1;
    let body_offset = component_offset + module_view.component_ids.len();
    let operation_offset = body_offset + view.canonical_body_order.len();
    let value_offset = operation_offset + view.canonical_node_order.len();
    let indexes = EntityIndexes {
        components: indexes(module_view.component_ids, component_offset),
        bodies: indexes(view.canonical_body_order, body_offset),
        nodes: indexes(view.canonical_node_order, operation_offset),
        values: indexes(view.canonical_value_order, value_offset),
    };

    let mut node_tokens = Vec::with_capacity(node_count);
    let mut node_attrs = Vec::with_capacity(node_count * usize::from(COMPILER_NODE_ATTR_DIM));
    append_node(
        &mut node_tokens,
        &mut node_attrs,
        MODULE_TOKEN,
        [
            count_feature(module_view.component_ids.len()),
            count_feature(view.canonical_body_order.len()),
            count_feature(view.canonical_node_order.len()),
            count_feature(view.canonical_value_order.len()),
            0.0,
            count_feature(module_view.state_ids.len()),
            count_feature(module_view.module_tensors.ordered().len()),
            count_feature(module_view.control_edges.iter().len()),
        ],
        Some(view.module_fingerprint.as_ref()),
    );
    for component in module_view.component_ids {
        let component = &graph[*component];
        append_node(
            &mut node_tokens,
            &mut node_attrs,
            COMPONENT_TOKEN,
            [
                count_feature(component.inputs.len()),
                count_feature(component.outputs.len()),
                count_feature(component.nodes.len()),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            None,
        );
    }
    for body in view.canonical_body_order {
        let body = &graph[*body];
        append_node(
            &mut node_tokens,
            &mut node_attrs,
            BODY_TOKEN,
            [
                count_feature(body.args.len()),
                count_feature(body.nodes.len()),
                index_feature(body.slot.0),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            None,
        );
    }
    for (node, fingerprint) in view
        .canonical_node_order
        .iter()
        .zip(view.node_attribute_fingerprints)
    {
        let node = &graph[*node];
        let token = OP_TOKEN_BASE
            .checked_add(node.op.index())
            .and_then(|token| u16::try_from(token).ok())
            .ok_or(FeatureError::InvalidRow(
                "compiler operation token exceeds schema vocabulary",
            ))?;
        append_node(
            &mut node_tokens,
            &mut node_attrs,
            token,
            [
                count_feature(node.operands.len()),
                count_feature(node.results.len()),
                count_feature(node.bodies.len()),
                count_feature(node.operand_slots()),
                count_feature(node.result_slots()),
                if matches!(node.parent, ScopeId::Body(_)) {
                    1.0
                } else {
                    0.0
                },
                0.0,
                0.0,
            ],
            Some(fingerprint.as_ref()),
        );
    }
    for (value, fingerprint) in view
        .canonical_value_order
        .iter()
        .zip(view.value_type_fingerprints)
    {
        let value = &graph[*value];
        append_node(
            &mut node_tokens,
            &mut node_attrs,
            value_token(&module[value.ty]),
            value_attrs(value, &module[value.ty]),
            Some(fingerprint.as_ref()),
        );
    }

    let mut edges = Vec::new();
    for component in module_view.component_ids {
        push(
            &mut edges,
            0,
            indexes.components[component],
            MODULE_TO_COMPONENT,
        );
        let owner = indexes.components[component];
        for (ordinal, input) in graph[*component].inputs.iter().enumerate() {
            push(
                &mut edges,
                owner,
                indexes.values[&input.value],
                positional(COMPONENT_INPUT_BASE, ordinal),
            );
        }
        for (ordinal, output) in graph[*component].outputs.iter().enumerate() {
            push(
                &mut edges,
                owner,
                indexes.values[&output.value],
                positional(COMPONENT_OUTPUT_BASE, ordinal),
            );
        }
    }
    for body in view.canonical_body_order {
        let owner = indexes.bodies[body];
        push(
            &mut edges,
            indexes.nodes[&graph[*body].owner],
            owner,
            OPERATION_TO_BODY,
        );
        for (ordinal, value) in graph[*body].args.iter().enumerate() {
            push(
                &mut edges,
                owner,
                indexes.values[value],
                positional(BODY_ARGUMENT_BASE, ordinal),
            );
        }
    }
    for node in view.canonical_node_order {
        let record = &graph[*node];
        let owner = indexes.nodes[node];
        let scope = match record.parent {
            ScopeId::Component(component) => indexes.components[&component],
            ScopeId::Body(body) => indexes.bodies[&body],
        };
        push(&mut edges, scope, owner, SCOPE_TO_OPERATION);
        for slot in 0..record.operand_slots() {
            for value in &record[OperandSlot::new(slot)] {
                push(
                    &mut edges,
                    owner,
                    indexes.values[value],
                    positional(OPERAND_BASE, slot),
                );
            }
        }
        for slot in 0..record.result_slots() {
            for value in &record[ResultSlot::new(slot)] {
                push(
                    &mut edges,
                    owner,
                    indexes.values[value],
                    positional(RESULT_BASE, slot),
                );
            }
        }
    }
    for (ordinal, (value, _)) in module.module_tensor_values().enumerate() {
        push(
            &mut edges,
            0,
            indexes.values[&value],
            positional(MODULE_TENSOR_BASE, ordinal),
        );
    }
    for edge in module_view.control_edges.iter() {
        push(
            &mut edges,
            indexes.nodes[&edge.before],
            indexes.nodes[&edge.after],
            CONTROL,
        );
    }
    if edges.len() > config.max_edges as usize {
        return Err(FeatureError::EdgeOverflow {
            max: config.max_edges,
            actual: edges.len(),
        });
    }
    Ok(StateFeatures {
        node_count: node_count as u32,
        node_tokens,
        node_attrs,
        edges,
        indexes,
    })
}

fn indexes<T: Copy + Eq + std::hash::Hash>(order: &[T], offset: usize) -> HashMap<T, u32> {
    order
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| (value, u32::try_from(offset + index).expect("bounded nodes")))
        .collect()
}

fn positional(base: u8, ordinal: usize) -> u8 {
    base + u8::try_from(ordinal.min(usize::from(SLOT_BUCKETS - 1))).expect("bounded slot")
}

fn push(edges: &mut Vec<FeatureEdge>, src: u32, dst: u32, edge_type: u8) {
    edges.push(FeatureEdge {
        src,
        dst,
        edge_type,
    });
}

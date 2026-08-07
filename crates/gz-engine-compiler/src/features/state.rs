use graphs_compiler::{
    actions::CompilerStateView,
    ir::{BodyId, ComponentId, NodeId, OperandSlot, ResultSlot, ScopeId, ValueId},
    matching::BoundObject,
};
use gz_features::{FeatureEdge, FeatureError, FeatureResult, FeatureSchemaConfig};
use std::{collections::HashMap, hash::Hash};

use super::value::{value_attrs, value_token};

// Canonical IR objects occupy the prefix. Explicit endpoint nodes retain
// operand/result slot and variadic order without expanding the shared edge
// wire format. Attrs are eight typed scalars followed by a 32-byte semantic
// fingerprint projected to [-1, 1].
pub(super) const COMPILER_NODE_ATTR_DIM: u16 = 40;
pub(super) const COMPILER_EDGE_TYPES: u8 = 6;
pub(super) const MODULE_TOKEN: u16 = 1;
pub(super) const COMPONENT_TOKEN: u16 = 2;
pub(super) const BODY_TOKEN: u16 = 3;
const MODULE_TENSOR_PORT_TOKEN: u16 = 16;
const COMPONENT_INPUT_PORT_TOKEN: u16 = 17;
const COMPONENT_OUTPUT_PORT_TOKEN: u16 = 18;
const BODY_ARGUMENT_PORT_TOKEN: u16 = 19;
const OPERAND_PORT_TOKEN: u16 = 20;
const RESULT_PORT_TOKEN: u16 = 21;
pub(super) const OP_TOKEN_BASE: usize = 22;

const OWNER_TO_PORT_EDGE: u8 = 0;
const PORT_TO_VALUE_EDGE: u8 = 1;
const MODULE_TO_COMPONENT_EDGE: u8 = 2;
const SCOPE_TO_OPERATION_EDGE: u8 = 3;
const OPERATION_TO_BODY_EDGE: u8 = 4;
const CONTROL_EDGE: u8 = 5;

pub(super) struct StateFeatures {
    pub(super) node_count: u32,
    pub(super) node_tokens: Vec<u16>,
    pub(super) node_attrs: Vec<f32>,
    pub(super) edges: Vec<FeatureEdge>,
    pub(super) indexes: EntityIndexes,
}

pub(super) struct EntityIndexes {
    pub(super) components: HashMap<ComponentId, u32>,
    pub(super) bodies: HashMap<BodyId, u32>,
    pub(super) nodes: HashMap<NodeId, u32>,
    pub(super) values: HashMap<ValueId, u32>,
}

impl EntityIndexes {
    pub(super) fn subject(&self, subject: BoundObject) -> Option<u32> {
        match subject {
            BoundObject::Op(node) => self.nodes.get(&node).copied(),
            BoundObject::Value(value) => self.values.get(&value).copied(),
            BoundObject::Component(component) => self.components.get(&component).copied(),
            BoundObject::Body(body) => self.bodies.get(&body).copied(),
        }
    }

    fn scope(&self, scope: ScopeId) -> Option<u32> {
        match scope {
            ScopeId::Component(component) => self.components.get(&component).copied(),
            ScopeId::Body(body) => self.bodies.get(&body).copied(),
        }
    }
}

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
    let component_ports = checked_total(module_view.component_ids.iter().map(|component| {
        let component = &graph[*component];
        component.inputs.len() + component.outputs.len()
    }))?;
    let body_ports = checked_total(
        view.canonical_body_order
            .iter()
            .map(|body| graph[*body].args.len()),
    )?;
    let operation_ports = checked_total(view.canonical_node_order.iter().map(|node| {
        let node = &graph[*node];
        node.operands.len() + node.results.len()
    }))?;
    let port_count = checked_sum(
        module.module_tensor_values().count(),
        component_ports,
        body_ports,
        operation_ports,
    )?;
    let entity_count = checked_sum(
        1,
        module_view.component_ids.len(),
        view.canonical_body_order.len(),
        checked_sum(
            view.canonical_node_order.len(),
            view.canonical_value_order.len(),
            0,
            0,
        )?,
    )?;
    let node_count = entity_count
        .checked_add(port_count)
        .ok_or(FeatureError::InvalidRow(
            "compiler feature node count overflow",
        ))?;
    if node_count > config.max_nodes as usize {
        return Err(FeatureError::NodeOverflow {
            max: config.max_nodes,
            actual: u32::try_from(node_count).unwrap_or(u32::MAX),
        });
    }

    let edge_count = module_view
        .component_ids
        .len()
        .checked_add(view.canonical_node_order.len())
        .and_then(|count| count.checked_add(view.canonical_body_order.len()))
        .and_then(|count| {
            port_count
                .checked_mul(2)
                .and_then(|ports| count.checked_add(ports))
        })
        .and_then(|count| count.checked_add(module_view.control_edges.iter().len()))
        .ok_or(FeatureError::InvalidRow(
            "compiler feature edge count overflow",
        ))?;
    if edge_count > config.max_edges as usize {
        return Err(FeatureError::EdgeOverflow {
            max: config.max_edges,
            actual: edge_count,
        });
    }

    let component_offset = 1;
    let body_offset = component_offset + module_view.component_ids.len();
    let node_offset = body_offset + view.canonical_body_order.len();
    let value_offset = node_offset + view.canonical_node_order.len();
    let indexes = EntityIndexes {
        components: ordered_indexes(module_view.component_ids, component_offset),
        bodies: ordered_indexes(view.canonical_body_order, body_offset),
        nodes: ordered_indexes(view.canonical_node_order, node_offset),
        values: ordered_indexes(view.canonical_value_order, value_offset),
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
            count_feature(port_count),
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
                bool_feature(matches!(node.parent, ScopeId::Body(_))),
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

    let mut edges = Vec::with_capacity(edge_count);
    for component in module_view.component_ids {
        edges.push(FeatureEdge {
            src: 0,
            dst: indexes.components[component],
            edge_type: MODULE_TO_COMPONENT_EDGE,
        });
    }
    for node in view.canonical_node_order {
        let record = &graph[*node];
        edges.push(FeatureEdge {
            src: indexes
                .scope(record.parent)
                .ok_or(FeatureError::InvalidRow(
                    "compiler operation scope is absent from the feature graph",
                ))?,
            dst: indexes.nodes[node],
            edge_type: SCOPE_TO_OPERATION_EDGE,
        });
    }
    for body in view.canonical_body_order {
        let record = &graph[*body];
        edges.push(FeatureEdge {
            src: indexes.nodes[&record.owner],
            dst: indexes.bodies[body],
            edge_type: OPERATION_TO_BODY_EDGE,
        });
    }

    for (ordinal, (value, _)) in module.module_tensor_values().enumerate() {
        append_port(
            &mut node_tokens,
            &mut node_attrs,
            &mut edges,
            MODULE_TENSOR_PORT_TOKEN,
            port_attrs(ordinal, 0, 0, 0.0),
            0,
            indexes.values[&value],
        );
    }
    for component in module_view.component_ids {
        let owner = indexes.components[component];
        let record = &graph[*component];
        for (ordinal, input) in record.inputs.iter().enumerate() {
            append_port(
                &mut node_tokens,
                &mut node_attrs,
                &mut edges,
                COMPONENT_INPUT_PORT_TOKEN,
                port_attrs(
                    ordinal,
                    0,
                    0,
                    bool_feature(!matches!(
                        input.binding,
                        graphs_compiler::ir::ComponentInputBinding::External
                    )),
                ),
                owner,
                indexes.values[&input.value],
            );
        }
        for (ordinal, output) in record.outputs.iter().enumerate() {
            append_port(
                &mut node_tokens,
                &mut node_attrs,
                &mut edges,
                COMPONENT_OUTPUT_PORT_TOKEN,
                port_attrs(
                    ordinal,
                    0,
                    0,
                    bool_feature(!matches!(
                        output.binding,
                        graphs_compiler::ir::ComponentOutputBinding::External
                    )),
                ),
                owner,
                indexes.values[&output.value],
            );
        }
    }
    for body in view.canonical_body_order {
        let owner = indexes.bodies[body];
        for (ordinal, value) in graph[*body].args.iter().copied().enumerate() {
            append_port(
                &mut node_tokens,
                &mut node_attrs,
                &mut edges,
                BODY_ARGUMENT_PORT_TOKEN,
                port_attrs(ordinal, 0, 0, 0.0),
                owner,
                indexes.values[&value],
            );
        }
    }
    for node in view.canonical_node_order {
        let owner = indexes.nodes[node];
        let record = &graph[*node];
        let mut absolute = 0;
        for slot in 0..record.operand_slots() {
            for (within_slot, value) in record[OperandSlot::new(slot)].iter().copied().enumerate() {
                append_port(
                    &mut node_tokens,
                    &mut node_attrs,
                    &mut edges,
                    OPERAND_PORT_TOKEN,
                    port_attrs(absolute, slot, within_slot, 0.0),
                    owner,
                    indexes.values[&value],
                );
                absolute += 1;
            }
        }
        absolute = 0;
        for slot in 0..record.result_slots() {
            for (within_slot, value) in record[ResultSlot::new(slot)].iter().copied().enumerate() {
                append_port(
                    &mut node_tokens,
                    &mut node_attrs,
                    &mut edges,
                    RESULT_PORT_TOKEN,
                    port_attrs(absolute, slot, within_slot, 0.0),
                    owner,
                    indexes.values[&value],
                );
                absolute += 1;
            }
        }
    }

    let mut control_edges =
        module_view
            .control_edges
            .iter()
            .map(|edge| {
                Ok(FeatureEdge {
                    src: indexes.nodes.get(&edge.before).copied().ok_or(
                        FeatureError::InvalidRow(
                            "compiler control source is absent from the feature graph",
                        ),
                    )?,
                    dst: indexes.nodes.get(&edge.after).copied().ok_or(
                        FeatureError::InvalidRow(
                            "compiler control destination is absent from the feature graph",
                        ),
                    )?,
                    edge_type: CONTROL_EDGE,
                })
            })
            .collect::<FeatureResult<Vec<_>>>()?;
    control_edges.sort_unstable_by_key(|edge| (edge.src, edge.dst));
    edges.extend(control_edges);

    debug_assert_eq!(node_tokens.len(), node_count);
    debug_assert_eq!(edges.len(), edge_count);
    Ok(StateFeatures {
        node_count: node_count as u32,
        node_tokens,
        node_attrs,
        edges,
        indexes,
    })
}

fn ordered_indexes<T: Copy + Eq + Hash>(order: &[T], offset: usize) -> HashMap<T, u32> {
    order
        .iter()
        .copied()
        .enumerate()
        .map(|(index, id)| {
            (
                id,
                u32::try_from(offset + index)
                    .expect("feature node count was checked against the u16 schema limit"),
            )
        })
        .collect()
}

fn append_port(
    node_tokens: &mut Vec<u16>,
    node_attrs: &mut Vec<f32>,
    edges: &mut Vec<FeatureEdge>,
    token: u16,
    attrs: [f32; 8],
    owner: u32,
    value: u32,
) {
    let port = u32::try_from(node_tokens.len())
        .expect("feature node count was checked against the u16 schema limit");
    append_node(node_tokens, node_attrs, token, attrs, None);
    edges.push(FeatureEdge {
        src: owner,
        dst: port,
        edge_type: OWNER_TO_PORT_EDGE,
    });
    edges.push(FeatureEdge {
        src: port,
        dst: value,
        edge_type: PORT_TO_VALUE_EDGE,
    });
}

pub(super) fn append_node(
    node_tokens: &mut Vec<u16>,
    node_attrs: &mut Vec<f32>,
    token: u16,
    attrs: [f32; 8],
    fingerprint: Option<&[u8; 32]>,
) {
    node_tokens.push(token);
    node_attrs.extend(attrs);
    match fingerprint {
        Some(fingerprint) => node_attrs.extend(
            fingerprint
                .iter()
                .map(|byte| (f32::from(*byte) - 127.5) / 127.5),
        ),
        None => node_attrs.extend([0.0; 32]),
    }
}

fn port_attrs(absolute: usize, slot: usize, within_slot: usize, binding: f32) -> [f32; 8] {
    [
        index_feature(absolute),
        index_feature(slot),
        index_feature(within_slot),
        binding,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
}

pub(super) fn count_feature(count: usize) -> f32 {
    let count = count as f32;
    count / (count + 1.0)
}

pub(super) fn index_feature(index: usize) -> f32 {
    let index = index as f32;
    (index + 1.0) / (index + 2.0)
}

const fn bool_feature(value: bool) -> f32 {
    if value { 1.0 } else { 0.0 }
}

pub(super) fn checked_sum(a: usize, b: usize, c: usize, d: usize) -> FeatureResult<usize> {
    a.checked_add(b)
        .and_then(|sum| sum.checked_add(c))
        .and_then(|sum| sum.checked_add(d))
        .ok_or(FeatureError::InvalidRow(
            "compiler feature cardinality overflow",
        ))
}

fn checked_total(values: impl IntoIterator<Item = usize>) -> FeatureResult<usize> {
    values.into_iter().try_fold(0_usize, |sum, value| {
        sum.checked_add(value).ok_or(FeatureError::InvalidRow(
            "compiler feature cardinality overflow",
        ))
    })
}

use crate::graph::{GraphBody, NO_NODE, OpCode, compact_graph};
use std::collections::HashSet;
use std::fmt;

const MAX_MATCHED: usize = 8;
pub const RULE_COUNT: u32 = 44;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum RuleId {
    AndFalse = 0,
    AndTrue = 1,
    OrFalse = 2,
    OrTrue = 3,
    NotFalse = 4,
    NotTrue = 5,
    AndIdempotent = 6,
    OrIdempotent = 7,
    AndComplement = 8,
    OrComplement = 9,
    DoubleNegation = 10,
    CommuteAnd = 11,
    CommuteOr = 12,
    AssociateAndLeft = 13,
    AssociateAndRight = 14,
    AssociateOrLeft = 15,
    AssociateOrRight = 16,
    AbsorbOrAnd = 17,
    AbsorbAndOr = 18,
    DeMorganNotAnd = 19,
    DeMorganNotOr = 20,
    DeMorganOrNots = 21,
    DeMorganAndNots = 22,
    DistributeOrFactor = 23,
    DistributeOrExpand = 24,
    DistributeAndFactor = 25,
    DistributeAndExpand = 26,
    ConsensusOrAdd = 27,
    ConsensusOrRemove = 28,
    ConsensusAndAdd = 29,
    ConsensusAndRemove = 30,
    AndFalseInverse = 31,
    AndTrueInverse = 32,
    OrFalseInverse = 33,
    OrTrueInverse = 34,
    NotFalseInverse = 35,
    NotTrueInverse = 36,
    AndIdempotentInverse = 37,
    OrIdempotentInverse = 38,
    AndComplementInverse = 39,
    OrComplementInverse = 40,
    DoubleNegationInverse = 41,
    AbsorbOrAndInverse = 42,
    AbsorbAndOrInverse = 43,
}

impl RuleId {
    pub(crate) fn from_u16(value: u16) -> Result<Self, RuleError> {
        use RuleId::*;

        match value {
            0 => Ok(AndFalse),
            1 => Ok(AndTrue),
            2 => Ok(OrFalse),
            3 => Ok(OrTrue),
            4 => Ok(NotFalse),
            5 => Ok(NotTrue),
            6 => Ok(AndIdempotent),
            7 => Ok(OrIdempotent),
            8 => Ok(AndComplement),
            9 => Ok(OrComplement),
            10 => Ok(DoubleNegation),
            11 => Ok(CommuteAnd),
            12 => Ok(CommuteOr),
            13 => Ok(AssociateAndLeft),
            14 => Ok(AssociateAndRight),
            15 => Ok(AssociateOrLeft),
            16 => Ok(AssociateOrRight),
            17 => Ok(AbsorbOrAnd),
            18 => Ok(AbsorbAndOr),
            19 => Ok(DeMorganNotAnd),
            20 => Ok(DeMorganNotOr),
            21 => Ok(DeMorganOrNots),
            22 => Ok(DeMorganAndNots),
            23 => Ok(DistributeOrFactor),
            24 => Ok(DistributeOrExpand),
            25 => Ok(DistributeAndFactor),
            26 => Ok(DistributeAndExpand),
            27 => Ok(ConsensusOrAdd),
            28 => Ok(ConsensusOrRemove),
            29 => Ok(ConsensusAndAdd),
            30 => Ok(ConsensusAndRemove),
            31 => Ok(AndFalseInverse),
            32 => Ok(AndTrueInverse),
            33 => Ok(OrFalseInverse),
            34 => Ok(OrTrueInverse),
            35 => Ok(NotFalseInverse),
            36 => Ok(NotTrueInverse),
            37 => Ok(AndIdempotentInverse),
            38 => Ok(OrIdempotentInverse),
            39 => Ok(AndComplementInverse),
            40 => Ok(OrComplementInverse),
            41 => Ok(DoubleNegationInverse),
            42 => Ok(AbsorbOrAndInverse),
            43 => Ok(AbsorbAndOrInverse),
            _ => Err(RuleError::UnknownRule(value)),
        }
    }

    pub(crate) const fn as_u16(self) -> u16 {
        self as u16
    }
}

pub fn rule_name(rule_id: u16) -> &'static str {
    match RuleId::from_u16(rule_id) {
        Ok(RuleId::AndFalse) => "AndFalse",
        Ok(RuleId::AndTrue) => "AndTrue",
        Ok(RuleId::OrFalse) => "OrFalse",
        Ok(RuleId::OrTrue) => "OrTrue",
        Ok(RuleId::NotFalse) => "NotFalse",
        Ok(RuleId::NotTrue) => "NotTrue",
        Ok(RuleId::AndIdempotent) => "AndIdempotent",
        Ok(RuleId::OrIdempotent) => "OrIdempotent",
        Ok(RuleId::AndComplement) => "AndComplement",
        Ok(RuleId::OrComplement) => "OrComplement",
        Ok(RuleId::DoubleNegation) => "DoubleNegation",
        Ok(RuleId::CommuteAnd) => "CommuteAnd",
        Ok(RuleId::CommuteOr) => "CommuteOr",
        Ok(RuleId::AssociateAndLeft) => "AssociateAndLeft",
        Ok(RuleId::AssociateAndRight) => "AssociateAndRight",
        Ok(RuleId::AssociateOrLeft) => "AssociateOrLeft",
        Ok(RuleId::AssociateOrRight) => "AssociateOrRight",
        Ok(RuleId::AbsorbOrAnd) => "AbsorbOrAnd",
        Ok(RuleId::AbsorbAndOr) => "AbsorbAndOr",
        Ok(RuleId::DeMorganNotAnd) => "DeMorganNotAnd",
        Ok(RuleId::DeMorganNotOr) => "DeMorganNotOr",
        Ok(RuleId::DeMorganOrNots) => "DeMorganOrNots",
        Ok(RuleId::DeMorganAndNots) => "DeMorganAndNots",
        Ok(RuleId::DistributeOrFactor) => "DistributeOrFactor",
        Ok(RuleId::DistributeOrExpand) => "DistributeOrExpand",
        Ok(RuleId::DistributeAndFactor) => "DistributeAndFactor",
        Ok(RuleId::DistributeAndExpand) => "DistributeAndExpand",
        Ok(RuleId::ConsensusOrAdd) => "ConsensusOrAdd",
        Ok(RuleId::ConsensusOrRemove) => "ConsensusOrRemove",
        Ok(RuleId::ConsensusAndAdd) => "ConsensusAndAdd",
        Ok(RuleId::ConsensusAndRemove) => "ConsensusAndRemove",
        Ok(RuleId::AndFalseInverse) => "AndFalseInverse",
        Ok(RuleId::AndTrueInverse) => "AndTrueInverse",
        Ok(RuleId::OrFalseInverse) => "OrFalseInverse",
        Ok(RuleId::OrTrueInverse) => "OrTrueInverse",
        Ok(RuleId::NotFalseInverse) => "NotFalseInverse",
        Ok(RuleId::NotTrueInverse) => "NotTrueInverse",
        Ok(RuleId::AndIdempotentInverse) => "AndIdempotentInverse",
        Ok(RuleId::OrIdempotentInverse) => "OrIdempotentInverse",
        Ok(RuleId::AndComplementInverse) => "AndComplementInverse",
        Ok(RuleId::OrComplementInverse) => "OrComplementInverse",
        Ok(RuleId::DoubleNegationInverse) => "DoubleNegationInverse",
        Ok(RuleId::AbsorbOrAndInverse) => "AbsorbOrAndInverse",
        Ok(RuleId::AbsorbAndOrInverse) => "AbsorbAndOrInverse",
        Err(_) => "Unknown",
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct RawCandidate {
    pub rule_id: u16,
    pub root: u32,
    pub match_len: u8,
    pub matched: [u32; MAX_MATCHED],
}

impl RawCandidate {
    pub(crate) fn matched_slice(&self) -> &[u32] {
        &self.matched[..usize::from(self.match_len)]
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Pair {
    a: u32,
    b: u32,
    valid: bool,
}

#[derive(Clone, Copy)]
struct SignalTables<'a> {
    not_signals: &'a [i32],
    and_pairs: &'a [Pair],
    or_pairs: &'a [Pair],
}

#[derive(Clone, Copy)]
struct ConsensusAdd {
    term_op: OpCode,
    rule: RuleId,
}

#[derive(Clone, Copy)]
struct ConsensusRemove {
    inner_op: OpCode,
    consensus_op: OpCode,
    rule: RuleId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct CandidateKey([u32; 11]);

struct Builder {
    include_reverse_constant_folding: bool,
    remaining_capacity: i32,
    items: Vec<RawCandidate>,
    seen: HashSet<CandidateKey>,
}

impl Builder {
    fn new(graph: &GraphBody, include_reverse_constant_folding: bool) -> Self {
        Self {
            include_reverse_constant_folding,
            remaining_capacity: i32::from(graph.capacity) - graph.op.len() as i32,
            items: Vec::with_capacity(1024),
            seen: HashSet::with_capacity(2048),
        }
    }

    fn add(&mut self, rule: RuleId, root: u32, rest: &[u32]) -> Result<(), RuleError> {
        if !self.include_reverse_constant_folding && reverse_constant_folding_rule(rule) {
            return Ok(());
        }
        if replacement_extra_nodes(rule) > self.remaining_capacity {
            return Ok(());
        }
        if rest.len() + 1 > MAX_MATCHED {
            return Err(RuleError::BadMatchLength);
        }

        let mut matched = [0; MAX_MATCHED];
        matched[0] = root;
        for (index, node) in rest.iter().copied().enumerate() {
            matched[index + 1] = node;
        }

        let candidate = RawCandidate {
            rule_id: rule.as_u16(),
            root,
            match_len: (rest.len() + 1) as u8,
            matched,
        };

        if self.seen.insert(candidate_key(candidate)) {
            self.items.push(candidate);
        }

        Ok(())
    }
}

fn candidate_key(candidate: RawCandidate) -> CandidateKey {
    let mut key = [0; 11];
    key[0] = candidate.rule_id.into();
    key[1] = candidate.root;
    key[2] = candidate.match_len.into();
    key[3..].copy_from_slice(&candidate.matched);
    CandidateKey(key)
}

fn pair_for(node: u32, kind: OpCode, and_pairs: &[Pair], or_pairs: &[Pair]) -> Pair {
    if kind == OpCode::And {
        and_pairs[node as usize]
    } else {
        or_pairs[node as usize]
    }
}

fn binary_of(
    node: u32,
    kind: OpCode,
    left: u32,
    right: u32,
    and_pairs: &[Pair],
    or_pairs: &[Pair],
) -> bool {
    let pair = pair_for(node, kind, and_pairs, or_pairs);
    pair.valid && ((pair.a == left && pair.b == right) || (pair.a == right && pair.b == left))
}

fn consensus_add_bindings(
    out: &mut Builder,
    root: u32,
    first: u32,
    second: u32,
    spec: ConsensusAdd,
    tables: SignalTables<'_>,
) -> Result<(), RuleError> {
    let first_pair = pair_for(first, spec.term_op, tables.and_pairs, tables.or_pairs);
    let second_pair = pair_for(second, spec.term_op, tables.and_pairs, tables.or_pairs);
    if !first_pair.valid || !second_pair.valid {
        return Ok(());
    }

    let a_orders = [first_pair.a, first_pair.b];
    let b_orders = [first_pair.b, first_pair.a];

    for x in 0..2 {
        if x == 1 && first_pair.a == first_pair.b {
            continue;
        }
        for y in 0..2 {
            if y == 1 && second_pair.a == second_pair.b {
                continue;
            }
            let not_a = if y == 0 { second_pair.a } else { second_pair.b };
            let inner = tables.not_signals[not_a as usize];
            if inner >= 0 && inner as u32 == a_orders[x] {
                out.add(
                    spec.rule,
                    root,
                    &[
                        first,
                        second,
                        a_orders[x],
                        b_orders[x],
                        if y == 0 { second_pair.b } else { second_pair.a },
                        not_a,
                    ],
                )?;
            }
        }
    }

    Ok(())
}

fn consensus_remove_bindings(
    out: &mut Builder,
    root: u32,
    inner_node: u32,
    consensus_term: u32,
    spec: ConsensusRemove,
    tables: SignalTables<'_>,
) -> Result<(), RuleError> {
    let pair = pair_for(inner_node, spec.inner_op, tables.and_pairs, tables.or_pairs);
    if !pair.valid {
        return Ok(());
    }

    let orders = [(pair.a, pair.b), (pair.b, pair.a)];

    for (first, second) in orders {
        if first == pair.b && pair.a == pair.b {
            continue;
        }

        let first_pair = pair_for(first, spec.consensus_op, tables.and_pairs, tables.or_pairs);
        let second_pair = pair_for(second, spec.consensus_op, tables.and_pairs, tables.or_pairs);
        if !first_pair.valid || !second_pair.valid {
            continue;
        }

        let a_orders = [first_pair.a, first_pair.b];
        let b_orders = [first_pair.b, first_pair.a];

        for x in 0..2 {
            if x == 1 && first_pair.a == first_pair.b {
                continue;
            }
            for y in 0..2 {
                if y == 1 && second_pair.a == second_pair.b {
                    continue;
                }
                let not_a = if y == 0 { second_pair.a } else { second_pair.b };
                let c_node = if y == 0 { second_pair.b } else { second_pair.a };
                let inner = tables.not_signals[not_a as usize];
                if inner >= 0
                    && inner as u32 == a_orders[x]
                    && binary_of(
                        consensus_term,
                        spec.consensus_op,
                        b_orders[x],
                        c_node,
                        tables.and_pairs,
                        tables.or_pairs,
                    )
                {
                    out.add(
                        spec.rule,
                        root,
                        &[
                            inner_node,
                            consensus_term,
                            first,
                            second,
                            a_orders[x],
                            b_orders[x],
                            c_node,
                        ],
                    )?;
                }
            }
        }
    }

    Ok(())
}

pub(crate) fn enumerate_graph(
    graph: &GraphBody,
    include_reverse_constant_folding: bool,
) -> Result<Vec<RawCandidate>, RuleError> {
    let mut out = Builder::new(graph, include_reverse_constant_folding);
    let n = graph.op.len();
    let ops = &graph.op;
    let arg0 = &graph.arg0;
    let arg1 = &graph.arg1;
    let mut signal_refs = vec![false; n];
    let mut const_signals = vec![-1; n];
    let mut not_signals = vec![-1; n];
    let mut and_pairs = vec![Pair::default(); n];
    let mut or_pairs = vec![Pair::default(); n];
    let mut false_const = -1;
    let mut true_const = -1;

    for node in 0..n {
        signal_refs[node] = ops[node] != OpCode::Output;
    }

    for node in 0..n {
        let op = ops[node];
        let left = arg0[node];
        let right = arg1[node];

        match op {
            OpCode::Const => {
                let value = i32::from(left != 0);
                const_signals[node] = value;
                if value == 0 && false_const < 0 {
                    false_const = node as i32;
                }
                if value != 0 && true_const < 0 {
                    true_const = node as i32;
                }
            }
            OpCode::Not => {
                if (left as usize) < n && signal_refs[left as usize] {
                    not_signals[node] = left as i32;
                }
            }
            OpCode::And => {
                if (left as usize) < n
                    && (right as usize) < n
                    && signal_refs[left as usize]
                    && signal_refs[right as usize]
                {
                    and_pairs[node] = Pair {
                        a: left,
                        b: right,
                        valid: true,
                    };
                }
            }
            OpCode::Or => {
                if (left as usize) < n
                    && (right as usize) < n
                    && signal_refs[left as usize]
                    && signal_refs[right as usize]
                {
                    or_pairs[node] = Pair {
                        a: left,
                        b: right,
                        valid: true,
                    };
                }
            }
            OpCode::Input | OpCode::Output => {}
        }
    }

    let tables = SignalTables {
        not_signals: &not_signals,
        and_pairs: &and_pairs,
        or_pairs: &or_pairs,
    };

    for root in 0..n as u32 {
        let op = ops[root as usize];
        let left = arg0[root as usize];
        let right = arg1[root as usize];

        if (op == OpCode::And || op == OpCode::Or)
            && (and_pairs[root as usize].valid || or_pairs[root as usize].valid)
        {
            let left_const = const_signals[left as usize];
            let right_const = const_signals[right as usize];
            let left_not = not_signals[left as usize];
            let right_not = not_signals[right as usize];
            let right_order = left != right;

            if op == OpCode::And {
                if left_const == 0 {
                    out.add(RuleId::AndFalse, root, &[left])?;
                }
                if left_const == 1 {
                    out.add(RuleId::AndTrue, root, &[left, right])?;
                }
                if right_order && right_const == 0 {
                    out.add(RuleId::AndFalse, root, &[right])?;
                }
                if right_order && right_const == 1 {
                    out.add(RuleId::AndTrue, root, &[right, left])?;
                }
                if left == right {
                    out.add(RuleId::AndIdempotent, root, &[left])?;
                }
                if right_not >= 0 && right_not as u32 == left {
                    out.add(RuleId::AndComplement, root, &[left, right])?;
                }
                if right_order && left_not >= 0 && left_not as u32 == right {
                    out.add(RuleId::AndComplement, root, &[right, left])?;
                }
                out.add(RuleId::CommuteAnd, root, &[left, right])?;

                let left_and = and_pairs[left as usize];
                let right_and = and_pairs[right as usize];
                if left_and.valid {
                    out.add(
                        RuleId::AssociateAndRight,
                        root,
                        &[left, left_and.a, left_and.b, right],
                    )?;
                }
                if right_and.valid {
                    out.add(
                        RuleId::AssociateAndLeft,
                        root,
                        &[right, left, right_and.a, right_and.b],
                    )?;
                }

                let right_or = or_pairs[right as usize];
                let left_or = or_pairs[left as usize];
                if right_or.valid {
                    if right_or.a == left {
                        out.add(RuleId::AbsorbAndOr, root, &[left, right, right_or.b])?;
                    }
                    if right_or.b == left && right_or.a != right_or.b {
                        out.add(RuleId::AbsorbAndOr, root, &[left, right, right_or.a])?;
                    }
                    out.add(
                        RuleId::DistributeOrExpand,
                        root,
                        &[right, left, right_or.a, right_or.b],
                    )?;
                }
                if right_order && left_or.valid {
                    if left_or.a == right {
                        out.add(RuleId::AbsorbAndOr, root, &[right, left, left_or.b])?;
                    }
                    if left_or.b == right && left_or.a != left_or.b {
                        out.add(RuleId::AbsorbAndOr, root, &[right, left, left_or.a])?;
                    }
                    out.add(
                        RuleId::DistributeOrExpand,
                        root,
                        &[left, right, left_or.a, left_or.b],
                    )?;
                }
                if left_not >= 0 && right_not >= 0 {
                    out.add(
                        RuleId::DeMorganAndNots,
                        root,
                        &[left, right, left_not as u32, right_not as u32],
                    )?;
                }
                if left_or.valid && right_or.valid {
                    for (la, lb) in [(left_or.a, left_or.b), (left_or.b, left_or.a)] {
                        if la == left_or.b && left_or.a == left_or.b {
                            continue;
                        }
                        for (ra, rc) in [(right_or.a, right_or.b), (right_or.b, right_or.a)] {
                            if ra == right_or.b && right_or.a == right_or.b {
                                continue;
                            }
                            if la == ra {
                                out.add(
                                    RuleId::DistributeAndFactor,
                                    root,
                                    &[left, right, la, lb, rc],
                                )?;
                            }
                        }
                    }
                }
                consensus_add_bindings(
                    &mut out,
                    root,
                    left,
                    right,
                    ConsensusAdd {
                        term_op: OpCode::Or,
                        rule: RuleId::ConsensusAndAdd,
                    },
                    tables,
                )?;
                if right_order {
                    consensus_add_bindings(
                        &mut out,
                        root,
                        right,
                        left,
                        ConsensusAdd {
                            term_op: OpCode::Or,
                            rule: RuleId::ConsensusAndAdd,
                        },
                        tables,
                    )?;
                }
                consensus_remove_bindings(
                    &mut out,
                    root,
                    left,
                    right,
                    ConsensusRemove {
                        inner_op: OpCode::And,
                        consensus_op: OpCode::Or,
                        rule: RuleId::ConsensusAndRemove,
                    },
                    tables,
                )?;
                if right_order {
                    consensus_remove_bindings(
                        &mut out,
                        root,
                        right,
                        left,
                        ConsensusRemove {
                            inner_op: OpCode::And,
                            consensus_op: OpCode::Or,
                            rule: RuleId::ConsensusAndRemove,
                        },
                        tables,
                    )?;
                }
            } else {
                if left_const == 0 {
                    out.add(RuleId::OrFalse, root, &[left, right])?;
                }
                if left_const == 1 {
                    out.add(RuleId::OrTrue, root, &[left])?;
                }
                if right_order && right_const == 0 {
                    out.add(RuleId::OrFalse, root, &[right, left])?;
                }
                if right_order && right_const == 1 {
                    out.add(RuleId::OrTrue, root, &[right])?;
                }
                if left == right {
                    out.add(RuleId::OrIdempotent, root, &[left])?;
                }
                if right_not >= 0 && right_not as u32 == left {
                    out.add(RuleId::OrComplement, root, &[left, right])?;
                }
                if right_order && left_not >= 0 && left_not as u32 == right {
                    out.add(RuleId::OrComplement, root, &[right, left])?;
                }
                out.add(RuleId::CommuteOr, root, &[left, right])?;

                let left_or = or_pairs[left as usize];
                let right_or = or_pairs[right as usize];
                if left_or.valid {
                    out.add(
                        RuleId::AssociateOrRight,
                        root,
                        &[left, left_or.a, left_or.b, right],
                    )?;
                }
                if right_or.valid {
                    out.add(
                        RuleId::AssociateOrLeft,
                        root,
                        &[right, left, right_or.a, right_or.b],
                    )?;
                }

                let right_and = and_pairs[right as usize];
                let left_and = and_pairs[left as usize];
                if right_and.valid {
                    if right_and.a == left {
                        out.add(RuleId::AbsorbOrAnd, root, &[left, right, right_and.b])?;
                    }
                    if right_and.b == left && right_and.a != right_and.b {
                        out.add(RuleId::AbsorbOrAnd, root, &[left, right, right_and.a])?;
                    }
                    out.add(
                        RuleId::DistributeAndExpand,
                        root,
                        &[right, left, right_and.a, right_and.b],
                    )?;
                }
                if right_order && left_and.valid {
                    if left_and.a == right {
                        out.add(RuleId::AbsorbOrAnd, root, &[right, left, left_and.b])?;
                    }
                    if left_and.b == right && left_and.a != left_and.b {
                        out.add(RuleId::AbsorbOrAnd, root, &[right, left, left_and.a])?;
                    }
                    out.add(
                        RuleId::DistributeAndExpand,
                        root,
                        &[left, right, left_and.a, left_and.b],
                    )?;
                }
                if left_not >= 0 && right_not >= 0 {
                    out.add(
                        RuleId::DeMorganOrNots,
                        root,
                        &[left, right, left_not as u32, right_not as u32],
                    )?;
                }
                if left_and.valid && right_and.valid {
                    for (la, lb) in [(left_and.a, left_and.b), (left_and.b, left_and.a)] {
                        if la == left_and.b && left_and.a == left_and.b {
                            continue;
                        }
                        for (ra, rc) in [(right_and.a, right_and.b), (right_and.b, right_and.a)] {
                            if ra == right_and.b && right_and.a == right_and.b {
                                continue;
                            }
                            if la == ra {
                                out.add(
                                    RuleId::DistributeOrFactor,
                                    root,
                                    &[left, right, la, lb, rc],
                                )?;
                            }
                        }
                    }
                }
                consensus_add_bindings(
                    &mut out,
                    root,
                    left,
                    right,
                    ConsensusAdd {
                        term_op: OpCode::And,
                        rule: RuleId::ConsensusOrAdd,
                    },
                    tables,
                )?;
                if right_order {
                    consensus_add_bindings(
                        &mut out,
                        root,
                        right,
                        left,
                        ConsensusAdd {
                            term_op: OpCode::And,
                            rule: RuleId::ConsensusOrAdd,
                        },
                        tables,
                    )?;
                }
                consensus_remove_bindings(
                    &mut out,
                    root,
                    left,
                    right,
                    ConsensusRemove {
                        inner_op: OpCode::Or,
                        consensus_op: OpCode::And,
                        rule: RuleId::ConsensusOrRemove,
                    },
                    tables,
                )?;
                if right_order {
                    consensus_remove_bindings(
                        &mut out,
                        root,
                        right,
                        left,
                        ConsensusRemove {
                            inner_op: OpCode::Or,
                            consensus_op: OpCode::And,
                            rule: RuleId::ConsensusOrRemove,
                        },
                        tables,
                    )?;
                }
            }
        } else if op == OpCode::Not && not_signals[root as usize] >= 0 {
            let left = arg0[root as usize];
            let value = const_signals[left as usize];
            if value == 0 {
                out.add(RuleId::NotFalse, root, &[left])?;
            }
            if value == 1 {
                out.add(RuleId::NotTrue, root, &[left])?;
            }
            let inner = not_signals[left as usize];
            if inner >= 0 {
                out.add(RuleId::DoubleNegation, root, &[left, inner as u32])?;
            }
            let and_pair = and_pairs[left as usize];
            let or_pair = or_pairs[left as usize];
            if and_pair.valid {
                out.add(
                    RuleId::DeMorganNotAnd,
                    root,
                    &[left, and_pair.a, and_pair.b],
                )?;
            }
            if or_pair.valid {
                out.add(RuleId::DeMorganNotOr, root, &[left, or_pair.a, or_pair.b])?;
            }
        }
    }

    for root in 0..n as u32 {
        let op = ops[root as usize];
        if op == OpCode::Output {
            continue;
        }

        out.add(RuleId::AndIdempotentInverse, root, &[])?;
        out.add(RuleId::OrIdempotentInverse, root, &[])?;
        out.add(RuleId::DoubleNegationInverse, root, &[])?;
        out.add(RuleId::AbsorbOrAndInverse, root, &[root])?;
        out.add(RuleId::AbsorbAndOrInverse, root, &[root])?;

        if true_const >= 0 {
            out.add(RuleId::AndTrueInverse, root, &[true_const as u32])?;
        }
        if false_const >= 0 {
            out.add(RuleId::OrFalseInverse, root, &[false_const as u32])?;
        }

        if op == OpCode::Const {
            let value = i32::from(arg0[root as usize] != 0);
            if value == 0 {
                out.add(RuleId::AndFalseInverse, root, &[root])?;
                if true_const >= 0 {
                    out.add(RuleId::NotTrueInverse, root, &[true_const as u32])?;
                }
            } else {
                out.add(RuleId::OrTrueInverse, root, &[root])?;
                if false_const >= 0 {
                    out.add(RuleId::NotFalseInverse, root, &[false_const as u32])?;
                }
            }

            for witness in 0..n as u32 {
                let inner = not_signals[witness as usize];
                if inner < 0 {
                    continue;
                }
                if value == 0 {
                    out.add(RuleId::AndComplementInverse, root, &[inner as u32, witness])?;
                } else {
                    out.add(RuleId::OrComplementInverse, root, &[inner as u32, witness])?;
                }
            }
        }
    }

    Ok(out.items)
}

pub(crate) fn apply_graph(
    graph: &GraphBody,
    candidate: RawCandidate,
) -> Result<GraphBody, RuleError> {
    if candidate.root as usize >= graph.op.len() {
        return Err(RuleError::InvalidCandidate("candidate root out of range"));
    }
    for node in candidate.matched_slice() {
        if *node as usize >= graph.op.len() {
            return Err(RuleError::InvalidCandidate(
                "candidate matched node out of range",
            ));
        }
    }

    let mut out = graph.clone();
    let old_n = out.op.len();
    let replacement = build_replacement(candidate, &mut out.op, &mut out.arg0, &mut out.arg1)?;

    if out.op.len() > usize::from(out.capacity) {
        return Err(RuleError::CapacityExceeded);
    }

    for node in 0..old_n {
        let code = out.op[node];
        if matches!(
            code,
            OpCode::And | OpCode::Or | OpCode::Not | OpCode::Output
        ) && out.arg0[node] == candidate.root
        {
            out.arg0[node] = replacement;
        }
        if matches!(code, OpCode::And | OpCode::Or) && out.arg1[node] == candidate.root {
            out.arg1[node] = replacement;
        }
    }

    compact_graph(&out).map_err(RuleError::Graph)
}

fn build_replacement(
    candidate: RawCandidate,
    op: &mut Vec<OpCode>,
    arg0: &mut Vec<u32>,
    arg1: &mut Vec<u32>,
) -> Result<u32, RuleError> {
    let root = candidate.root;
    let m = candidate.matched;

    Ok(match RuleId::from_u16(candidate.rule_id)? {
        RuleId::AndFalse | RuleId::OrTrue => m[1],
        RuleId::AndTrue | RuleId::OrFalse => m[2],
        RuleId::AndIdempotent
        | RuleId::OrIdempotent
        | RuleId::AbsorbOrAnd
        | RuleId::AbsorbAndOr
        | RuleId::ConsensusOrRemove
        | RuleId::ConsensusAndRemove => m[1],
        RuleId::DoubleNegation => m[2],
        RuleId::AndFalseInverse => append_node(op, arg0, arg1, OpCode::And, m[1], root),
        RuleId::AndTrueInverse => append_node(op, arg0, arg1, OpCode::And, root, m[1]),
        RuleId::OrFalseInverse => append_node(op, arg0, arg1, OpCode::Or, root, m[1]),
        RuleId::OrTrueInverse => append_node(op, arg0, arg1, OpCode::Or, m[1], root),
        RuleId::NotFalseInverse | RuleId::NotTrueInverse => {
            append_node(op, arg0, arg1, OpCode::Not, m[1], NO_NODE)
        }
        RuleId::AndIdempotentInverse => append_node(op, arg0, arg1, OpCode::And, root, root),
        RuleId::OrIdempotentInverse => append_node(op, arg0, arg1, OpCode::Or, root, root),
        RuleId::AndComplementInverse => append_node(op, arg0, arg1, OpCode::And, m[1], m[2]),
        RuleId::OrComplementInverse => append_node(op, arg0, arg1, OpCode::Or, m[1], m[2]),
        RuleId::DoubleNegationInverse => {
            let first = append_node(op, arg0, arg1, OpCode::Not, root, NO_NODE);
            append_node(op, arg0, arg1, OpCode::Not, first, NO_NODE)
        }
        RuleId::AbsorbOrAndInverse => {
            let first = append_node(op, arg0, arg1, OpCode::And, root, m[1]);
            append_node(op, arg0, arg1, OpCode::Or, root, first)
        }
        RuleId::AbsorbAndOrInverse => {
            let first = append_node(op, arg0, arg1, OpCode::Or, root, m[1]);
            append_node(op, arg0, arg1, OpCode::And, root, first)
        }
        RuleId::NotFalse | RuleId::OrComplement => {
            append_node(op, arg0, arg1, OpCode::Const, 1, NO_NODE)
        }
        RuleId::NotTrue | RuleId::AndComplement => {
            append_node(op, arg0, arg1, OpCode::Const, 0, NO_NODE)
        }
        RuleId::CommuteAnd => append_node(op, arg0, arg1, OpCode::And, m[2], m[1]),
        RuleId::CommuteOr => append_node(op, arg0, arg1, OpCode::Or, m[2], m[1]),
        RuleId::AssociateAndLeft | RuleId::AssociateOrLeft => {
            let code = if candidate.rule_id == RuleId::AssociateAndLeft.as_u16() {
                OpCode::And
            } else {
                OpCode::Or
            };
            let first = append_node(op, arg0, arg1, code, m[2], m[3]);
            append_node(op, arg0, arg1, code, first, m[4])
        }
        RuleId::AssociateAndRight | RuleId::AssociateOrRight => {
            let code = if candidate.rule_id == RuleId::AssociateAndRight.as_u16() {
                OpCode::And
            } else {
                OpCode::Or
            };
            let first = append_node(op, arg0, arg1, code, m[3], m[4]);
            append_node(op, arg0, arg1, code, m[2], first)
        }
        RuleId::DeMorganNotAnd | RuleId::DeMorganNotOr => {
            let na = append_node(op, arg0, arg1, OpCode::Not, m[2], NO_NODE);
            let nb = append_node(op, arg0, arg1, OpCode::Not, m[3], NO_NODE);
            let code = if candidate.rule_id == RuleId::DeMorganNotAnd.as_u16() {
                OpCode::Or
            } else {
                OpCode::And
            };
            append_node(op, arg0, arg1, code, na, nb)
        }
        RuleId::DeMorganOrNots | RuleId::DeMorganAndNots => {
            let code = if candidate.rule_id == RuleId::DeMorganOrNots.as_u16() {
                OpCode::And
            } else {
                OpCode::Or
            };
            let first = append_node(op, arg0, arg1, code, m[3], m[4]);
            append_node(op, arg0, arg1, OpCode::Not, first, NO_NODE)
        }
        RuleId::DistributeOrFactor | RuleId::DistributeAndFactor => {
            let first_code = if candidate.rule_id == RuleId::DistributeOrFactor.as_u16() {
                OpCode::Or
            } else {
                OpCode::And
            };
            let second_code = if candidate.rule_id == RuleId::DistributeOrFactor.as_u16() {
                OpCode::And
            } else {
                OpCode::Or
            };
            let first = append_node(op, arg0, arg1, first_code, m[4], m[5]);
            append_node(op, arg0, arg1, second_code, m[3], first)
        }
        RuleId::DistributeOrExpand | RuleId::DistributeAndExpand => {
            let inner = if candidate.rule_id == RuleId::DistributeOrExpand.as_u16() {
                OpCode::And
            } else {
                OpCode::Or
            };
            let outer = if candidate.rule_id == RuleId::DistributeOrExpand.as_u16() {
                OpCode::Or
            } else {
                OpCode::And
            };
            let first = append_node(op, arg0, arg1, inner, m[2], m[3]);
            let second = append_node(op, arg0, arg1, inner, m[2], m[4]);
            append_node(op, arg0, arg1, outer, first, second)
        }
        RuleId::ConsensusOrAdd | RuleId::ConsensusAndAdd => {
            let first_code = if candidate.rule_id == RuleId::ConsensusOrAdd.as_u16() {
                OpCode::And
            } else {
                OpCode::Or
            };
            let second_code = if candidate.rule_id == RuleId::ConsensusOrAdd.as_u16() {
                OpCode::Or
            } else {
                OpCode::And
            };
            let first = append_node(op, arg0, arg1, first_code, m[4], m[5]);
            append_node(op, arg0, arg1, second_code, root, first)
        }
    })
}

fn append_node(
    op: &mut Vec<OpCode>,
    arg0: &mut Vec<u32>,
    arg1: &mut Vec<u32>,
    code: OpCode,
    a: u32,
    b: u32,
) -> u32 {
    op.push(code);
    arg0.push(a);
    arg1.push(b);
    (op.len() - 1) as u32
}

pub(crate) const fn reverse_constant_folding_rule(rule: RuleId) -> bool {
    matches!(
        rule,
        RuleId::AndFalseInverse
            | RuleId::AndTrueInverse
            | RuleId::OrFalseInverse
            | RuleId::OrTrueInverse
            | RuleId::NotFalseInverse
            | RuleId::NotTrueInverse
    )
}

pub(crate) fn inverse_rule_id(rule_id: u16) -> Option<u16> {
    use RuleId::*;

    let rule = RuleId::from_u16(rule_id).ok()?;
    let inverse = match rule {
        AndFalse => AndFalseInverse,
        AndTrue => AndTrueInverse,
        OrFalse => OrFalseInverse,
        OrTrue => OrTrueInverse,
        NotFalse => NotFalseInverse,
        NotTrue => NotTrueInverse,
        AndIdempotent => AndIdempotentInverse,
        OrIdempotent => OrIdempotentInverse,
        AndComplement => AndComplementInverse,
        OrComplement => OrComplementInverse,
        DoubleNegation => DoubleNegationInverse,
        CommuteAnd => CommuteAnd,
        CommuteOr => CommuteOr,
        AssociateAndLeft => AssociateAndRight,
        AssociateAndRight => AssociateAndLeft,
        AssociateOrLeft => AssociateOrRight,
        AssociateOrRight => AssociateOrLeft,
        AbsorbOrAnd => AbsorbOrAndInverse,
        AbsorbAndOr => AbsorbAndOrInverse,
        DeMorganNotAnd => DeMorganOrNots,
        DeMorganNotOr => DeMorganAndNots,
        DeMorganOrNots => DeMorganNotAnd,
        DeMorganAndNots => DeMorganNotOr,
        DistributeOrFactor => DistributeOrExpand,
        DistributeOrExpand => DistributeOrFactor,
        DistributeAndFactor => DistributeAndExpand,
        DistributeAndExpand => DistributeAndFactor,
        ConsensusOrAdd => ConsensusOrRemove,
        ConsensusOrRemove => ConsensusOrAdd,
        ConsensusAndAdd => ConsensusAndRemove,
        ConsensusAndRemove => ConsensusAndAdd,
        AndFalseInverse => AndFalse,
        AndTrueInverse => AndTrue,
        OrFalseInverse => OrFalse,
        OrTrueInverse => OrTrue,
        NotFalseInverse => NotFalse,
        NotTrueInverse => NotTrue,
        AndIdempotentInverse => AndIdempotent,
        OrIdempotentInverse => OrIdempotent,
        AndComplementInverse => AndComplement,
        OrComplementInverse => OrComplement,
        DoubleNegationInverse => DoubleNegation,
        AbsorbOrAndInverse => AbsorbOrAnd,
        AbsorbAndOrInverse => AbsorbAndOr,
    };
    Some(inverse.as_u16())
}

pub(crate) fn category_weight(rule_id: u16) -> f64 {
    use RuleId::*;

    match RuleId::from_u16(rule_id) {
        Ok(CommuteAnd | CommuteOr) => 0.5,
        Ok(AssociateAndLeft | AssociateAndRight | AssociateOrLeft | AssociateOrRight) => 0.5,
        Ok(ConsensusOrAdd | ConsensusOrRemove | ConsensusAndAdd | ConsensusAndRemove) => 0.5,
        _ => 1.0,
    }
}

pub(crate) const fn replacement_extra_nodes(rule: RuleId) -> i32 {
    match rule {
        RuleId::AndFalse
        | RuleId::AndTrue
        | RuleId::OrFalse
        | RuleId::OrTrue
        | RuleId::AndIdempotent
        | RuleId::OrIdempotent
        | RuleId::AbsorbOrAnd
        | RuleId::AbsorbAndOr
        | RuleId::ConsensusOrRemove
        | RuleId::ConsensusAndRemove
        | RuleId::DoubleNegation => 0,
        RuleId::AndFalseInverse
        | RuleId::AndTrueInverse
        | RuleId::OrFalseInverse
        | RuleId::OrTrueInverse
        | RuleId::NotFalseInverse
        | RuleId::NotTrueInverse
        | RuleId::AndIdempotentInverse
        | RuleId::OrIdempotentInverse
        | RuleId::AndComplementInverse
        | RuleId::OrComplementInverse
        | RuleId::NotFalse
        | RuleId::NotTrue
        | RuleId::AndComplement
        | RuleId::OrComplement
        | RuleId::CommuteAnd
        | RuleId::CommuteOr => 1,
        RuleId::DoubleNegationInverse
        | RuleId::AbsorbOrAndInverse
        | RuleId::AbsorbAndOrInverse
        | RuleId::AssociateAndLeft
        | RuleId::AssociateAndRight
        | RuleId::AssociateOrLeft
        | RuleId::AssociateOrRight
        | RuleId::DeMorganOrNots
        | RuleId::DeMorganAndNots
        | RuleId::DistributeOrFactor
        | RuleId::DistributeAndFactor
        | RuleId::ConsensusOrAdd
        | RuleId::ConsensusAndAdd => 2,
        _ => 3,
    }
}

#[derive(Debug)]
pub(crate) enum RuleError {
    UnknownRule(u16),
    BadMatchLength,
    InvalidCandidate(&'static str),
    CapacityExceeded,
    Graph(crate::graph::GraphError),
}

impl fmt::Display for RuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownRule(rule) => write!(f, "unknown rule {rule}"),
            Self::BadMatchLength => f.write_str("bad matched-node length"),
            Self::InvalidCandidate(message) => f.write_str(message),
            Self::CapacityExceeded => f.write_str("rewrite exceeds graph capacity"),
            Self::Graph(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for RuleError {}

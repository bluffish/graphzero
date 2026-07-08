use super::tree::Node;

pub fn considered_visit_sequence(max_considered: usize, simulations: usize) -> Vec<u32> {
    if max_considered <= 1 {
        return (0..simulations as u32).collect();
    }

    let log2max = (max_considered as f64).log2().ceil() as usize;
    let mut sequence = Vec::with_capacity(simulations);
    let mut visits = vec![0_u32; max_considered];
    let mut considered = max_considered;

    while sequence.len() < simulations {
        let extra = (simulations / (log2max * considered)).max(1);
        for _ in 0..extra {
            sequence.extend_from_slice(&visits[..considered]);
            for visit in &mut visits[..considered] {
                *visit += 1;
            }
        }
        considered = (considered / 2).max(2);
    }

    sequence.truncate(simulations);
    sequence
}

pub(super) fn considered_actions(base_scores: &[f32], max_considered: usize) -> Vec<usize> {
    let mut actions = (0..base_scores.len()).collect::<Vec<_>>();
    actions.sort_by(|&left, &right| {
        base_scores[right]
            .total_cmp(&base_scores[left])
            .then_with(|| left.cmp(&right))
    });
    actions.truncate(max_considered.min(actions.len()));
    actions
}

pub(super) fn best_eligible<G, C>(
    node: &Node<G, C>,
    considered: &[usize],
    target_visits: u32,
    scores: &[f32],
    tree_reuse: bool,
) -> Option<usize> {
    considered
        .iter()
        .copied()
        .filter(|&action| {
            node.logits[action].is_finite()
                && if tree_reuse {
                    node.visits[action] <= target_visits
                } else {
                    node.visits[action] == target_visits
                }
        })
        .max_by(|&left, &right| {
            scores[left]
                .total_cmp(&scores[right])
                .then_with(|| right.cmp(&left))
        })
}

pub(super) fn selectable_root_actions<G, C>(node: &Node<G, C>, considered: &[usize]) -> Vec<usize> {
    let mut actions = considered
        .iter()
        .copied()
        .filter(|&action| node.logits[action].is_finite())
        .collect::<Vec<_>>();

    if actions.is_empty() {
        actions.extend(
            node.logits
                .iter()
                .enumerate()
                .filter_map(|(action, logit)| logit.is_finite().then_some(action)),
        );
    }

    actions
}

pub(super) fn best_count_action(visits: &[u32], considered: &[usize], scores: &[f32]) -> usize {
    considered
        .iter()
        .copied()
        .max_by(|&left, &right| {
            visits[left]
                .cmp(&visits[right])
                .then_with(|| scores[left].total_cmp(&scores[right]))
                .then_with(|| right.cmp(&left))
        })
        .expect("considered actions is non-empty")
}

pub(super) fn completed_q<G, C>(node: &Node<G, C>) -> Vec<f32> {
    let mixed = mixed_value(node);
    node.visits
        .iter()
        .zip(&node.q)
        .map(|(visits, q)| if *visits > 0 { *q } else { mixed })
        .collect()
}

pub(super) fn mixed_value<G, C>(node: &Node<G, C>) -> f32 {
    let visits = node.visits.iter().copied().sum::<u32>();
    if visits == 0 {
        return node.value;
    }

    let mut prior_mass = 0.0;
    let mut weighted = 0.0;

    for ((visits, prior), q) in node.visits.iter().zip(&node.priors).zip(&node.q) {
        if *visits == 0 {
            continue;
        }
        prior_mass += prior;
        weighted += prior * q;
    }

    if prior_mass <= 0.0 {
        return node.value;
    }

    (node.value + visits as f32 * weighted / prior_mass) / (1.0 + visits as f32)
}

pub(super) fn search_value<G, C>(node: &Node<G, C>) -> f32 {
    let mut visits = 0;
    let mut value = 0.0;

    for (count, q) in node.visits.iter().zip(&node.q) {
        if *count == 0 {
            continue;
        }
        visits += *count;
        value += *count as f32 * *q;
    }

    if visits == 0 {
        node.value
    } else {
        value / visits as f32
    }
}

pub(super) fn root_q_max<G, C>(node: &Node<G, C>) -> f32 {
    node.visits
        .iter()
        .zip(&node.q)
        .filter_map(|(visits, q)| (*visits > 0).then_some(*q))
        .reduce(f32::max)
        .unwrap_or(node.value)
}

pub(super) fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .reduce(f32::max);
    let Some(max) = max else {
        return vec![1.0 / values.len() as f32; values.len()];
    };

    let mut out = Vec::with_capacity(values.len());
    let mut total = 0.0;

    for value in values {
        let next = if value.is_finite() {
            (*value - max).exp()
        } else {
            0.0
        };
        total += next;
        out.push(next);
    }

    if total <= 0.0 || !total.is_finite() {
        let legal = values.iter().filter(|value| value.is_finite()).count();
        let uniform = 1.0 / legal.max(1) as f32;
        for (out, value) in out.iter_mut().zip(values) {
            *out = if value.is_finite() { uniform } else { 0.0 };
        }
        return out;
    }

    for value in &mut out {
        *value /= total;
    }

    out
}

/// The Gumbel scale at which a noisy root argmax lands in the prior's
/// top-m actions with probability `overlap + 0.05` (whittlezero's
/// gumbel_noise_overlap). argmax(logits + s*Gumbel) distributes as
/// softmax(logits/s), so the top-m mass is monotone decreasing in s and
/// an 18-step bisection over [1e-3, 64] pins the target. Masked actions
/// (-inf logits) are excluded; when m covers every legal action the base
/// scale is returned unchanged.
pub(super) fn overlap_noise_scale(
    logits: &[f32],
    considered: usize,
    overlap: f32,
    base_scale: f32,
) -> f32 {
    let mut legal: Vec<f32> = logits.iter().copied().filter(|l| l.is_finite()).collect();
    if legal.len() <= considered {
        return base_scale;
    }
    legal.sort_unstable_by(|a, b| b.total_cmp(a));

    let floor = considered as f32 / legal.len() as f32 + 1e-6;
    let target = (overlap + 0.05).clamp(floor, 0.999_999);
    let (mut lo, mut hi) = (1e-3_f32, 64.0_f32);
    for _ in 0..18 {
        let mid = 0.5 * (lo + hi);
        if top_mass(&legal, considered, mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn top_mass(sorted_desc: &[f32], m: usize, scale: f32) -> f32 {
    let max = sorted_desc[0];
    let mut top = 0.0;
    let mut total = 0.0;
    for (index, logit) in sorted_desc.iter().enumerate() {
        let weight = ((logit - max) / scale).exp();
        total += weight;
        if index < m {
            top += weight;
        }
    }
    top / total
}

pub(super) fn sample_root_gumbels(count: usize, scale: f32, rng: &mut GumbelRng) -> Vec<f32> {
    if scale == 0.0 {
        return vec![0.0; count];
    }

    (0..count)
        .map(|_| scale * -(-rng.unit().ln()).ln())
        .collect()
}

pub(super) fn sample_count_action(
    rng: &mut GumbelRng,
    visits: &[u32],
    allowed: &[usize],
    temperature: f32,
    fallback: usize,
) -> usize {
    if temperature <= 0.0 {
        return fallback;
    }

    let inv_temp = 1.0 / temperature;
    let mut total = 0.0;
    let mut weights = vec![0.0; visits.len()];

    for &action in allowed {
        let count = visits[action];
        let weight = if count == 0 {
            0.0
        } else {
            (count as f32).powf(inv_temp)
        };
        total += weight;
        weights[action] = weight;
    }

    if total <= 0.0 || !total.is_finite() {
        return fallback;
    }

    let mut threshold = rng.unit() * total;
    for (index, weight) in weights.into_iter().enumerate() {
        if threshold <= weight {
            return index;
        }
        threshold -= weight;
    }

    fallback
}

pub(super) fn budget_fraction(max_steps: usize, step: usize) -> f32 {
    if max_steps == 0 {
        1.0
    } else {
        max_steps.saturating_sub(step) as f32 / max_steps as f32
    }
}

pub(super) fn root_seed(seed: u64, root_step: u32) -> u64 {
    seed ^ 0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(u64::from(root_step) + 1)
}

pub(super) struct GumbelRng {
    state: u64,
}

impl GumbelRng {
    const STEP: u64 = 0x9e37_79b9_7f4a_7c15;

    pub(super) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn unit(&mut self) -> f32 {
        let value = self.next_u64() >> 40;
        let unit = (value as f32 + 0.5) / (1_u32 << 24) as f32;
        unit.clamp(1.0e-7, 1.0 - 1.0e-7)
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(Self::STEP);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::{overlap_noise_scale, top_mass};

    #[test]
    fn overlap_scale_shrinks_for_flat_priors_and_grows_for_sharp_ones() {
        // Flat priors: top-m mass is m/n at every scale, below the target,
        // so the bisection settles at the minimum -- no noise can help.
        let flat = vec![0.0; 100];
        assert!(overlap_noise_scale(&flat, 8, 0.5, 1.0) < 0.01);

        // One dominant logit: the solution of (1 + 7x)/(1 + 99x) = 0.55
        // with x = exp(-20/s) gives s ~ 4.3.
        let mut sharp = vec![0.0; 100];
        sharp[0] = 20.0;
        let scale = overlap_noise_scale(&sharp, 8, 0.5, 1.0);
        assert!((2.0..8.0).contains(&scale), "scale {scale}");

        let mut sorted = sharp.clone();
        sorted.sort_unstable_by(|a, b| b.total_cmp(a));
        let mass = top_mass(&sorted, 8, scale);
        assert!((mass - 0.55).abs() < 0.01, "mass {mass}");
    }

    #[test]
    fn overlap_scale_keeps_base_when_everything_is_considered() {
        assert_eq!(overlap_noise_scale(&[1.0, 2.0], 8, 0.5, 0.7), 0.7);
        // Masked (-inf) actions do not count toward the legal set.
        let mut masked = vec![f32::NEG_INFINITY; 10];
        masked[0] = 1.0;
        masked[1] = 0.0;
        assert_eq!(overlap_noise_scale(&masked, 8, 0.5, 0.7), 0.7);
    }
}

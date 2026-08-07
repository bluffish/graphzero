from __future__ import annotations

import importlib
import math
from functools import lru_cache
from typing import TYPE_CHECKING

from gz.codec import FeatureSchemaConfig
from gz.model.tensors import GraphBatchTensors, GraphStateTensors

if TYPE_CHECKING:
    from gz.model.exphormer import ArchConfig


@lru_cache(maxsize=1)
def model_class():
    torch = _torch()
    nn = torch.nn

    class GraphModel(nn.Module):
        def __init__(self, schema: FeatureSchemaConfig, arch: ArchConfig) -> None:
            super().__init__()
            self.schema = schema
            self.arch = arch
            self.node_embedding = nn.Embedding(
                schema.node_vocab_size,
                arch.dim,
                padding_idx=0,
            )
            self.attr_proj = (
                nn.Linear(schema.node_attr_dim, arch.dim, bias=False)
                if schema.node_attr_dim
                else None
            )
            self.position_proj = nn.Linear(4, arch.dim)
            self.board_role_embedding = nn.Embedding(2, arch.dim)
            self.global_tokens = nn.Parameter(torch.zeros(1, arch.dim))
            self.layers = nn.ModuleList(
                [GraphLayer(schema, arch) for _ in range(arch.layers)]
            )
            self.node_output_norm = nn.LayerNorm(arch.dim)
            self.global_output_norm = nn.LayerNorm(arch.dim)
            self.kind_embedding = nn.Embedding(
                schema.action_kind_vocab_size,
                arch.dim,
                padding_idx=0,
            )
            policy_class = (
                FactorizedPointerPolicyHead
                if arch.policy_head == "factorized-pointer"
                else PointerPolicyHead
            )
            self.policy = policy_class(
                arch,
                _action_feat_dim(arch),
                schema.action_kind_vocab_size,
            )
            self.value = _mlp(
                nn,
                arch.dim,
                arch.ffn_dim,
                1,
                arch.activation,
                arch.dropout,
            )
            value_dim = arch.dim
            if arch.auxiliary_heads in {
                "v8-v32-score",
                "v8-v32-score-soft-policy",
                "v8-v32-score-soft-policy-v2",
            }:
                auxiliary_hidden = arch.value_hidden or arch.ffn_dim
                self.horizon_value = _mlp(
                    nn,
                    value_dim,
                    auxiliary_hidden,
                    2,
                    arch.activation,
                    arch.dropout,
                )
                self.terminal_score = _mlp(
                    nn,
                    value_dim,
                    auxiliary_hidden,
                    1,
                    arch.activation,
                    arch.dropout,
                )
            else:
                self.horizon_value = None
                self.terminal_score = None
            self.soft_policy_kind_embedding = None
            self.soft_policy = None
            if arch.auxiliary_heads == "v8-v32-score-soft-policy":
                # Checkpoint compatibility for the retired shared-head
                # experiment. Training configuration rejects this layout.
                self.policy.add_legacy_soft_policy_readout()
            elif arch.auxiliary_heads == "v8-v32-score-soft-policy-v2":
                # Initialize the auxiliary head after every pre-existing module
                # so a fixed model seed preserves their historical initialization.
                # It shares only encoded graph tensors with the serving policy.
                self.soft_policy_kind_embedding = nn.Embedding(
                    schema.action_kind_vocab_size,
                    arch.dim,
                    padding_idx=0,
                )
                self.soft_policy = policy_class(
                    arch,
                    _action_feat_dim(arch),
                    schema.action_kind_vocab_size,
                )

        def forward(
            self,
            batch: GraphBatchTensors,
            value_trunk_grad_scale: float = 1.0,
        ):
            graph, node_roles = self._model_graph(batch)
            h, g_readout, node_mask = self._encode_graph(graph, node_roles)
            return (
                self._value_output(g_readout, value_trunk_grad_scale),
                self._policy_logits(batch, h, g_readout, node_mask),
            )

        def policy_logits(self, batch: GraphBatchTensors):
            graph, node_roles = self._model_graph(batch)
            h, g_readout, node_mask = self._encode_graph(graph, node_roles)
            return self._policy_logits(batch, h, g_readout, node_mask)

        def value_only(
            self,
            batch: GraphBatchTensors,
            value_trunk_grad_scale: float = 1.0,
        ):
            graph, node_roles = self._model_graph(batch)
            _, g_readout, _ = self._encode_graph(graph, node_roles)
            return self._value_output(g_readout, value_trunk_grad_scale)

        def training_forward(
            self,
            batch: GraphBatchTensors,
            value_trunk_grad_scale: float = 1.0,
        ):
            graph, node_roles = self._model_graph(batch)
            h, g_readout, node_mask = self._encode_graph(graph, node_roles)
            value_raw, horizon_raw, score_raw = self._training_value_outputs(
                g_readout,
                value_trunk_grad_scale,
            )
            logits = self._policy_logits(batch, h, g_readout, node_mask)
            return value_raw, horizon_raw, score_raw, logits, g_readout

        def training_forward_with_soft_policy(
            self,
            batch: GraphBatchTensors,
            value_trunk_grad_scale: float = 1.0,
            soft_policy_trunk_grad_scale: float = 1.0,
        ):
            graph, node_roles = self._model_graph(batch)
            h, g_readout, node_mask = self._encode_graph(graph, node_roles)
            value_raw, horizon_raw, score_raw = self._training_value_outputs(
                g_readout,
                value_trunk_grad_scale,
            )
            logits, soft_policy_logits = self._training_policy_logits(
                batch,
                h,
                g_readout,
                node_mask,
                soft_policy_trunk_grad_scale,
            )
            return (
                value_raw,
                horizon_raw,
                score_raw,
                logits,
                soft_policy_logits,
                g_readout,
            )

        def training_policy_logits(
            self,
            batch: GraphBatchTensors,
            soft_policy_trunk_grad_scale: float = 1.0,
        ):
            graph, node_roles = self._model_graph(batch)
            h, g_readout, node_mask = self._encode_graph(graph, node_roles)
            return self._training_policy_logits(
                batch,
                h,
                g_readout,
                node_mask,
                soft_policy_trunk_grad_scale,
            )

        def training_values(
            self,
            batch: GraphBatchTensors,
            value_trunk_grad_scale: float = 1.0,
        ):
            graph, node_roles = self._model_graph(batch)
            _, g_readout, _ = self._encode_graph(graph, node_roles)
            return (
                *self._training_value_outputs(g_readout, value_trunk_grad_scale),
                g_readout,
            )

        def _model_graph(self, batch: GraphBatchTensors):
            return _joint_board_graph(torch, batch)

        def _policy_logits(self, batch, h, g_readout, node_mask):
            action_feat, action_mask = self._policy_inputs(
                batch,
                h,
                g_readout,
                node_mask,
                self.kind_embedding,
            )
            return self.policy(
                g_readout,
                action_feat,
                batch.action_kind,
                action_mask,
            )

        def _training_policy_logits(
            self,
            batch,
            h,
            g_readout,
            node_mask,
            soft_policy_trunk_grad_scale,
        ):
            if (
                self.soft_policy is None
                or self.soft_policy_kind_embedding is None
            ):
                raise ValueError("soft-policy training requires an independent head")
            action_feat, action_mask = self._policy_inputs(
                batch,
                h,
                g_readout,
                node_mask,
                self.kind_embedding,
            )
            logits = self.policy(
                g_readout,
                action_feat,
                batch.action_kind,
                action_mask,
            )
            soft_h = _scale_gradient(h, soft_policy_trunk_grad_scale)
            soft_g_readout = _scale_gradient(
                g_readout,
                soft_policy_trunk_grad_scale,
            )
            soft_action_feat, soft_action_mask = self._policy_inputs(
                batch,
                soft_h,
                soft_g_readout,
                node_mask,
                self.soft_policy_kind_embedding,
            )
            soft_policy_logits = self.soft_policy(
                soft_g_readout,
                soft_action_feat,
                batch.action_kind,
                soft_action_mask,
            )
            return logits, soft_policy_logits

        def _policy_inputs(self, batch, h, g_readout, node_mask, kind_embedding):
            kind = kind_embedding(
                batch.action_kind.clamp(0, self.schema.action_kind_vocab_size - 1)
            )
            subject_feat = _subject_pool(
                torch,
                h,
                node_mask,
                batch.action_subjects,
                batch.subject_count,
            )
            readout = g_readout.unsqueeze(1).expand(
                -1,
                batch.action_kind.shape[1],
                -1,
            )
            action_feat = torch.cat(
                (kind, batch.action_prior.unsqueeze(-1), subject_feat, readout),
                dim=-1,
            )
            action_index = torch.arange(action_feat.shape[1], device=action_feat.device)
            action_mask = action_index.unsqueeze(0) < batch.action_count.unsqueeze(1)
            return action_feat, action_mask

        def _value_output(
            self,
            g_readout,
            value_trunk_grad_scale,
        ):
            return self._value_head_output(
                _scale_gradient(g_readout, value_trunk_grad_scale)
            )

        def _value_head_output(self, value_input):
            return torch.tanh(self.value(value_input).squeeze(-1))

        def _training_value_outputs(self, g_readout, value_trunk_grad_scale):
            if self.horizon_value is None or self.terminal_score is None:
                raise ValueError("training auxiliary outputs require v8-v32-score heads")
            value_input = _scale_gradient(g_readout, value_trunk_grad_scale)
            value_raw = self._value_head_output(value_input)
            horizon_raw = torch.tanh(self.horizon_value(value_input))
            score_raw = self.terminal_score(value_input).squeeze(-1)
            return value_raw, horizon_raw, score_raw

        def decode_value(self, value_raw):
            return value_raw

        def _encode_graph(self, graph: GraphStateTensors, node_roles: object = None):
            b, n = graph.node_tokens.shape
            device = graph.node_tokens.device
            node_index = torch.arange(n, device=device)
            node_mask = node_index.unsqueeze(0) < graph.node_count.unsqueeze(1)
            h = self.node_embedding(
                graph.node_tokens.clamp(0, self.schema.node_vocab_size - 1)
            )
            if self.attr_proj is not None:
                h = h + self.attr_proj(graph.node_attrs)
            if node_roles is None:
                raise ValueError("joint-board graph is missing node roles")
            h = h + self.board_role_embedding(node_roles)
            h = h * node_mask.unsqueeze(-1)

            g = self.global_tokens.unsqueeze(0).expand(b, -1, -1)
            position = _joint_remaining_budget_position(torch, graph.position)
            g = g + self.position_proj(position).unsqueeze(1)
            for layer in self.layers:
                h, g = layer(h, g, graph, node_mask)

            h = self.node_output_norm(h) * node_mask.unsqueeze(-1)
            g = self.global_output_norm(g)

            return h, g.mean(dim=1), node_mask

    class PointerPolicyHead(nn.Module):
        # A multi-head glimpse over the action tokens refines the graph
        # readout into a board query, and
        # a single-head dot product against the same tokens produces the
        # logits, tanh-bounded to +/-CLIP. Scores are relative across the
        # action set; the per-candidate MLP scores each action in
        # isolation and its logit scale is unbounded.
        CLIP = 10.0

        def __init__(
            self,
            arch: ArchConfig,
            action_feat_dim: int,
            action_kind_vocab_size: int,
        ) -> None:
            super().__init__()
            del action_kind_vocab_size
            dim = arch.dim
            self.heads = arch.heads
            self.token_proj = nn.Linear(action_feat_dim, dim)
            self.glimpse_query = nn.Linear(dim, dim, bias=False)
            self.glimpse_key = nn.Linear(dim, dim, bias=False)
            self.glimpse_value = nn.Linear(dim, dim, bias=False)
            self.glimpse_unify = nn.Linear(dim, dim)
            self.board_ffn = nn.Sequential(
                nn.Linear(dim, arch.ffn_dim),
                _activation_module(nn, arch.activation),
                nn.Linear(arch.ffn_dim, dim),
            )
            self.pointer_board_norm = nn.LayerNorm(dim)
            self.pointer_token_norm = nn.LayerNorm(dim)
            self.pointer_key = nn.Linear(dim, dim, bias=False)
            self.soft_pointer_key = None

        def add_legacy_soft_policy_readout(self) -> None:
            self.soft_pointer_key = nn.Linear(
                self.pointer_key.in_features,
                self.pointer_key.out_features,
                bias=False,
            )

        def forward(self, readout, action_feat, action_kind, action_mask):
            del action_kind
            board, tokens = self._representations(readout, action_feat, action_mask)
            return self._pointer_logits(
                board,
                tokens,
                self.pointer_key,
                action_mask,
            )

        def _representations(self, readout, action_feat, action_mask):
            tokens = self.token_proj(action_feat)
            return self._glimpse(readout, tokens, action_mask)

        def _glimpse(self, readout, tokens, token_mask):
            b, a, _ = tokens.shape
            dim = tokens.shape[-1]
            split = dim // self.heads
            query = self.glimpse_query(readout).view(b, self.heads, split)
            keys = self.glimpse_key(tokens).view(b, a, self.heads, split)
            values = self.glimpse_value(tokens).view(b, a, self.heads, split)
            scores = torch.einsum("bhs,bahs->bha", query, keys) / math.sqrt(split)
            # -1e9, not -inf: rows past row_count have zero valid actions,
            # and an all--inf softmax row is NaN.
            scores = scores.masked_fill(~token_mask.unsqueeze(1), -1.0e9)
            board = torch.einsum("bha,bahs->bhs", torch.softmax(scores, dim=-1), values)
            board = self.glimpse_unify(board.reshape(b, dim))
            board = board + self.board_ffn(board)
            board = self.pointer_board_norm(board)
            tokens = self.pointer_token_norm(tokens)
            return board, tokens

        def _pointer_logits(self, board, tokens, pointer_key, action_mask):
            raw = torch.einsum("bd,bad->ba", board, pointer_key(tokens)) / math.sqrt(
                tokens.shape[-1]
            )
            weight = action_mask.to(raw.dtype)
            center = (raw * weight).sum(dim=-1, keepdim=True) / weight.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1.0)
            # Softmax scores are relative. Center before clipping so a common
            # score offset cannot saturate every valid action at the same
            # endpoint and erase both policy differences and gradients.
            return self.CLIP * torch.tanh(raw - center)

    class FactorizedPointerPolicyHead(PointerPolicyHead):
        def __init__(
            self,
            arch: ArchConfig,
            action_feat_dim: int,
            action_kind_vocab_size: int,
        ) -> None:
            super().__init__(arch, action_feat_dim, action_kind_vocab_size)
            self.action_kind_vocab_size = action_kind_vocab_size
            self.family_key = nn.Linear(arch.dim, arch.dim, bias=False)

        def forward(self, readout, action_feat, action_kind, action_mask):
            kind = action_kind.clamp(0, self.action_kind_vocab_size - 1)
            tokens = self.token_proj(action_feat)
            family_tokens, family_count = self._family_tokens(
                tokens,
                kind,
                action_mask,
            )
            board, family_tokens = self._glimpse(
                readout,
                family_tokens,
                family_count > 0,
            )
            tokens = self.pointer_token_norm(tokens)
            family_logits = self._family_logits(board, family_tokens, family_count)
            conditional_logits = self._conditional_logits(
                board,
                tokens,
                kind,
                action_mask,
                family_count,
            )
            selected_family_logits = torch.gather(family_logits, 1, kind)
            logits = selected_family_logits + conditional_logits
            return logits.masked_fill(~action_mask, 0.0)

        def _family_tokens(self, tokens, kind, action_mask):
            b, _, dim = tokens.shape
            family_tokens = torch.zeros(
                (b, self.action_kind_vocab_size, dim),
                dtype=tokens.dtype,
                device=tokens.device,
            )
            family_tokens.scatter_add_(
                1,
                kind.unsqueeze(-1).expand(-1, -1, dim),
                tokens * action_mask.unsqueeze(-1).to(tokens.dtype),
            )
            family_count = torch.zeros(
                (b, self.action_kind_vocab_size),
                dtype=tokens.dtype,
                device=tokens.device,
            )
            family_count.scatter_add_(1, kind, action_mask.to(tokens.dtype))
            family_tokens = family_tokens / family_count.clamp_min(1.0).unsqueeze(-1)
            return family_tokens, family_count

        def _family_logits(self, board, family_tokens, family_count):
            raw = torch.einsum(
                "bd,bkd->bk",
                board,
                self.family_key(family_tokens),
            ) / math.sqrt(family_tokens.shape[-1])
            family_mask = family_count > 0
            weight = family_mask.to(raw.dtype)
            center = (raw * weight).sum(dim=-1, keepdim=True) / weight.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1.0)
            return self.CLIP * torch.tanh(raw - center)

        def _conditional_logits(
            self,
            board,
            tokens,
            kind,
            action_mask,
            family_count,
        ):
            raw = torch.einsum(
                "bd,bad->ba",
                board,
                self.pointer_key(tokens),
            ) / math.sqrt(tokens.shape[-1])
            score_family_count = family_count.to(raw.dtype)
            family_sum = torch.zeros_like(score_family_count)
            family_sum.scatter_add_(1, kind, raw * action_mask.to(raw.dtype))
            family_center = family_sum / score_family_count.clamp_min(1.0)
            centered = raw - torch.gather(family_center, 1, kind)
            # Normalize conditionals in fp32. Scores are bounded to +/-10, so
            # exponentiation is safe without a grouped-max reduction. Avoiding
            # that reduction is important: Inductor's BF16 amax backward
            # produces NaNs when every candidate ties at neutral initialization.
            score = (self.CLIP * torch.tanh(centered)).float()
            score_family_count = score_family_count.float()
            exp_score = torch.exp(score) * action_mask.to(score.dtype)
            family_denom = exp_score.new_zeros(score_family_count.shape)
            family_denom.scatter_add_(1, kind, exp_score)
            log_denom = torch.log(torch.gather(family_denom, 1, kind).clamp_min(1.0))
            return score - log_denom

    class GraphLayer(nn.Module):
        def __init__(self, schema: FeatureSchemaConfig, arch: ArchConfig) -> None:
            super().__init__()
            self.norm_edge = nn.LayerNorm(arch.dim)
            self.norm_exchange_h = nn.LayerNorm(arch.dim)
            self.norm_exchange_g = nn.LayerNorm(arch.dim)
            self.norm_read_h = nn.LayerNorm(arch.dim)
            self.norm_read_g = nn.LayerNorm(arch.dim)
            self.norm_ffn_h = nn.LayerNorm(arch.dim)
            self.norm_ffn_g = nn.LayerNorm(arch.dim)
            self.edge = EdgeAttention(schema, arch)
            self.exchange = DenseAttention(arch)
            self.read = DenseAttention(arch)
            self.ffn_h = _mlp(nn, arch.dim, arch.ffn_dim, arch.dim, arch.activation, arch.dropout)
            self.ffn_g = _mlp(nn, arch.dim, arch.ffn_dim, arch.dim, arch.activation, arch.dropout)

        def forward(self, h, g, graph: GraphStateTensors, node_mask):
            h_mask = node_mask.unsqueeze(-1)
            h = h + self.edge(self.norm_edge(h), graph, node_mask) * h_mask
            h = h + self.exchange(self.norm_exchange_h(h), self.norm_exchange_g(g), None) * h_mask
            g = g + self.read(self.norm_read_g(g), self.norm_read_h(h), node_mask)
            h = h + self.ffn_h(self.norm_ffn_h(h)) * h_mask
            g = g + self.ffn_g(self.norm_ffn_g(g))
            h = h * h_mask
            return h, g

    class EdgeAttention(nn.Module):
        def __init__(self, schema: FeatureSchemaConfig, arch: ArchConfig) -> None:
            super().__init__()
            self.edge_type_count = schema.edge_type_count
            self.heads = arch.heads
            self.head_dim = arch.dim // arch.heads
            self.q_proj = nn.Linear(arch.dim, arch.dim, bias=False)
            self.k_proj = nn.Linear(arch.dim, arch.dim, bias=False)
            self.v_proj = nn.Linear(arch.dim, arch.dim, bias=False)
            self.o_proj = nn.Linear(arch.dim, arch.dim, bias=False)
            self.edge_embedding = nn.Embedding(max(1, 2 * schema.edge_type_count), arch.dim)

        def forward(self, h, graph: GraphStateTensors, node_mask):
            b, n, d = h.shape
            src, dst, typ, mask = _mirrored_edges(torch, graph, node_mask, self.edge_type_count)
            q = self.q_proj(h).reshape(b, n, self.heads, self.head_dim)
            k = self.k_proj(h).reshape(b, n, self.heads, self.head_dim)
            v = self.v_proj(h).reshape(b, n, self.heads, self.head_dim)
            q_dst = _gather_nodes(torch, q.reshape(b, n, d), dst).reshape(b, -1, self.heads, self.head_dim)
            k_src = _gather_nodes(torch, k.reshape(b, n, d), src).reshape(b, -1, self.heads, self.head_dim)
            v_src = _gather_nodes(torch, v.reshape(b, n, d), src).reshape(b, -1, self.heads, self.head_dim)
            e = self.edge_embedding(typ).reshape(b, -1, self.heads, self.head_dim)
            score = (q_dst * k_src * e).sum(dim=-1) / math.sqrt(self.head_dim)
            score = score.masked_fill(~mask.unsqueeze(-1), -1.0e9)
            scatter_index = dst.unsqueeze(-1).expand(-1, -1, self.heads)
            amax = torch.full((b, n, self.heads), -1.0e9, dtype=score.dtype, device=score.device)
            amax.scatter_reduce_(1, scatter_index, score, reduce="amax", include_self=True)
            edge_amax = torch.gather(amax, 1, scatter_index)
            weight = torch.exp(score - edge_amax) * mask.unsqueeze(-1).to(score.dtype)
            denom = torch.zeros((b, n, self.heads), dtype=score.dtype, device=score.device)
            denom.scatter_add_(1, scatter_index, weight)
            msg = weight.unsqueeze(-1) * v_src
            out = torch.zeros((b, n, self.heads, self.head_dim), dtype=h.dtype, device=h.device)
            out.scatter_add_(1, dst.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.heads, self.head_dim), msg)
            out = out / denom.clamp_min(1.0e-6).unsqueeze(-1)
            return self.o_proj(out.reshape(b, n, d))

    class DenseAttention(nn.Module):
        def __init__(self, arch: ArchConfig) -> None:
            super().__init__()
            self.heads = arch.heads
            self.head_dim = arch.dim // arch.heads
            self.q = nn.Linear(arch.dim, arch.dim, bias=False)
            self.k = nn.Linear(arch.dim, arch.dim, bias=False)
            self.v = nn.Linear(arch.dim, arch.dim, bias=False)
            self.o = nn.Linear(arch.dim, arch.dim, bias=False)

        def forward(self, query, source, source_mask):
            b, q_len, d = query.shape
            k_len = source.shape[1]
            q = self.q(query).reshape(b, q_len, self.heads, self.head_dim).transpose(1, 2)
            k = self.k(source).reshape(b, k_len, self.heads, self.head_dim).transpose(1, 2)
            v = self.v(source).reshape(b, k_len, self.heads, self.head_dim).transpose(1, 2)
            score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if source_mask is not None:
                score = score.masked_fill(~source_mask.unsqueeze(1).unsqueeze(2), -1.0e9)
            weight = torch.softmax(score, dim=-1)
            out = torch.matmul(weight, v).transpose(1, 2).reshape(b, q_len, d)
            return self.o(out)

    return GraphModel


def _remaining_budget_position(torch: object, position: object):
    remaining = (position[..., 2] - position[..., 1] * position[..., 3]).clamp(0.0, 1.0)
    return torch.stack((remaining, position[..., 3]), dim=-1)


def _joint_remaining_budget_position(torch: object, position: object):
    own = _remaining_budget_position(torch, position[..., :4])
    opponent = _remaining_budget_position(torch, position[..., 4:])
    return torch.cat((own, opponent), dim=-1)


def _scale_gradient(tensor: object, scale: float):
    if scale == 1.0:
        return tensor
    detached = tensor.detach()
    return detached + (tensor - detached) * scale


def _joint_board_graph(torch: object, batch: GraphBatchTensors):
    batch_size, node_capacity = batch.node_tokens.shape
    node_index = torch.arange(node_capacity * 2, device=batch.node_tokens.device).unsqueeze(0)
    node_index = node_index.expand(batch_size, -1)
    own_node_count = batch.node_count.unsqueeze(1)
    opponent_present = batch.opponent_state_present > 0
    opponent_node_count = batch.opponent_node_count * opponent_present.to(batch.opponent_node_count.dtype)
    opponent_node_count_2d = opponent_node_count.unsqueeze(1)
    own_node = node_index < own_node_count
    opponent_node_index = node_index - own_node_count
    node_valid = own_node | ((opponent_node_index >= 0) & (opponent_node_index < opponent_node_count_2d))
    own_gather = node_index.clamp(0, node_capacity - 1)
    opponent_gather = opponent_node_index.clamp(0, node_capacity - 1)

    node_tokens = torch.where(
        own_node,
        batch.node_tokens.gather(1, own_gather),
        batch.opponent_node_tokens.gather(1, opponent_gather),
    )
    attr_dim = batch.node_attrs.shape[-1]
    own_attr_gather = own_gather.unsqueeze(-1).expand(-1, -1, attr_dim)
    opponent_attr_gather = opponent_gather.unsqueeze(-1).expand(-1, -1, attr_dim)
    node_attrs = torch.where(
        own_node.unsqueeze(-1),
        batch.node_attrs.gather(1, own_attr_gather),
        batch.opponent_node_attrs.gather(1, opponent_attr_gather),
    )
    node_tokens = node_tokens * node_valid.to(node_tokens.dtype)
    node_attrs = node_attrs * node_valid.unsqueeze(-1).to(node_attrs.dtype)
    node_roles = (~own_node).to(torch.long) * node_valid.to(torch.long)

    _, edge_capacity = batch.edge_src.shape
    edge_index = torch.arange(edge_capacity * 2, device=batch.edge_src.device).unsqueeze(0)
    edge_index = edge_index.expand(batch_size, -1)
    own_edge_count = batch.edge_count.unsqueeze(1)
    opponent_edge_count = batch.opponent_edge_count * opponent_present.to(batch.opponent_edge_count.dtype)
    opponent_edge_count_2d = opponent_edge_count.unsqueeze(1)
    own_edge = edge_index < own_edge_count
    opponent_edge_index = edge_index - own_edge_count
    edge_valid = own_edge | ((opponent_edge_index >= 0) & (opponent_edge_index < opponent_edge_count_2d))
    own_edge_gather = edge_index.clamp(0, edge_capacity - 1)
    opponent_edge_gather = opponent_edge_index.clamp(0, edge_capacity - 1)

    own_src = batch.edge_src.gather(1, own_edge_gather)
    own_dst = batch.edge_dst.gather(1, own_edge_gather)
    opponent_src = batch.opponent_edge_src.gather(1, opponent_edge_gather) + own_node_count
    opponent_dst = batch.opponent_edge_dst.gather(1, opponent_edge_gather) + own_node_count
    edge_src = torch.where(own_edge, own_src, opponent_src)
    edge_dst = torch.where(own_edge, own_dst, opponent_dst)
    edge_type = torch.where(
        own_edge,
        batch.edge_type.gather(1, own_edge_gather),
        batch.opponent_edge_type.gather(1, opponent_edge_gather),
    )
    edge_src = edge_src * edge_valid.to(edge_src.dtype)
    edge_dst = edge_dst * edge_valid.to(edge_dst.dtype)
    edge_type = edge_type * edge_valid.to(edge_type.dtype)

    graph = GraphStateTensors(
        node_count=batch.node_count + opponent_node_count,
        node_tokens=node_tokens,
        node_attrs=node_attrs,
        edge_count=batch.edge_count + opponent_edge_count,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_type=edge_type,
        position=torch.cat((batch.position, batch.opponent_position), dim=-1),
    )
    return graph, node_roles


def _mirrored_edges(
    torch: object,
    graph: GraphStateTensors,
    node_mask: object,
    edge_type_count: int,
):
    e = graph.edge_src.shape[1]
    edge_index = torch.arange(e, device=graph.edge_src.device)
    base_mask = edge_index.unsqueeze(0) < graph.edge_count.unsqueeze(1)
    src_valid = graph.edge_src < graph.node_count.unsqueeze(1)
    dst_valid = graph.edge_dst < graph.node_count.unsqueeze(1)
    type_valid = graph.edge_type < edge_type_count
    base_mask = base_mask & src_valid & dst_valid & type_valid
    dummy_node = torch.arange(e, dtype=graph.edge_src.dtype, device=graph.edge_src.device)
    dummy_node = (dummy_node % node_mask.shape[1]).unsqueeze(0)
    base_src = torch.where(base_mask, graph.edge_src, dummy_node)
    base_dst = torch.where(base_mask, graph.edge_dst, dummy_node)
    dummy_type = torch.arange(e, dtype=graph.edge_type.dtype, device=graph.edge_type.device)
    dummy_type = (dummy_type % edge_type_count).unsqueeze(0)
    base_type = torch.where(base_mask, graph.edge_type, dummy_type)
    src = (
        torch.cat((base_src, base_dst), dim=1)
        .clamp(0, node_mask.shape[1] - 1)
        .to(torch.int64)
    )
    dst = (
        torch.cat((base_dst, base_src), dim=1)
        .clamp(0, node_mask.shape[1] - 1)
        .to(torch.int64)
    )
    typ = torch.cat((base_type, base_type + edge_type_count), dim=1).clamp(
        0, max(0, 2 * edge_type_count - 1)
    )
    mask = torch.cat((base_mask, base_mask), dim=1)
    return src, dst, typ, mask


def _gather_nodes(torch: object, h: object, index: object):
    d = h.shape[-1]
    index = index.to(torch.int64)
    return torch.gather(h, 1, index.unsqueeze(-1).expand(-1, -1, d))


def _action_feat_dim(arch: ArchConfig) -> int:
    return arch.dim * 3 + 1


def _gather_subjects(
    torch: object,
    h: object,
    node_mask: object,
    action_subjects: object,
    subject_count: object,
):
    b, n, d = h.shape
    a = action_subjects.shape[1]
    s = action_subjects.shape[2]
    subject_index = torch.arange(s, device=h.device)
    valid = subject_index.reshape(1, 1, s) < subject_count.unsqueeze(-1)
    valid = valid & (action_subjects < node_mask.sum(dim=1).reshape(b, 1, 1))
    dummy = torch.arange(a * s, dtype=action_subjects.dtype, device=h.device)
    dummy = (dummy % n).reshape(1, a, s)
    safe = torch.where(valid, action_subjects, dummy).clamp(0, n - 1)
    # Gather over h's node dim directly: routing the gather through an
    # (b, a, n, d) expand made the backward materialize that full tensor
    # (tens of GiB at wide action masks) before reducing it.
    flat = safe.to(torch.int64).reshape(b, a * s, 1).expand(b, a * s, d)
    gathered = torch.gather(h, 1, flat).reshape(b, a, s, d)
    return gathered, valid


def _subject_pool(torch: object, h: object, node_mask: object, action_subjects: object, subject_count: object):
    gathered, valid = _gather_subjects(torch, h, node_mask, action_subjects, subject_count)
    weight = valid.unsqueeze(-1).to(h.dtype)
    denom = weight.sum(dim=2).clamp_min(1.0)
    return (gathered * weight).sum(dim=2) / denom


def _mlp(nn: object, in_dim: int, hidden_dim: int, out_dim: int, activation: str, dropout: float):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        _activation_module(nn, activation),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


def _activation_module(nn: object, activation: str):
    if activation == "gelu":
        return nn.GELU()
    if activation == "relu":
        return nn.ReLU()
    raise ValueError("unsupported activation")


def _torch():
    return importlib.import_module("torch")

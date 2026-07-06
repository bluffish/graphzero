from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from gz.codec import BatchView, FeatureSchemaConfig


@dataclass(frozen=True, slots=True)
class ArchConfig:
    name: str = "gz-graph-v1"
    dim: int = 128
    layers: int = 4
    heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    aggregation: str = "attention"
    global_tokens: int = 1
    value_input: str = "single"
    policy_head: str = "mlp"

    def __post_init__(self) -> None:
        if self.name != "gz-graph-v1":
            raise ValueError("unsupported graph arch name")
        if self.dim <= 0 or self.layers <= 0 or self.heads <= 0 or self.ffn_dim <= 0:
            raise ValueError("arch dimensions must be positive")
        if self.dim % self.heads != 0:
            raise ValueError("dim must be divisible by heads")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout out of range")
        if self.activation not in {"gelu", "relu"}:
            raise ValueError("unsupported activation")
        if self.aggregation not in {"attention", "gine"}:
            raise ValueError("unsupported aggregation")
        if self.global_tokens <= 0:
            raise ValueError("global_tokens must be positive")
        if self.value_input not in {"single", "scalar", "pair"}:
            raise ValueError("unsupported value_input")
        if self.policy_head not in {"mlp", "pointer"}:
            raise ValueError("unsupported policy_head")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dim": self.dim,
            "layers": self.layers,
            "heads": self.heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "activation": self.activation,
            "aggregation": self.aggregation,
            "global_tokens": self.global_tokens,
            "value_input": self.value_input,
            "policy_head": self.policy_head,
        }

    def encode(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    def hash(self) -> bytes:
        hasher = hashlib.blake2b(digest_size=32)
        _update_chunk(hasher, b"gz-arch-config-v1")
        _update_chunk(hasher, self.encode())
        return hasher.digest()

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ArchConfig:
        fields = {
            "name",
            "dim",
            "layers",
            "heads",
            "ffn_dim",
            "dropout",
            "activation",
            "aggregation",
            "global_tokens",
            "value_input",
            "policy_head",
        }
        optional = {"value_input", "policy_head"}
        keys = set(value)
        if not (fields - optional <= keys <= fields):
            raise ValueError("arch config fields mismatch")
        return cls(
            name=_str(value, "name"),
            dim=_int(value, "dim"),
            layers=_int(value, "layers"),
            heads=_int(value, "heads"),
            ffn_dim=_int(value, "ffn_dim"),
            dropout=_float(value, "dropout"),
            activation=_str(value, "activation"),
            aggregation=_str(value, "aggregation"),
            global_tokens=_int(value, "global_tokens"),
            value_input=_str(value, "value_input", "single"),
            policy_head=_str(value, "policy_head", "mlp"),
        )


class GraphBatchTensors(NamedTuple):
    node_count: object
    node_tokens: object
    node_attrs: object
    edge_count: object
    edge_src: object
    edge_dst: object
    edge_type: object
    action_count: object
    action_kind: object
    action_prior: object
    subject_count: object
    action_subjects: object
    position: object
    opponent_reward: object
    opponent_present: object
    opponent_state_present: object
    opponent_node_count: object
    opponent_node_tokens: object
    opponent_node_attrs: object
    opponent_edge_count: object
    opponent_edge_src: object
    opponent_edge_dst: object
    opponent_edge_type: object
    opponent_position: object


class GraphStateTensors(NamedTuple):
    node_count: object
    node_tokens: object
    node_attrs: object
    edge_count: object
    edge_src: object
    edge_dst: object
    edge_type: object
    position: object


def build_model(schema: FeatureSchemaConfig, arch: ArchConfig):
    return _model_class()(schema, arch)


def tensors_from_batch(view: BatchView, device: str | object, pinned_staging: bool = True) -> GraphBatchTensors:
    return BatchStager.from_view(view, device=device, pinned_staging=pinned_staging).copy(view)


class BatchStager:
    def __init__(self, schema: FeatureSchemaConfig, capacity: int, device: str | object, pinned_staging: bool = True) -> None:
        torch = _torch()
        self.schema = schema
        self.capacity = capacity
        self.device = torch.device(device)
        self.pin = bool(pinned_staging and self.device.type == "cuda")
        b = capacity
        n = schema.max_nodes
        e = schema.max_edges
        a = schema.max_actions
        s = schema.max_subjects
        d = schema.node_attr_dim
        self.node_count = _StagedTensor((b,), torch.int64, self.device, self.pin)
        self.node_tokens = _StagedTensor((b, n), torch.int64, self.device, self.pin)
        self.node_attrs = _StagedTensor((b, n, d), torch.float32, self.device, self.pin)
        self.edge_count = _StagedTensor((b,), torch.int64, self.device, self.pin)
        self.edge_src = _StagedTensor((b, e), torch.int64, self.device, self.pin)
        self.edge_dst = _StagedTensor((b, e), torch.int64, self.device, self.pin)
        self.edge_type = _StagedTensor((b, e), torch.int64, self.device, self.pin)
        self.action_count = _StagedTensor((b,), torch.int64, self.device, self.pin)
        self.action_kind = _StagedTensor((b, a), torch.int64, self.device, self.pin)
        self.action_prior = _StagedTensor((b, a), torch.float32, self.device, self.pin)
        self.subject_count = _StagedTensor((b, a), torch.int64, self.device, self.pin)
        self.action_subjects = _StagedTensor((b, a, s), torch.int64, self.device, self.pin)
        self.position = _StagedTensor((b, 4), torch.float32, self.device, self.pin)
        self.opponent_reward = _StagedTensor((b,), torch.float32, self.device, self.pin)
        self.opponent_present = _StagedTensor((b,), torch.float32, self.device, self.pin)
        self.opponent_state_present = _StagedTensor((b,), torch.float32, self.device, self.pin)
        self.opponent_node_count = _StagedTensor((b,), torch.int64, self.device, self.pin)
        self.opponent_node_tokens = _StagedTensor((b, n), torch.int64, self.device, self.pin)
        self.opponent_node_attrs = _StagedTensor((b, n, d), torch.float32, self.device, self.pin)
        self.opponent_edge_count = _StagedTensor((b,), torch.int64, self.device, self.pin)
        self.opponent_edge_src = _StagedTensor((b, e), torch.int64, self.device, self.pin)
        self.opponent_edge_dst = _StagedTensor((b, e), torch.int64, self.device, self.pin)
        self.opponent_edge_type = _StagedTensor((b, e), torch.int64, self.device, self.pin)
        self.opponent_position = _StagedTensor((b, 4), torch.float32, self.device, self.pin)

    @classmethod
    def from_view(cls, view: BatchView, device: str | object, pinned_staging: bool = True) -> BatchStager:
        schema = FeatureSchemaConfig(
            name="batch-view",
            node_vocab_size=max(2, int(view.node_tokens.max(initial=0)) + 1),
            node_attr_dim=view.dims.node_attr_dim,
            edge_type_count=max(1, int(view.edge_type.max(initial=0)) + 1),
            action_kind_vocab_size=max(3, int(view.action_kind.max(initial=0)) + 1),
            max_nodes=view.dims.max_nodes,
            max_edges=view.dims.max_edges,
            max_actions=view.dims.max_actions,
            max_subjects=view.dims.max_subjects,
            opponent_reward_scale=256.0,
            expander_degree=0,
            expander_seed=0,
        )
        return cls(schema, view.batch_capacity, device, pinned_staging)

    def copy(self, view: BatchView) -> GraphBatchTensors:
        self._check_view(view)
        self.node_count.copy(view.node_count)
        self.node_tokens.copy(view.node_tokens)
        if view.node_attrs is None:
            self.node_attrs.zero_()
        else:
            self.node_attrs.copy(view.node_attrs)
        self.edge_count.copy(view.edge_count)
        self.edge_src.copy(view.edge_src)
        self.edge_dst.copy(view.edge_dst)
        self.edge_type.copy(view.edge_type)
        self.action_count.copy(view.action_count)
        self.action_kind.copy(view.action_kind)
        self.action_prior.copy(view.action_prior)
        self.subject_count.copy(view.subject_count)
        self.action_subjects.copy(view.action_subjects)
        self.position.copy(view.position)
        self.opponent_reward.copy(view.opponent_reward)
        self.opponent_present.copy(view.opponent_present)
        self.opponent_state_present.copy(view.opponent_state_present)
        self.opponent_node_count.copy(view.opponent_node_count)
        self.opponent_node_tokens.copy(view.opponent_node_tokens)
        if view.opponent_node_attrs is None:
            self.opponent_node_attrs.zero_()
        else:
            self.opponent_node_attrs.copy(view.opponent_node_attrs)
        self.opponent_edge_count.copy(view.opponent_edge_count)
        self.opponent_edge_src.copy(view.opponent_edge_src)
        self.opponent_edge_dst.copy(view.opponent_edge_dst)
        self.opponent_edge_type.copy(view.opponent_edge_type)
        self.opponent_position.copy(view.opponent_position)
        return self.tensors()

    def dummy(self) -> GraphBatchTensors:
        self.node_count.fill_(1)
        self.node_tokens.zero_()
        self.node_tokens.cpu[..., 0] = 1
        self.node_attrs.zero_()
        self.edge_count.zero_()
        self.edge_src.zero_()
        self.edge_dst.zero_()
        self.edge_type.zero_()
        self.action_count.fill_(1)
        self.action_kind.zero_()
        self.action_kind.cpu[..., 0] = 1
        self.action_prior.zero_()
        self.subject_count.zero_()
        self.action_subjects.fill_(0xFFFF_FFFF)
        self.position.zero_()
        self.opponent_reward.zero_()
        self.opponent_present.zero_()
        self.opponent_state_present.zero_()
        self.opponent_node_count.zero_()
        self.opponent_node_tokens.zero_()
        self.opponent_node_attrs.zero_()
        self.opponent_edge_count.zero_()
        self.opponent_edge_src.zero_()
        self.opponent_edge_dst.zero_()
        self.opponent_edge_type.zero_()
        self.opponent_position.zero_()
        for tensor in self._all():
            tensor.sync()
        return self.tensors()

    def tensors(self) -> GraphBatchTensors:
        return GraphBatchTensors(
            node_count=self.node_count.device_tensor,
            node_tokens=self.node_tokens.device_tensor,
            node_attrs=self.node_attrs.device_tensor,
            edge_count=self.edge_count.device_tensor,
            edge_src=self.edge_src.device_tensor,
            edge_dst=self.edge_dst.device_tensor,
            edge_type=self.edge_type.device_tensor,
            action_count=self.action_count.device_tensor,
            action_kind=self.action_kind.device_tensor,
            action_prior=self.action_prior.device_tensor,
            subject_count=self.subject_count.device_tensor,
            action_subjects=self.action_subjects.device_tensor,
            position=self.position.device_tensor,
            opponent_reward=self.opponent_reward.device_tensor,
            opponent_present=self.opponent_present.device_tensor,
            opponent_state_present=self.opponent_state_present.device_tensor,
            opponent_node_count=self.opponent_node_count.device_tensor,
            opponent_node_tokens=self.opponent_node_tokens.device_tensor,
            opponent_node_attrs=self.opponent_node_attrs.device_tensor,
            opponent_edge_count=self.opponent_edge_count.device_tensor,
            opponent_edge_src=self.opponent_edge_src.device_tensor,
            opponent_edge_dst=self.opponent_edge_dst.device_tensor,
            opponent_edge_type=self.opponent_edge_type.device_tensor,
            opponent_position=self.opponent_position.device_tensor,
        )

    def _check_view(self, view: BatchView) -> None:
        dims = view.dims
        if view.batch_capacity != self.capacity:
            raise ValueError("batch capacity mismatch")
        if dims.max_nodes != self.schema.max_nodes:
            raise ValueError("max_nodes mismatch")
        if dims.max_edges != self.schema.max_edges:
            raise ValueError("max_edges mismatch")
        if dims.max_actions != self.schema.max_actions:
            raise ValueError("max_actions mismatch")
        if dims.max_subjects != self.schema.max_subjects:
            raise ValueError("max_subjects mismatch")
        if dims.node_attr_dim != self.schema.node_attr_dim:
            raise ValueError("node_attr_dim mismatch")

    def _all(self) -> tuple[_StagedTensor, ...]:
        return (
            self.node_count,
            self.node_tokens,
            self.node_attrs,
            self.edge_count,
            self.edge_src,
            self.edge_dst,
            self.edge_type,
            self.action_count,
            self.action_kind,
            self.action_prior,
            self.subject_count,
            self.action_subjects,
            self.position,
            self.opponent_reward,
            self.opponent_present,
            self.opponent_state_present,
            self.opponent_node_count,
            self.opponent_node_tokens,
            self.opponent_node_attrs,
            self.opponent_edge_count,
            self.opponent_edge_src,
            self.opponent_edge_dst,
            self.opponent_edge_type,
            self.opponent_position,
        )


class _StagedTensor:
    def __init__(self, shape: tuple[int, ...], dtype: object, device: object, pin: bool) -> None:
        torch = _torch()
        self.cpu = torch.empty(shape, dtype=dtype, pin_memory=pin)
        self.device_tensor = torch.empty(shape, dtype=dtype, device=device)
        self.non_blocking = pin

    def copy(self, array: np.ndarray) -> None:
        np.copyto(self.cpu.numpy(), array, casting="unsafe")
        self.sync()

    def zero_(self) -> None:
        self.cpu.zero_()
        self.sync()

    def fill_(self, value: int | float) -> None:
        self.cpu.fill_(value)
        self.sync()

    def sync(self) -> None:
        self.device_tensor.copy_(self.cpu, non_blocking=self.non_blocking)


@lru_cache(maxsize=1)
def _model_class():
    torch = _torch()
    nn = torch.nn
    functional = torch.nn.functional

    class GraphModel(nn.Module):
        def __init__(self, schema: FeatureSchemaConfig, arch: ArchConfig) -> None:
            super().__init__()
            self.schema = schema
            self.arch = arch
            self.node_embedding = nn.Embedding(schema.node_vocab_size, arch.dim, padding_idx=0)
            self.attr_proj = nn.Linear(schema.node_attr_dim, arch.dim, bias=False) if schema.node_attr_dim else None
            self.position_proj = nn.Linear(4, arch.dim)
            self.global_tokens = nn.Parameter(torch.zeros(arch.global_tokens, arch.dim))
            self.layers = nn.ModuleList([GraphLayer(schema, arch) for _ in range(arch.layers)])
            self.kind_embedding = nn.Embedding(schema.action_kind_vocab_size, arch.dim, padding_idx=0)
            if arch.policy_head == "pointer":
                self.policy = PointerPolicyHead(arch)
            else:
                self.policy = _mlp(nn, arch.dim * 3 + 1, arch.ffn_dim, 1, arch.activation, arch.dropout)
            value_dim = arch.dim
            if arch.value_input == "scalar":
                value_dim += 2
            elif arch.value_input == "pair":
                value_dim *= 2
            self.value = _mlp(nn, value_dim, arch.ffn_dim, 1, arch.activation, arch.dropout)

        def forward(self, batch: GraphBatchTensors, value_flip: object = None):
            graph = _self_graph(batch)
            h, g_readout, node_mask = self._encode_graph(graph)
            subject_pool = _subject_pool(torch, h, node_mask, batch.action_subjects, batch.subject_count)
            kind = self.kind_embedding(batch.action_kind.clamp(0, self.schema.action_kind_vocab_size - 1))
            prior = batch.action_prior.unsqueeze(-1)
            readout = g_readout.unsqueeze(1).expand(-1, batch.action_kind.shape[1], -1)
            action_feat = torch.cat((kind, prior, subject_pool, readout), dim=-1)
            if self.arch.policy_head == "pointer":
                action_index = torch.arange(action_feat.shape[1], device=action_feat.device)
                action_mask = action_index.unsqueeze(0) < batch.action_count.unsqueeze(1)
                logits = self.policy(g_readout, action_feat, action_mask)
            else:
                logits = self.policy(action_feat).squeeze(-1)

            value_input = g_readout
            if self.arch.value_input == "scalar":
                opponent = torch.stack((batch.opponent_reward, batch.opponent_present), dim=-1).to(g_readout.dtype)
                value_input = torch.cat((g_readout, opponent), dim=-1)
            elif self.arch.value_input == "pair":
                _, opponent_readout, _ = self._encode_graph(_opponent_graph(batch))
                present = (batch.opponent_state_present > 0).unsqueeze(-1)
                opponent_readout = torch.where(present, opponent_readout, torch.zeros_like(opponent_readout))
                if value_flip is not None:
                    flip = value_flip.to(torch.bool).unsqueeze(-1) & present
                    left = torch.where(flip, opponent_readout, g_readout)
                    right = torch.where(flip, g_readout, opponent_readout)
                    value_input = torch.cat((left, right), dim=-1)
                else:
                    value_input = torch.cat((g_readout, opponent_readout), dim=-1)
            value_raw = self.value(value_input).squeeze(-1)
            return value_raw, logits

        def _encode_graph(self, graph: GraphStateTensors):
            b, n = graph.node_tokens.shape
            device = graph.node_tokens.device
            node_index = torch.arange(n, device=device)
            node_mask = node_index.unsqueeze(0) < graph.node_count.unsqueeze(1)
            h = self.node_embedding(graph.node_tokens.clamp(0, self.schema.node_vocab_size - 1))
            if self.attr_proj is not None:
                h = h + self.attr_proj(graph.node_attrs)
            h = h * node_mask.unsqueeze(-1)

            position = self.position_proj(graph.position).unsqueeze(1)
            g = self.global_tokens.unsqueeze(0).expand(b, -1, -1) + position
            for layer in self.layers:
                h, g = layer(h, g, graph, node_mask)

            return h, g.mean(dim=1), node_mask

    class PointerPolicyHead(nn.Module):
        # whittlezero's tsp_pointer scorer: a multi-head glimpse over the
        # action tokens refines the graph readout into a board query, and
        # a single-head dot product against the same tokens produces the
        # logits, tanh-bounded to +/-CLIP. Scores are relative across the
        # action set; the per-candidate MLP scores each action in
        # isolation and its logit scale is unbounded.
        CLIP = 10.0

        def __init__(self, arch: ArchConfig) -> None:
            super().__init__()
            dim = arch.dim
            self.heads = arch.heads
            self.token_proj = nn.Linear(arch.dim * 3 + 1, dim)
            self.glimpse_query = nn.Linear(dim, dim, bias=False)
            self.glimpse_key = nn.Linear(dim, dim, bias=False)
            self.glimpse_value = nn.Linear(dim, dim, bias=False)
            self.glimpse_unify = nn.Linear(dim, dim)
            self.board_ffn = nn.Sequential(
                nn.Linear(dim, arch.ffn_dim),
                _activation_module(nn, arch.activation),
                nn.Linear(arch.ffn_dim, dim),
            )
            self.pointer_key = nn.Linear(dim, dim, bias=False)

        def forward(self, readout, action_feat, action_mask):
            b, a, _ = action_feat.shape
            tokens = self.token_proj(action_feat)
            dim = tokens.shape[-1]
            split = dim // self.heads
            query = self.glimpse_query(readout).view(b, self.heads, split)
            keys = self.glimpse_key(tokens).view(b, a, self.heads, split)
            values = self.glimpse_value(tokens).view(b, a, self.heads, split)
            scores = torch.einsum("bhs,bahs->bha", query, keys) / math.sqrt(split)
            # -1e9, not -inf: rows past row_count have zero valid actions,
            # and an all--inf softmax row is NaN.
            scores = scores.masked_fill(~action_mask.unsqueeze(1), -1.0e9)
            board = torch.einsum("bha,bahs->bhs", torch.softmax(scores, dim=-1), values)
            board = self.glimpse_unify(board.reshape(b, dim))
            board = board + self.board_ffn(board)
            raw = torch.einsum("bd,bad->ba", board, self.pointer_key(tokens)) / math.sqrt(dim)
            return self.CLIP * torch.tanh(raw)

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
            self.edge = EdgeAttention(schema, arch) if arch.aggregation == "attention" else EdgeGine(schema, arch)
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

    class EdgeGine(nn.Module):
        def __init__(self, schema: FeatureSchemaConfig, arch: ArchConfig) -> None:
            super().__init__()
            self.edge_type_count = schema.edge_type_count
            self.k_proj = nn.Linear(arch.dim, arch.dim, bias=False)
            self.edge_embedding = nn.Embedding(max(1, 2 * schema.edge_type_count), arch.dim)
            self.eps = nn.Parameter(torch.zeros(()))
            self.out = _mlp(nn, arch.dim, arch.ffn_dim, arch.dim, arch.activation, arch.dropout)
            self.activation = _activation(functional, arch.activation)

        def forward(self, h, graph: GraphStateTensors, node_mask):
            b, n, d = h.shape
            src, dst, typ, mask = _mirrored_edges(torch, graph, node_mask, self.edge_type_count)
            src_h = _gather_nodes(torch, self.k_proj(h), src)
            msg = self.activation(src_h + self.edge_embedding(typ)) * mask.unsqueeze(-1).to(h.dtype)
            out = torch.zeros((b, n, d), dtype=h.dtype, device=h.device)
            out.scatter_add_(1, dst.unsqueeze(-1).expand(-1, -1, d), msg)
            return self.out((1.0 + self.eps) * h + out)

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


def _self_graph(batch: GraphBatchTensors) -> GraphStateTensors:
    return GraphStateTensors(
        node_count=batch.node_count,
        node_tokens=batch.node_tokens,
        node_attrs=batch.node_attrs,
        edge_count=batch.edge_count,
        edge_src=batch.edge_src,
        edge_dst=batch.edge_dst,
        edge_type=batch.edge_type,
        position=batch.position,
    )


def _opponent_graph(batch: GraphBatchTensors) -> GraphStateTensors:
    return GraphStateTensors(
        node_count=batch.opponent_node_count,
        node_tokens=batch.opponent_node_tokens,
        node_attrs=batch.opponent_node_attrs,
        edge_count=batch.opponent_edge_count,
        edge_src=batch.opponent_edge_src,
        edge_dst=batch.opponent_edge_dst,
        edge_type=batch.opponent_edge_type,
        position=batch.opponent_position,
    )


def _mirrored_edges(torch: object, graph: GraphStateTensors, node_mask: object, edge_type_count: int):
    e = graph.edge_src.shape[1]
    edge_index = torch.arange(e, device=graph.edge_src.device)
    base_mask = edge_index.unsqueeze(0) < graph.edge_count.unsqueeze(1)
    src_valid = graph.edge_src < graph.node_count.unsqueeze(1)
    dst_valid = graph.edge_dst < graph.node_count.unsqueeze(1)
    type_valid = graph.edge_type < edge_type_count
    base_mask = base_mask & src_valid & dst_valid & type_valid
    src = torch.cat((graph.edge_src, graph.edge_dst), dim=1).clamp(0, node_mask.shape[1] - 1)
    dst = torch.cat((graph.edge_dst, graph.edge_src), dim=1).clamp(0, node_mask.shape[1] - 1)
    typ = torch.cat((graph.edge_type, graph.edge_type + edge_type_count), dim=1).clamp(0, max(0, 2 * edge_type_count - 1))
    mask = torch.cat((base_mask, base_mask), dim=1)
    return src, dst, typ, mask


def _gather_nodes(torch: object, h: object, index: object):
    d = h.shape[-1]
    return torch.gather(h, 1, index.unsqueeze(-1).expand(-1, -1, d))


def _subject_pool(torch: object, h: object, node_mask: object, action_subjects: object, subject_count: object):
    b, n, d = h.shape
    a = action_subjects.shape[1]
    s = action_subjects.shape[2]
    subject_index = torch.arange(s, device=h.device)
    valid = subject_index.reshape(1, 1, s) < subject_count.unsqueeze(-1)
    valid = valid & (action_subjects < node_mask.sum(dim=1).reshape(b, 1, 1))
    safe = action_subjects.clamp(0, n - 1)
    # Gather over h's node dim directly: routing the gather through an
    # (b, a, n, d) expand made the backward materialize that full tensor
    # (tens of GiB at wide action masks) before reducing it.
    flat = safe.reshape(b, a * s, 1).expand(b, a * s, d)
    gathered = torch.gather(h, 1, flat).reshape(b, a, s, d)
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


def _activation(functional: object, activation: str):
    if activation == "gelu":
        return functional.gelu
    if activation == "relu":
        return functional.relu
    raise ValueError("unsupported activation")


def _torch():
    return importlib.import_module("torch")


def _update_chunk(hasher: object, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "little"))
    hasher.update(value)


def _int(value: dict[str, object], name: str) -> int:
    field = value[name]
    if not isinstance(field, int):
        raise ValueError(f"{name} must be an integer")
    return field


def _float(value: dict[str, object], name: str) -> float:
    field = value[name]
    if not isinstance(field, (float, int)):
        raise ValueError(f"{name} must be numeric")
    return float(field)


def _str(value: dict[str, object], name: str, default: str | None = None) -> str:
    field = value.get(name, default)
    if not isinstance(field, str):
        raise ValueError(f"{name} must be a string")
    return field

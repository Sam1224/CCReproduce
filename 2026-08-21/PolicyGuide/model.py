from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class PolicyNode:
    node_id: str
    description: str
    required_evidence: Tuple[str, ...] = ()
    actions: Tuple[str, ...] = ()
    next_nodes: Tuple[str, ...] = ()


@dataclass
class PolicyGraph:
    nodes: Dict[str, PolicyNode]
    start: str
    terminal: str

    def node(self, node_id: str) -> PolicyNode:
        return self.nodes[node_id]

    def successors(self, node_id: str) -> Tuple[str, ...]:
        return self.nodes[node_id].next_nodes


@dataclass
class DialogState:
    current_node: str
    evidence: Set[str] = field(default_factory=set)
    completed_nodes: Set[str] = field(default_factory=set)
    open_requests: Set[str] = field(default_factory=set)

    def clone(self) -> "DialogState":
        return DialogState(
            current_node=self.current_node,
            evidence=set(self.evidence),
            completed_nodes=set(self.completed_nodes),
            open_requests=set(self.open_requests),
        )


class TextFeatureEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 96):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, hidden_dim, mode="mean")
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, token_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(token_ids, offsets))


class PolicyGuideScorer(nn.Module):
    def __init__(self, vocab_size: int, num_nodes: int, hidden_dim: int = 96):
        super().__init__()
        self.text_encoder = TextFeatureEncoder(vocab_size, hidden_dim)
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_nodes),
        )

    def forward(self, token_ids: torch.Tensor, offsets: torch.Tensor, current_node: torch.Tensor) -> torch.Tensor:
        text_repr = self.text_encoder(token_ids, offsets)
        node_repr = self.node_embedding(current_node)
        return self.classifier(torch.cat([text_repr, node_repr], dim=-1))


@dataclass
class VerificationResult:
    allowed: bool
    next_node: str
    remediation: str
    missing_evidence: Tuple[str, ...]
    recommended_actions: Tuple[str, ...]


class PolicyGuide:
    def __init__(self, graph: PolicyGraph):
        self.graph = graph

    def verify_turn(self, state: DialogState, observed_evidence: Iterable[str], proposed_action: Optional[str] = None) -> VerificationResult:
        new_state = state.clone()
        new_state.evidence.update(observed_evidence)
        node = self.graph.node(new_state.current_node)
        missing = tuple(item for item in node.required_evidence if item not in new_state.evidence)
        if missing:
            return VerificationResult(
                allowed=False,
                next_node=node.node_id,
                remediation=self._remediation(node, missing),
                missing_evidence=missing,
                recommended_actions=node.actions,
            )
        new_state.completed_nodes.add(node.node_id)
        if proposed_action and proposed_action not in node.actions and node.actions:
            return VerificationResult(
                allowed=False,
                next_node=node.node_id,
                remediation=f"当前动作 `{proposed_action}` 与策略节点 `{node.description}` 不一致，应先执行：{', '.join(node.actions)}。",
                missing_evidence=(),
                recommended_actions=node.actions,
            )
        next_node = self._advance(node.node_id, new_state.evidence)
        allowed = next_node == self.graph.terminal or proposed_action in node.actions or not node.actions
        return VerificationResult(
            allowed=allowed,
            next_node=next_node,
            remediation="已满足当前节点要求，可沿策略图继续执行。" if allowed else self.graph.node(next_node).description,
            missing_evidence=(),
            recommended_actions=self.graph.node(next_node).actions if next_node != self.graph.terminal else (),
        )

    def _advance(self, node_id: str, evidence: Set[str]) -> str:
        for candidate in self.graph.successors(node_id):
            candidate_node = self.graph.node(candidate)
            if any(item not in evidence for item in candidate_node.required_evidence):
                return candidate
        successors = self.graph.successors(node_id)
        return successors[0] if successors else self.graph.terminal

    @staticmethod
    def _remediation(node: PolicyNode, missing: Sequence[str]) -> str:
        return f"在 `{node.description}` 阶段缺少证据：{', '.join(missing)}。请先补齐这些步骤，再允许敏感动作。"


def build_retail_policy_graph() -> PolicyGraph:
    nodes = {
        "identify_user": PolicyNode("identify_user", "确认用户身份", ("user_id",), ("ask_user_id",), ("load_order",)),
        "load_order": PolicyNode("load_order", "读取订单并确认归属", ("order_id", "order_owner_verified"), ("get_order",), ("check_eligibility",)),
        "check_eligibility": PolicyNode("check_eligibility", "检查退换/补偿资格", ("policy_rule", "eligibility_result"), ("check_policy",), ("confirm_action",)),
        "confirm_action": PolicyNode("confirm_action", "向用户复述结果并获取确认", ("user_confirmation",), ("ask_confirmation",), ("execute_mutation",)),
        "execute_mutation": PolicyNode("execute_mutation", "执行变更动作", ("mutation_args_grounded",), ("refund", "exchange", "coupon"), ("done",)),
        "done": PolicyNode("done", "流程完成"),
    }
    return PolicyGraph(nodes=nodes, start="identify_user", terminal="done")


def batch_to_tensors(batch: List[Dict[str, object]], vocab: Dict[str, int], node_to_id: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    token_stream: List[int] = []
    offsets: List[int] = []
    current_nodes: List[int] = []
    targets: List[int] = []
    for example in batch:
        offsets.append(len(token_stream))
        tokens = str(example["text"]).lower().split()
        token_stream.extend(vocab.get(token, vocab["<unk>"]) for token in tokens)
        current_nodes.append(node_to_id[str(example["current_node"])])
        targets.append(node_to_id[str(example["target_node"])])
    return (
        torch.tensor(token_stream, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(current_nodes, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )

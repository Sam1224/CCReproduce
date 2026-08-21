from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PolicyExample:
    text: str
    current_node: str
    target_node: str
    evidence: Tuple[str, ...]


class RetailPolicyDataset(Dataset):
    def __init__(self) -> None:
        self.examples = [
            PolicyExample("user asks refund but no identity supplied", "identify_user", "identify_user", ()),
            PolicyExample("customer provides user_id and asks refund order", "identify_user", "load_order", ("user_id",)),
            PolicyExample("order_id provided but ownership has not been verified", "load_order", "load_order", ("user_id", "order_id")),
            PolicyExample("order owner verified and order_id is available", "load_order", "check_eligibility", ("order_id", "order_owner_verified")),
            PolicyExample("policy rule says product is returnable and eligibility_result true", "check_eligibility", "confirm_action", ("policy_rule", "eligibility_result")),
            PolicyExample("agent did not ask confirmation before refund", "confirm_action", "confirm_action", ("policy_rule", "eligibility_result")),
            PolicyExample("user_confirmation yes after summary", "confirm_action", "execute_mutation", ("user_confirmation",)),
            PolicyExample("refund tool arguments grounded in policy and order", "execute_mutation", "done", ("mutation_args_grounded",)),
            PolicyExample("exchange request with verified account", "identify_user", "load_order", ("user_id",)),
            PolicyExample("coupon compensation requested before eligibility check", "check_eligibility", "check_eligibility", ("policy_rule",)),
            PolicyExample("telecom user confirms after troubleshooting and eligibility", "confirm_action", "execute_mutation", ("user_confirmation",)),
            PolicyExample("retail cancellation needs order owner verified", "load_order", "load_order", ("order_id",)),
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        example = self.examples[index]
        return {
            "text": example.text,
            "current_node": example.current_node,
            "target_node": example.target_node,
            "evidence": example.evidence,
        }


def build_vocab(dataset: RetailPolicyDataset) -> Dict[str, int]:
    vocab = {"<unk>": 0}
    for example in dataset.examples:
        for token in example.text.lower().split():
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def node_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    nodes = ["identify_user", "load_order", "check_eligibility", "confirm_action", "execute_mutation", "done"]
    node_to_id = {node: index for index, node in enumerate(nodes)}
    return node_to_id, {index: node for node, index in node_to_id.items()}


def collate_policy_batch(batch: List[Dict[str, object]], vocab: Dict[str, int], node_to_id: Dict[str, int]):
    from model import batch_to_tensors

    return batch_to_tensors(batch, vocab, node_to_id)


def make_loader(batch_size: int = 4, shuffle: bool = True):
    dataset = RetailPolicyDataset()
    vocab = build_vocab(dataset)
    node_to_id, id_to_node = node_mappings()

    def collate(batch):
        return collate_policy_batch(batch, vocab, node_to_id)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate), vocab, node_to_id, id_to_node

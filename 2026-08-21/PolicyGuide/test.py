from __future__ import annotations

import torch

from dataset import make_loader
from model import DialogState, PolicyGuide, PolicyGuideScorer, build_retail_policy_graph


def test_symbolic_policy_guide() -> None:
    graph = build_retail_policy_graph()
    guide = PolicyGuide(graph)
    state = DialogState(current_node=graph.start)
    blocked = guide.verify_turn(state, observed_evidence=[])
    assert not blocked.allowed
    assert blocked.next_node == "identify_user"
    assert blocked.missing_evidence == ("user_id",)

    identified = guide.verify_turn(state, observed_evidence=["user_id"], proposed_action="ask_user_id")
    assert identified.allowed
    assert identified.next_node == "load_order"

    ready_to_mutate = DialogState(current_node="execute_mutation", evidence={"mutation_args_grounded"})
    allowed = guide.verify_turn(ready_to_mutate, observed_evidence=[], proposed_action="refund")
    assert allowed.allowed
    assert allowed.next_node == "done"


def test_neural_scorer_forward() -> None:
    loader, vocab, node_to_id, _ = make_loader(batch_size=3, shuffle=False)
    token_ids, offsets, current_nodes, targets = next(iter(loader))
    model = PolicyGuideScorer(len(vocab), len(node_to_id))
    logits = model(token_ids, offsets, current_nodes)
    assert logits.shape == (targets.numel(), len(node_to_id))
    loss = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.isfinite(loss)


if __name__ == "__main__":
    test_symbolic_policy_guide()
    test_neural_scorer_forward()
    print("PolicyGuide smoke tests passed")

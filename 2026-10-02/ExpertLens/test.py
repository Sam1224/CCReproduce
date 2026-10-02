from expertlens import ExpertLens, MultimodalMoE, accuracy, freeze_for_selected_experts, make_token_embeddings, train_epoch
from toy_data import DOMAIN_KEYWORDS, DOMAINS, VOCAB, make_toy_batch


def test_expertlens_pipeline() -> None:
    model = MultimodalMoE(input_dim=24, hidden_dim=24, num_experts=8, num_classes=len(DOMAINS))
    batches = [make_toy_batch(domain, n=32) for domain in DOMAINS for _ in range(3)]
    for _ in range(20):
        train_epoch(model, batches, lr=2e-2)
    lens = ExpertLens(model, VOCAB, make_token_embeddings(VOCAB, 24, DOMAIN_KEYWORDS))
    selected = lens.select_domain_experts(DOMAIN_KEYWORDS["ecommerce"])
    assert selected
    freeze_for_selected_experts(model, selected)
    assert any(parameter.requires_grad for expert_id in selected for parameter in model.experts[expert_id].parameters())
    ecommerce_batches = [make_toy_batch("ecommerce", n=32) for _ in range(2)]
    for _ in range(5):
        train_epoch(model, ecommerce_batches, lr=1e-2)
    assert accuracy(model, ecommerce_batches) >= 0.80


if __name__ == "__main__":
    test_expertlens_pipeline()
    print("ExpertLens reproduction test passed")

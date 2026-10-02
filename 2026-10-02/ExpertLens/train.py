from expertlens import ExpertLens, MultimodalMoE, accuracy, freeze_for_selected_experts, make_token_embeddings, train_epoch
from toy_data import DOMAIN_KEYWORDS, DOMAINS, VOCAB, make_toy_batch


def main() -> None:
    model = MultimodalMoE(input_dim=24, hidden_dim=24, num_experts=8, num_classes=len(DOMAINS))
    base_batches = [make_toy_batch(domain, n=48) for domain in DOMAINS for _ in range(4)]
    for _ in range(30):
        train_epoch(model, base_batches, lr=2e-2)
    token_embeddings = make_token_embeddings(VOCAB, 24, DOMAIN_KEYWORDS)
    lens = ExpertLens(model, VOCAB, token_embeddings)
    selected = lens.select_domain_experts(DOMAIN_KEYWORDS["ecommerce"])
    freeze_for_selected_experts(model, selected)
    ecommerce_batches = [make_toy_batch("ecommerce", n=64) for _ in range(6)]
    for _ in range(12):
        train_epoch(model, ecommerce_batches, lr=1e-2)
    print({
        "selected_experts": sorted(selected),
        "ecommerce_accuracy": accuracy(model, ecommerce_batches),
        "decoded_expert_0": lens.decode_expert(0).tokens,
    })


if __name__ == "__main__":
    main()

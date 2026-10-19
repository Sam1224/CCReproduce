import random
import torch
import torch.nn.functional as F

from data import build_vocab, bow, toy_cases
from model import NeuralRanker, RuleBasedEditor, initial_handbook


def train_ranker(model, cases, vocab, epochs=180):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        random.shuffle(cases)
        for case in cases:
            q = bow(case.query, vocab).repeat(len(case.candidates), 1)
            p = torch.stack([bow(product.as_text(), vocab) for product in case.candidates])
            scores = model(q, p).unsqueeze(0)
            target = torch.tensor([next(i for i, pdt in enumerate(case.candidates) if pdt.product_id == case.target_id)])
            loss = F.cross_entropy(scores, target)
            opt.zero_grad()
            loss.backward()
            opt.step()


def rank_position(ranking, target_id):
    return ranking.index(target_id)


def main():
    torch.manual_seed(7)
    cases = toy_cases()
    vocab = build_vocab(cases)
    ranker = NeuralRanker(len(vocab))
    train_ranker(ranker, cases, vocab)

    handbook = initial_handbook()
    editor = RuleBasedEditor()
    history = []
    for epoch in range(6):
        failures = []
        for case in cases:
            before = ranker.rank(case, vocab)
            before_pos = rank_position(before, case.target_id)
            target = next(p for p in case.candidates if p.product_id == case.target_id)
            rewritten, applied = editor.rewrite(target, case.query, handbook)
            rewritten_candidates = [rewritten if p.product_id == target.product_id else p for p in case.candidates]
            rewritten_case = type(case)(case.query, case.target_id, rewritten_candidates)
            after = ranker.rank(rewritten_case, vocab)
            after_pos = rank_position(after, case.target_id)
            gain = before_pos - after_pos
            handbook.update_stats(applied, gain)
            history.append({"epoch": epoch, "query": case.query, "applied": applied, "gain": gain})
            if gain < 0:
                failures.append(case)
        handbook.add_or_modify_from_failures(failures)

    print("GLOBE reproduction finished")
    print("Episodes:", len(history))
    print("Mean rank gain:", round(sum(h["gain"] for h in history) / len(history), 3))
    for b in handbook.behaviors.values():
        print(b.behavior_id, b.name, "active=", b.active, "h+/h-/h0=", (b.helpful, b.harmful, b.neutral))


if __name__ == "__main__":
    main()

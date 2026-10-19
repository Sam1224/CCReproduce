import torch
import torch.nn.functional as F

from data import toy_samples
from model import GraphAnalyzer, LongForgeSynthesizer, SupportScorer, bow, build_vocab


def train_support_scorer(model, samples, vocab, epochs=160):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    pairs = []
    for sample in samples:
        for assertion, label in zip(sample.answer_sentences, sample.gold_supported):
            evidence = " ".join(sample.document)
            pairs.append((evidence, assertion, float(label)))
    for _ in range(epochs):
        for evidence, assertion, label in pairs:
            logit = model(bow(evidence, vocab), bow(assertion, vocab))
            loss = F.binary_cross_entropy_with_logits(logit, torch.tensor(label))
            opt.zero_grad()
            loss.backward()
            opt.step()


def main():
    torch.manual_seed(11)
    samples = toy_samples()
    vocab = build_vocab(samples)
    scorer = SupportScorer(len(vocab))
    train_support_scorer(scorer, samples, vocab)

    analyzer = GraphAnalyzer(scorer, vocab)
    synthesizer = LongForgeSynthesizer(analyzer)
    accepted = synthesizer.filter_and_mix(samples)

    print("LongForge reproduction finished")
    for i, sample in enumerate(samples):
        metrics = analyzer.analyze(sample).metrics()
        print(f"sample={i}", metrics)
    print("accepted samples:", len(accepted))


if __name__ == "__main__":
    main()

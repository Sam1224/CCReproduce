from data import toy_samples
from model import GraphAnalyzer, LongForgeSynthesizer, SupportScorer, build_vocab
from train import train_support_scorer


def test_pipeline():
    samples = toy_samples()
    vocab = build_vocab(samples)
    scorer = SupportScorer(len(vocab))
    train_support_scorer(scorer, samples, vocab, epochs=5)
    analyzer = GraphAnalyzer(scorer, vocab)
    graph = analyzer.analyze(samples[0])
    metrics = graph.metrics()
    assert {"breadth", "depth", "correctness", "completeness"} <= metrics.keys()
    accepted = LongForgeSynthesizer(analyzer).filter_and_mix(samples, min_correctness=0.0, min_completeness=0.0)
    assert accepted
    print("LongForge smoke test passed")


if __name__ == "__main__":
    test_pipeline()

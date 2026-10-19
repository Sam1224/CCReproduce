from data import build_vocab, toy_cases
from model import NeuralRanker, RuleBasedEditor, initial_handbook
from train import train_ranker


def test_pipeline():
    cases = toy_cases()
    vocab = build_vocab(cases)
    ranker = NeuralRanker(len(vocab))
    train_ranker(ranker, cases, vocab, epochs=5)
    handbook = initial_handbook()
    editor = RuleBasedEditor()
    rewritten, applied = editor.rewrite(cases[0].candidates[0], cases[0].query, handbook)
    assert rewritten.as_text() != cases[0].candidates[0].as_text()
    assert applied
    assert isinstance(ranker.rank(cases[0], vocab), list)
    print("GLOBE smoke test passed")


if __name__ == "__main__":
    test_pipeline()

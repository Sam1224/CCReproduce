from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


@dataclass
class QAExample:
    question: str
    answer: str
    doc_id: str


DOCUMENTS = [
    Document(
        "policy_live_stream",
        "Live commerce misleading claim policy",
        "Creators must not claim impossible delivery speed or unsupported product effects. First violations require evidence review, repeated violations require traffic restriction and operator confirmation.",
    ),
    Document(
        "appeal_sop",
        "Creator appeal review SOP",
        "Appeal handling requires account identity verification, retrieval of original evidence, counter-evidence review, policy exception review, and a final operator confirmation before restoration.",
    ),
    Document(
        "quality_labeling",
        "Content quality labeling guideline",
        "Large-scale content labels should include category, risk type, evidence span, confidence, and reviewer notes. Ambiguous samples are routed to senior reviewers and reused as hard examples.",
    ),
]

QA = [
    QAExample("What is required before restoring a creator after appeal?", "identity verification, original evidence retrieval, counter-evidence review, exception review, and operator confirmation", "appeal_sop"),
    QAExample("What happens after repeated misleading claims?", "traffic restriction after evidence review and operator confirmation", "policy_live_stream"),
    QAExample("Which fields are used for content quality labels?", "category, risk type, evidence span, confidence, and reviewer notes", "quality_labeling"),
]


def tokenize(text: str) -> List[str]:
    return text.lower().replace(".", " ").replace(",", " ").replace("?", " ").split()


def build_vocab(texts: Iterable[str]) -> Dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for text in texts:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: Dict[str, int], max_len: int = 64) -> List[int]:
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokenize(text)][:max_len]
    return ids + [0] * (max_len - len(ids))


def build_iar_examples() -> Tuple[Dict[str, int], List[Tuple[str, str, str]]]:
    inject_pairs = []
    for doc in DOCUMENTS:
        words = doc.text.split()
        midpoint = max(4, len(words) // 2)
        inject_pairs.append(("continue: " + " ".join(words[:midpoint]), " ".join(words[midpoint:]), "inject-continuation"))
        inject_pairs.append(("rewrite policy note: " + doc.text, doc.title + " states " + doc.text, "inject-rewrite"))
        inject_pairs.append(("reconstruct from title: " + doc.title, doc.text, "inject-reconstruction"))
    align_pairs = [("answer: " + item.question, item.answer, "align-qa") for item in QA]
    recovery_pairs = [("general instruction: summarize carefully", "Use concise, faithful answers and avoid inventing facts.", "recover-general")]
    all_pairs = inject_pairs + align_pairs + recovery_pairs
    vocab = build_vocab([text for pair in all_pairs for text in pair[:2]])
    return vocab, all_pairs

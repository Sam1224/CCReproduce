import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset


@dataclass
class Vocabulary:
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]

    @classmethod
    def build(cls, samples: List[Dict[str, str]]) -> "Vocabulary":
        tokens = {"<pad>", "<bos>", "<eos>", "<unk>"}
        for item in samples:
            tokens.update(item["question"].lower().split())
            tokens.update(item["answer"].lower().split())
        ordered = sorted(tokens)
        token_to_id = {token: index for index, token in enumerate(ordered)}
        id_to_token = {index: token for token, index in token_to_id.items()}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    def encode(self, text: str, max_length: int) -> torch.Tensor:
        ids = [self.token_to_id.get(token, self.token_to_id["<unk>"]) for token in text.lower().split()]
        ids = ids[: max_length - 1] + [self.eos_id]
        ids += [self.pad_id] * (max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: List[int]) -> str:
        words = []
        for index in ids:
            token = self.id_to_token.get(int(index), "<unk>")
            if token == "<eos>":
                break
            if token not in {"<pad>", "<bos>"}:
                words.append(token)
        return " ".join(words)


def build_toy_samples() -> List[Dict[str, str]]:
    factual = [
        ("Which policy detects counterfeit product videos", "multimodal violation detection"),
        ("What module retrieves reliable creator evidence", "retrieval augmented generation"),
        ("How should duplicate shopping posts be grouped", "embedding clustering"),
        ("Which signal improves caption quality checks", "image text consistency"),
        ("What reduces noisy creator labels", "data cleaning"),
        ("Which model routes tokens through sparse experts", "mixture of experts"),
    ]
    hallucinated = [
        ("Which policy detects counterfeit product videos", "random celebrity ranking"),
        ("What module retrieves reliable creator evidence", "unverified rumor generation"),
        ("How should duplicate shopping posts be grouped", "manual weather forecast"),
        ("Which signal improves caption quality checks", "irrelevant sports score"),
        ("What reduces noisy creator labels", "unchecked synthetic claims"),
        ("Which model routes tokens through sparse experts", "single dense shortcut"),
    ]
    samples = []
    for question, answer in factual:
        samples.append({"question": question, "answer": answer, "is_factual": "1"})
    for question, answer in hallucinated:
        samples.append({"question": question, "answer": answer, "is_factual": "0"})
    random.Random(7).shuffle(samples)
    return samples


class ToyQADataset(Dataset):
    def __init__(self, samples: List[Dict[str, str]], vocab: Vocabulary, max_question_len: int = 12, max_answer_len: int = 5):
        self.samples = samples
        self.vocab = vocab
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.samples[index]
        return {
            "input_ids": self.vocab.encode(item["question"], self.max_question_len),
            "target_ids": self.vocab.encode(item["answer"], self.max_answer_len),
            "is_factual": torch.tensor(float(item["is_factual"]), dtype=torch.float),
        }

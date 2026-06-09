from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sample:
    input_text: str
    target_text: str
    credit_limit: int
    eligible: bool


def _render_profile(limit: int, eligible: bool) -> str:
    eligible_str = "yes" if eligible else "no"
    return f"profile:limit={limit};eligible={eligible_str}"


def build_dataset(*, n: int, seed: int) -> list[Sample]:
    rng = np.random.default_rng(seed)

    queries = [
        "我可以分期购买吗？",
        "我的信用额度是多少？",
        "我想分期，这单可以吗？",
        "我能不能用分期？额度够吗？",
        "我额度是多少，能分期吗？",
    ]

    tmpl_ok = [
        "可以的，你当前可用额度是{limit}元，支持分期购买。",
        "支持分期哦～你的额度为{limit}元，可以直接选择分期。",
        "没问题，额度{limit}元，满足分期条件。",
    ]
    tmpl_no = [
        "暂不支持分期，你当前可用额度是{limit}元。",
        "目前无法分期（额度{limit}元），建议更换支付方式。",
        "抱歉暂时不满足分期条件，你的额度为{limit}元。",
    ]

    samples: list[Sample] = []
    for _ in range(n):
        limit = int(rng.choice([1000, 1500, 2000, 2500, 3000, 4000, 5000, 8000]))
        eligible = bool(rng.random() < 0.65)

        q = str(rng.choice(queries))
        profile = _render_profile(limit, eligible)
        input_text = f"query:{q}\n{profile}"

        if eligible:
            target = str(rng.choice(tmpl_ok)).format(limit=limit)
        else:
            target = str(rng.choice(tmpl_no)).format(limit=limit)

        samples.append(Sample(input_text=input_text, target_text=target, credit_limit=limit, eligible=eligible))

    return samples


def build_vocab(samples: list[Sample]) -> dict[str, int]:
    chars = set()
    for s in samples:
        chars.update(list(s.input_text))
        chars.update(list(s.target_text))

    special = ["<pad>", "<bos>", "<eos>"]
    vocab = {tok: i for i, tok in enumerate(special)}

    for ch in sorted(chars):
        if ch not in vocab:
            vocab[ch] = len(vocab)

    return vocab


def encode(text: str, vocab: dict[str, int], *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
    ids = []
    if add_bos:
        ids.append(vocab["<bos>"])
    ids.extend(vocab[c] for c in text)
    if add_eos:
        ids.append(vocab["<eos>"])
    return ids


def decode(ids: list[int], inv_vocab: dict[int, str]) -> str:
    out = []
    for i in ids:
        ch = inv_vocab.get(i, "")
        if ch in ("<pad>", "<bos>", "<eos>"):
            continue
        out.append(ch)
    return "".join(out)

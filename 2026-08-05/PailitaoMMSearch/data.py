"""data.py

Toy 电商多模态检索数据：文本 + 图像张量 + 目标商品（item_id）。

设计原则：
- 数据必须足够简单，CPU 上几秒可训练；
- 同时要“长得像”真实电商多模态搜索：query 是 (text,image)，目标是从 catalog 中找到正确商品；
- 允许通过统计规律学到检索（颜色/品类/属性）。

注意（与论文差距）：
- 论文数据规模与分布远比 toy 复杂；真实系统会有更长文本、更高分辨率图像、点击/曝光等行为信号。
- 这里的图像只是张量合成，不涉及 PIL/真实图片读取。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Item:
    item_id: int
    title: str
    image: torch.Tensor  # [3, 32, 32]
    meta: Dict[str, str]


class SimpleTokenizer:
    """最小 tokenizer：按空格切分，建 vocab，pad 到固定长度。"""

    def __init__(self, vocab: Dict[str, int] | None = None):
        if vocab is None:
            vocab = {"<pad>": 0, "<unk>": 1}
        self.vocab = dict(vocab)
        self.inv_vocab = {i: w for w, i in self.vocab.items()}

    @property
    def pad_id(self) -> int:
        return self.vocab["<pad>"]

    def build(self, texts: List[str]) -> None:
        for t in texts:
            for w in t.strip().split():
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
        self.inv_vocab = {i: w for w, i in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        ids = []
        for w in text.strip().split():
            ids.append(self.vocab.get(w, self.vocab["<unk>"]))
        return ids

    def state_dict(self) -> Dict:
        return {"vocab": self.vocab}

    @classmethod
    def from_state_dict(cls, state: Dict) -> "SimpleTokenizer":
        return cls(vocab=state["vocab"])


def _make_base_image(color: str, ptype: str, size: int = 32) -> torch.Tensor:
    """合成一张 toy 图像。

    - color 控制 RGB 均值
    - ptype 控制空间纹理（条纹/对角线/中心块等）
    """

    # 颜色基底
    color_map = {
        "red": torch.tensor([1.0, 0.1, 0.1]),
        "green": torch.tensor([0.1, 1.0, 0.1]),
        "blue": torch.tensor([0.1, 0.1, 1.0]),
        "black": torch.tensor([0.05, 0.05, 0.05]),
        "white": torch.tensor([0.95, 0.95, 0.95]),
        "yellow": torch.tensor([0.9, 0.9, 0.1]),
    }
    base = color_map[color].view(3, 1, 1).repeat(1, size, size)

    # 形状纹理（用简单 mask 模拟“品类”）
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    yy = yy.float() / (size - 1)
    xx = xx.float() / (size - 1)

    if ptype == "shoe":
        mask = (xx > yy).float()  # 对角线
    elif ptype == "shirt":
        mask = ((yy * 10).floor() % 2).float()  # 横条纹
    elif ptype == "phone":
        mask = (((xx > 0.25) & (xx < 0.75) & (yy > 0.15) & (yy < 0.85))).float()
    elif ptype == "bag":
        mask = (((xx - 0.5) ** 2 + (yy - 0.5) ** 2) < 0.18).float()  # 圆
    else:
        mask = torch.ones(size, size)

    img = base * (0.6 + 0.4 * mask.unsqueeze(0))
    img = img + 0.03 * torch.randn_like(img)
    img = img.clamp(0.0, 1.0)
    return img


def build_toy_catalog(
    seed: int = 0,
    num_items: int = 200,
) -> List[Item]:
    """生成 toy 商品库。"""

    rng = np.random.RandomState(seed)

    colors = ["red", "green", "blue", "black", "white", "yellow"]
    ptypes = ["shoe", "shirt", "phone", "bag"]
    styles = ["sport", "casual", "classic", "mini", "pro"]

    items: List[Item] = []
    for item_id in range(num_items):
        c = colors[rng.randint(len(colors))]
        p = ptypes[rng.randint(len(ptypes))]
        s = styles[rng.randint(len(styles))]

        # 标题包含强信号（颜色/品类/风格）
        title = f"{c} {p} {s}"
        img = _make_base_image(c, p)
        items.append(Item(item_id=item_id, title=title, image=img, meta={"color": c, "ptype": p, "style": s}))

    return items


def build_toy_queries(
    catalog: List[Item],
    seed: int = 0,
    num_queries: int = 600,
    text_drop_prob: float = 0.3,
    image_noise: float = 0.08,
) -> List[Dict]:
    """从 catalog 采样 query。

    每个 query 对应一个 target_item_id。
    """

    rng = np.random.RandomState(seed + 17)
    queries: List[Dict] = []

    for _ in range(num_queries):
        item = catalog[rng.randint(len(catalog))]

        # query 文本：从 item.title 里随机丢词/加一点噪声词
        words = item.title.split()
        kept = []
        for w in words:
            if rng.rand() > text_drop_prob:
                kept.append(w)
        if len(kept) == 0:
            kept = [words[rng.randint(len(words))]]

        # 加一个弱噪声词（模拟 query rewrite/口语）
        if rng.rand() < 0.2:
            kept.append("nice")
        qtext = " ".join(kept)

        # query 图像：目标图像 + 噪声
        qimg = item.image + image_noise * torch.randn_like(item.image)
        qimg = qimg.clamp(0.0, 1.0)

        queries.append(
            {
                "query_text": qtext,
                "query_image": qimg,
                "target_item_id": item.item_id,
            }
        )

    return queries


def build_toy_mmsearch_data(
    seed: int = 0,
    num_items: int = 200,
    num_queries: int = 600,
    train_ratio: float = 0.8,
) -> Tuple[List[Item], List[Dict], List[Dict], SimpleTokenizer]:
    """构造 catalog + train/test queries + tokenizer。"""

    catalog = build_toy_catalog(seed=seed, num_items=num_items)
    queries = build_toy_queries(catalog=catalog, seed=seed, num_queries=num_queries)

    # tokenizer 用 catalog title + query text 建 vocab
    tok = SimpleTokenizer()
    all_texts = [it.title for it in catalog] + [q["query_text"] for q in queries]
    tok.build(all_texts)

    n_train = int(len(queries) * train_ratio)
    train_q = queries[:n_train]
    test_q = queries[n_train:]
    return catalog, train_q, test_q, tok


class ToyMMSearchDataset(Dataset):
    """(query -> target item) 的训练/测试 dataset。

    返回字段：
    - query_input_ids: LongTensor [L]
    - query_image: FloatTensor [3, 32, 32]
    - pos_input_ids: LongTensor [L]
    - pos_image: FloatTensor [3, 32, 32]
    - target_item_id: LongTensor []

    注意：这里把正样本 item 的文本/图像也直接返回，用于双塔对齐训练。
    """

    def __init__(
        self,
        catalog: List[Item],
        queries: List[Dict],
        tokenizer: SimpleTokenizer,
        max_len: int = 12,
    ):
        self.catalog = catalog
        self.queries = queries
        self.tok = tokenizer
        self.max_len = max_len

        self._item_by_id = {it.item_id: it for it in catalog}

    def __len__(self) -> int:
        return len(self.queries)

    def _encode_pad(self, text: str) -> torch.Tensor:
        ids = self.tok.encode(text)[: self.max_len]
        if len(ids) < self.max_len:
            ids = ids + [self.tok.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict:
        q = self.queries[idx]
        item = self._item_by_id[int(q["target_item_id"])]

        return {
            "query_input_ids": self._encode_pad(q["query_text"]),
            "query_image": q["query_image"].float(),
            "pos_input_ids": self._encode_pad(item.title),
            "pos_image": item.image.float(),
            "target_item_id": torch.tensor(item.item_id, dtype=torch.long),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    # 已经 pad 过，这里直接 stack
    return {
        "query_input_ids": torch.stack([b["query_input_ids"] for b in batch], dim=0),
        "query_image": torch.stack([b["query_image"] for b in batch], dim=0),
        "pos_input_ids": torch.stack([b["pos_input_ids"] for b in batch], dim=0),
        "pos_image": torch.stack([b["pos_image"] for b in batch], dim=0),
        "target_item_id": torch.stack([b["target_item_id"] for b in batch], dim=0),
    }

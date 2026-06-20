"""Toy multimodal IE dataset for SAMA (MNER / MRE / MEE).

Each sample is a text-image pair with task-specific ground-truth labels:
  * MNER: entity spans + entity types  (PER/ORG/LOC/MISC)
  * MRE : (subject, relation, object)  triple
  * MEE : event type (trigger) + arguments (role: arg)

Images are tiny synthetic tensors [3,H,W]. The top-left ANCHOR patch encodes the
subject entity's identity (a colour keyed by entity id); the rest is background.
This makes the Anchor-Preserving Diffusion mask M meaningful: anchor pixels are
kept, background is diversified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

IMG_C, IMG_H, IMG_W = 3, 8, 8
ANCHOR_HW = 4                      # top-left ANCHOR_HW x ANCHOR_HW patch = entity region

# --- small symbolic vocab -----------------------------------------------------
ENTITIES = {  # name -> (type, entity_id used to colour the anchor patch)
    "lebron": ("PER", 1), "messi": ("PER", 2), "einstein": ("PER", 3),
    "lakers": ("ORG", 4), "google": ("ORG", 5), "uestc": ("ORG", 6),
    "paris": ("LOC", 7), "tokyo": ("LOC", 8),
    "trophy": ("MISC", 9), "ball": ("MISC", 10),
}
ENTITY_TYPES = ["PER", "ORG", "LOC", "MISC"]
RELATIONS = ["member_of", "located_in", "plays_for", "win"]
EVENTS = ["clinching", "meeting", "award"]
ARG_ROLES = ["Agent", "Target"]

TASKS = ["MNER", "MRE", "MEE"]


@dataclass
class Sample:
    task: str
    tokens: List[str]                       # the original sentence
    image: torch.Tensor                     # [3,H,W]
    entities: List[Tuple[str, str]] = field(default_factory=list)   # (word,type)
    relation: Tuple[str, str, str] = None   # (subj, rel, obj)
    event: Tuple[str, str, List[Tuple[str, str]]] = None  # (etype, trigger, [(role,arg)])
    gold_type: str = ""                     # unified downstream label


def _make_image(entity_id: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    img = torch.rand(IMG_C, IMG_H, IMG_W, generator=g) * 0.4        # background
    # anchor patch colour = deterministic function of entity id
    colour = torch.tensor([(entity_id % 3) / 3.0,
                           ((entity_id // 3) % 3) / 3.0,
                           ((entity_id // 9) % 3) / 3.0 + 0.4])
    img[:, :ANCHOR_HW, :ANCHOR_HW] = colour.view(3, 1, 1)
    return img.clamp(0, 1)


def anchor_mask() -> torch.Tensor:
    """Binary preservation mask M: 1 on the anchor (entity) region, 0 background."""
    m = torch.zeros(1, IMG_H, IMG_W)
    m[:, :ANCHOR_HW, :ANCHOR_HW] = 1.0
    return m


def _build():
    samples: List[Sample] = []
    sid = 0
    # MNER samples
    mner_specs = [
        (["lebron", "scored", "for", "lakers"], [("lebron", "PER")], "PER"),
        (["messi", "joined", "google"], [("messi", "PER")], "PER"),
        (["lakers", "are", "in", "paris"], [("lakers", "ORG")], "ORG"),
        (["einstein", "visited", "tokyo"], [("einstein", "PER")], "PER"),
        (["google", "opened", "in", "paris"], [("google", "ORG")], "ORG"),
        (["uestc", "is", "in", "tokyo"], [("uestc", "ORG")], "ORG"),
    ]
    for toks, ents, gt in mner_specs:
        eid = ENTITIES[ents[0][0]][1]
        samples.append(Sample("MNER", toks, _make_image(eid, sid), entities=ents, gold_type=gt)); sid += 1
    # MRE samples
    mre_specs = [
        (["lebron", "plays", "for", "lakers"], ("lebron", "plays_for", "lakers"), "plays_for"),
        (["messi", "is", "member", "of", "google"], ("messi", "member_of", "google"), "member_of"),
        (["lakers", "located", "in", "paris"], ("lakers", "located_in", "paris"), "located_in"),
        (["lakers", "win", "trophy"], ("lakers", "win", "trophy"), "win"),
        (["uestc", "located", "in", "tokyo"], ("uestc", "located_in", "tokyo"), "located_in"),
        (["einstein", "member", "of", "uestc"], ("einstein", "member_of", "uestc"), "member_of"),
    ]
    for toks, rel, gt in mre_specs:
        eid = ENTITIES[rel[0]][1]
        samples.append(Sample("MRE", toks, _make_image(eid, sid), relation=rel, gold_type=gt)); sid += 1
    # MEE samples
    mee_specs = [
        (["lebron", "clinching", "the", "trophy"], ("clinching", "clinching",
         [("Agent", "lebron"), ("Target", "trophy")]), "clinching"),
        (["messi", "meeting", "einstein"], ("meeting", "meeting",
         [("Agent", "messi"), ("Target", "einstein")]), "meeting"),
        (["google", "award", "to", "uestc"], ("award", "award",
         [("Agent", "google"), ("Target", "uestc")]), "award"),
        (["lakers", "clinching", "the", "ball"], ("clinching", "clinching",
         [("Agent", "lakers"), ("Target", "ball")]), "clinching"),
        (["einstein", "meeting", "messi"], ("meeting", "meeting",
         [("Agent", "einstein"), ("Target", "messi")]), "meeting"),
        (["uestc", "award", "to", "google"], ("award", "award",
         [("Agent", "uestc"), ("Target", "google")]), "award"),
    ]
    for toks, ev, gt in mee_specs:
        eid = ENTITIES[ev[2][0][1]][1]
        samples.append(Sample("MEE", toks, _make_image(eid, sid), event=ev, gold_type=gt)); sid += 1
    return samples


SAMPLES = _build()

# unified downstream label space across all three tasks
GOLD_LABELS = ENTITY_TYPES + RELATIONS + EVENTS
GOLD2ID = {g: i for i, g in enumerate(GOLD_LABELS)}


# --- vocabulary (words + anchor markup tokens) --------------------------------
def build_vocab(samples: List[Sample]):
    toks = set()
    for s in samples:
        toks.update(s.tokens)
    # markup tokens used by the anchor strings
    markup = ["<PER>", "</PER>", "<ORG>", "</ORG>", "<LOC>", "</LOC>",
              "<MISC>", "</MISC>", "<r>", "</r>", "<ev>", "</ev>",
              "(", ")", ",", "[", "]", ":", "Trigger", "Arg", "Subject", "Object"]
    markup += RELATIONS + EVENTS + ARG_ROLES
    vocab = ["<pad>", "<bos>", "<eos>", "<unk>"] + sorted(toks) + markup
    stoi = {t: i for i, t in enumerate(vocab)}
    return vocab, stoi


VOCAB, STOI = build_vocab(SAMPLES)


def encode(tokens: List[str]) -> List[int]:
    return [STOI.get(t, STOI["<unk>"]) for t in tokens]


def split_train_test(low_resource_frac: float = 0.5):
    """Per-task split; train uses only `frac` of samples (low-resource setting)."""
    train, test = [], []
    for t in TASKS:
        st = [s for s in SAMPLES if s.task == t]
        k = max(1, int(round(len(st) * low_resource_frac)))
        train += st[:k]
        test += st[k:]
    return train, test

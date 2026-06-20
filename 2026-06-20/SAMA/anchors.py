"""Semantic Anchor Construction (SAMA Sec III.C).

Re-encode ground-truth labels into structured inline-tagged anchor strings. These
inline tags act as a hard constraint that minimises H(Y|X) during decoding:
  * Entity (MNER):  <PER> lebron </PER>
  * Relation (MRE): <r> ( subject , plays_for , object ) </r>
  * Event (MEE):    <ev> clinching Trigger [ Arg : Agent lebron ] ... </ev>
"""
from __future__ import annotations

from typing import List

from data import Sample, encode

TYPE_OPEN = {"PER": "<PER>", "ORG": "<ORG>", "LOC": "<LOC>", "MISC": "<MISC>"}
TYPE_CLOSE = {"PER": "</PER>", "ORG": "</ORG>", "LOC": "</LOC>", "MISC": "</MISC>"}


def anchor_tokens(sample: Sample) -> List[str]:
    """Return the inline-tagged anchor string as a token list."""
    if sample.task == "MNER":
        out = []
        for word, typ in sample.entities:
            out += [TYPE_OPEN[typ], word, TYPE_CLOSE[typ]]
        return out
    if sample.task == "MRE":
        s, r, o = sample.relation
        return ["<r>", "(", "Subject", s, ",", r, ",", "Object", o, ")", "</r>"]
    if sample.task == "MEE":
        etype, trigger, args = sample.event
        out = ["<ev>", etype, "Trigger"]
        for role, arg in args:
            out += ["[", "Arg", ":", role, arg, "]"]
        out += ["</ev>"]
        return out
    raise ValueError(sample.task)


def anchor_ids(sample: Sample) -> List[int]:
    return encode(anchor_tokens(sample))


def entity_anchor_words(sample: Sample) -> List[str]:
    """Words to emphasise in the diffusion prompt (Eq. 10, the A_entity terms)."""
    if sample.task == "MNER":
        return [w for w, _ in sample.entities]
    if sample.task == "MRE":
        return [sample.relation[0], sample.relation[2]]
    if sample.task == "MEE":
        return [a for _, a in sample.event[2]]
    return []

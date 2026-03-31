from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass
class Doc:
    doc_id: str
    text: str
    entities: List[str]


class KnowledgeBase:
    def __init__(self):
        self.docs: Dict[str, Doc] = {}
        self.graph: Dict[str, Set[str]] = {}

    def add_doc(self, doc: Doc) -> None:
        self.docs[doc.doc_id] = doc
        ents = [e.strip() for e in doc.entities if e.strip()]
        for e in ents:
            self.graph.setdefault(e, set())
        # co-occurrence edges (undirected)
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                a, b = ents[i], ents[j]
                self.graph[a].add(b)
                self.graph[b].add(a)

    def neighbors(self, entity: str) -> Set[str]:
        return self.graph.get(entity, set())


def tokenize(text: str) -> List[str]:
    return [t.strip(".,!?()[]{}\"'").lower() for t in text.split() if t.strip()]


def make_toy_kb() -> KnowledgeBase:
    kb = KnowledgeBase()
    kb.add_doc(
        Doc(
            doc_id="d1",
            text="Creator A sells skincare product X. The product is promoted in a short video.",
            entities=["CreatorA", "Skincare", "ProductX", "ShortVideo"],
        )
    )
    kb.add_doc(
        Doc(
            doc_id="d2",
            text="Product X has ingredients list and compliance policy tags. Policy forbids misleading claims.",
            entities=["ProductX", "Policy", "Compliance"],
        )
    )
    kb.add_doc(
        Doc(
            doc_id="d3",
            text="Creator B posts a livestream about discount deals. The content includes medical claims.",
            entities=["CreatorB", "Livestream", "MedicalClaim", "Discount"],
        )
    )
    kb.add_doc(
        Doc(
            doc_id="d4",
            text="Governance system uses retrieval + rules to detect violations and trigger review.",
            entities=["Governance", "Retrieval", "Violation", "Review"],
        )
    )
    return kb

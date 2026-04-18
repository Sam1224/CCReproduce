from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    tags: list[str]


@dataclass
class Node:
    node_id: str
    title: str
    summary: str
    doc_ids: list[str]
    children: list[Node]

    def to_dict(self) -> dict:
        return {
            **{k: v for k, v in asdict(self).items() if k not in {"children"}},
            "children": [c.to_dict() for c in self.children],
        }


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_corpus(path: Path) -> list[Document]:
    docs: list[Document] = []
    for row in _iter_jsonl(path):
        docs.append(
            Document(
                doc_id=row["doc_id"],
                title=row["title"],
                text=row["text"],
                tags=list(row.get("tags", [])),
            )
        )
    return docs


def _keyword_summary(vectorizer: TfidfVectorizer, doc_texts: list[str], top_k: int = 8) -> str:
    x = vectorizer.transform(doc_texts)
    mean_tfidf = np.asarray(x.mean(axis=0)).reshape(-1)
    top_idx = mean_tfidf.argsort()[::-1][:top_k]
    vocab = np.array(vectorizer.get_feature_names_out())
    keywords = [vocab[i] for i in top_idx if mean_tfidf[i] > 0]
    return ", ".join(keywords[:top_k]) if keywords else "(no keywords)"


def _make_node_title(documents: list[Document], vectorizer: TfidfVectorizer) -> str:
    summary_kw = _keyword_summary(vectorizer, [d.title + "\n" + d.text for d in documents], top_k=5)
    return f"Cluster: {summary_kw}"


def _build_tree(
    *,
    node_id: str,
    documents: list[Document],
    vectorizer: TfidfVectorizer,
    max_depth: int,
    branching_factor: int,
    max_leaf_size: int,
    depth: int,
) -> Node:
    doc_texts = [d.title + "\n" + d.text for d in documents]
    summary_kw = _keyword_summary(vectorizer, doc_texts, top_k=10)

    title = _make_node_title(documents, vectorizer)
    summary = (
        f"This node groups {len(documents)} documents. "
        f"Representative keywords: {summary_kw}. "
        f"Example titles: " + "; ".join(d.title for d in documents[:3])
    )

    if depth >= max_depth or len(documents) <= max_leaf_size:
        return Node(node_id=node_id, title=title, summary=summary, doc_ids=[d.doc_id for d in documents], children=[])

    x = vectorizer.transform(doc_texts)
    k = min(branching_factor, len(documents))
    if k <= 1:
        return Node(node_id=node_id, title=title, summary=summary, doc_ids=[d.doc_id for d in documents], children=[])

    km = KMeans(n_clusters=k, random_state=0, n_init="auto")
    labels = km.fit_predict(x)

    children: list[Node] = []
    for cluster_idx in range(k):
        cluster_docs = [d for d, lab in zip(documents, labels, strict=True) if lab == cluster_idx]
        if not cluster_docs:
            continue
        child_id = f"{node_id}-{cluster_idx:02d}"
        children.append(
            _build_tree(
                node_id=child_id,
                documents=cluster_docs,
                vectorizer=vectorizer,
                max_depth=max_depth,
                branching_factor=branching_factor,
                max_leaf_size=max_leaf_size,
                depth=depth + 1,
            )
        )

    return Node(node_id=node_id, title=title, summary=summary, doc_ids=[d.doc_id for d in documents], children=children)


def _write_docs(docs: list[Document], docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        (docs_dir / f"{doc.doc_id}.md").write_text(
            f"# {doc.title}\n\n" + doc.text.strip() + "\n",
            encoding="utf-8",
        )


def _write_skill_files(node: Node, skills_dir: Path) -> None:
    """Emit SKILL.md / INDEX.md following the paper’s filesystem-skill story."""

    node_dir = skills_dir / node.node_id
    node_dir.mkdir(parents=True, exist_ok=True)

    # Root-like node uses SKILL.md, internal grouping nodes use INDEX.md.
    md_name = "SKILL.md" if node.node_id.count("-") <= 1 else "INDEX.md"
    children_list = "\n".join(f"- `{c.node_id}/` — {c.title}" for c in node.children) if node.children else "(leaf)"
    doc_list = "\n".join(f"- `{doc_id}`" for doc_id in node.doc_ids[:20]) if node.doc_ids else "(none)"

    content = (
        f"# {node.title}\n\n"
        f"{node.summary}\n\n"
        f"## Children\n{children_list}\n\n"
        f"## Documents (IDs)\n{doc_list}\n"
    )
    (node_dir / md_name).write_text(content, encoding="utf-8")

    for child in node.children:
        _write_skill_files(child, skills_dir)


def _compute_node_summary_embeddings(node: Node, vectorizer: TfidfVectorizer) -> dict[str, list[float]]:
    """Compute TF-IDF embeddings for each node summary for routing."""
    out: dict[str, list[float]] = {}

    def walk(n: Node) -> None:
        vec = vectorizer.transform([n.summary]).toarray()[0]
        out[n.node_id] = vec.tolist()
        for c in n.children:
            walk(c)

    walk(node)
    return out


def build_skill_tree(
    *,
    corpus_path: Path,
    out_dir: Path,
    max_depth: int = 2,
    branching_factor: int = 3,
    max_leaf_size: int = 4,
) -> None:
    docs = _load_corpus(corpus_path)
    texts = [d.title + "\n" + d.text for d in docs]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=2000,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w[\w\-]+\b",
    )
    vectorizer.fit(texts)

    root = _build_tree(
        node_id="skill-00",
        documents=docs,
        vectorizer=vectorizer,
        max_depth=max_depth,
        branching_factor=branching_factor,
        max_leaf_size=max_leaf_size,
        depth=0,
    )

    skills_dir = out_dir / "skills"
    docs_dir = out_dir / "docs"

    _write_docs(docs, docs_dir)
    _write_skill_files(root, skills_dir)

    index = {
        "root": root.to_dict(),
        "node_embeddings": _compute_node_summary_embeddings(root, vectorizer),
    }

    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    joblib.dump(vectorizer, out_dir / "vectorizer.joblib")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])

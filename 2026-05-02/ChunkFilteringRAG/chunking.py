from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def chunk_text(text: str, *, doc_id: str, chunk_size: int = 80, overlap: int = 20) -> List[Chunk]:
    """Simple whitespace chunking with overlap.

    This intentionally introduces redundancy when overlap > 0.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    tokens = text.split()
    chunks: List[Chunk] = []

    start = 0
    chunk_index = 0
    stride = chunk_size - overlap
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        chunk_tokens = tokens[start:end]
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::chunk_{chunk_index}",
                doc_id=doc_id,
                text=" ".join(chunk_tokens),
            )
        )
        if end == len(tokens):
            break
        start += stride
        chunk_index += 1

    return chunks


def chunk_corpus(docs: Iterable[tuple[str, str]], *, chunk_size: int = 80, overlap: int = 20) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for doc_id, doc_text in docs:
        all_chunks.extend(chunk_text(doc_text, doc_id=doc_id, chunk_size=chunk_size, overlap=overlap))
    return all_chunks

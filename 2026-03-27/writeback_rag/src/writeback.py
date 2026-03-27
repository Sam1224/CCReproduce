from __future__ import annotations

from typing import List


class EvidenceDistiller:
    def distill(self, question: str, answer: str, evidence_docs: List[str]) -> str:
        """Distill evidence into a compact knowledge unit.

        Paper behavior:
          - An LLM distills + fuses evidence across multiple documents.
          - Output is self-contained and optimized for future retrieval.

        This toy implementation is extractive: it keeps only the evidence-bearing
        sentences and formats them as a compact note.

        TODO (realistic distiller):
          - Use a small summarization model (e.g., T5/BART) or an instruction-tuned LLM.
          - Prompt with: question + all evidence docs; request a short factual paragraph.
          - Ensure the output is entity-anchored and avoids introducing new facts.
        """

        key = answer.split(";")[0].strip().lower()
        sentences: list[str] = []

        for doc in evidence_docs:
            for raw in doc.split("."):
                s = raw.strip()
                if not s:
                    continue
                if key in s.lower() or any(tok in s.lower() for tok in key.split()):
                    sentences.append(s)

        if not sentences:
            sentences = [d.strip() for d in evidence_docs if d.strip()]

        fused = ". ".join(sentences)
        fused = fused.strip(" .")

        return (
            f"[WRITEBACK UNIT]\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Evidence: {fused}.\n"
        )

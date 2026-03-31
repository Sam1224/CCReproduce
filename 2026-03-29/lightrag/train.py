from __future__ import annotations

import os
from pathlib import Path

from kb import Doc, make_toy_kb
from retriever import retrieve


def show(kb, q: str) -> None:
    res = retrieve(kb, q)
    print(f"\nQ: {q}")
    for r in res:
        doc = kb.docs[r.doc_id]
        print(f"  - {r.doc_id} score={r.score:.3f} via={r.why} ents={doc.entities}")


def main() -> None:
    kb = make_toy_kb()

    show(kb, "Is CreatorA allowed to claim medical effects for ProductX under Policy?")
    show(kb, "Which content types are linked to governance review?")

    # incremental update: add a new doc that connects CreatorA with MedicalClaim
    kb.add_doc(
        Doc(
            doc_id="d5",
            text="CreatorA previously received a warning for MedicalClaim violations in Livestream.",
            entities=["CreatorA", "MedicalClaim", "Livestream", "Violation"],
        )
    )

    show(kb, "Any prior MedicalClaim risk for CreatorA?")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()

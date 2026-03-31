from __future__ import annotations

import os
from pathlib import Path

from kb import make_toy_kb
from retriever import retrieve


def main() -> None:
    kb = make_toy_kb()
    res = retrieve(kb, "Policy for ProductX")
    assert res and isinstance(res[0].doc_id, str)
    print("lightrag smoke test ok")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()

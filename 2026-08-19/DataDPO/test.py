from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "artifacts"
    res = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

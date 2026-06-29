from __future__ import annotations

import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    base = Path(__file__).resolve().parent
    _run(["python", str(base / "train.py")])
    _run(["python", str(base / "eval.py")])


if __name__ == "__main__":
    main()

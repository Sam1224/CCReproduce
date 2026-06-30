from __future__ import annotations

import subprocess
import sys


def main() -> None:
    py = sys.executable
    subprocess.check_call([py, "train.py"])
    subprocess.check_call([py, "eval.py"])


if __name__ == "__main__":
    main()

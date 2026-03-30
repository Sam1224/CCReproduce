from __future__ import annotations

import subprocess
import sys


def main() -> None:
    proc = subprocess.run([sys.executable, 'train.py'], check=False)
    raise SystemExit(proc.returncode)


if __name__ == '__main__':
    main()

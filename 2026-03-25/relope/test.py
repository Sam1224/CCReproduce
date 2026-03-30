from __future__ import annotations

import subprocess
import sys


def main() -> None:
    # Just a smoke import + quick train for a few steps.
    # We run train.py directly (it prints progress).
    cmd = [sys.executable, 'train.py']
    proc = subprocess.run(cmd, check=False)
    raise SystemExit(proc.returncode)


if __name__ == '__main__':
    main()

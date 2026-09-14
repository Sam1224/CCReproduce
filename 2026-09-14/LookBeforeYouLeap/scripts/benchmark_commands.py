from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from look_before_you_leap.command_verifier import verify_shell_command

DATA = ROOT / "data"


def _load(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.strip().startswith("#")]


def main() -> None:
    valid_cmds = _load(DATA / "commands_valid.txt")
    invalid_cmds = _load(DATA / "commands_invalid.txt")

    tp = fp = tn = fn = 0

    for cmd in valid_cmds:
        r = verify_shell_command(cmd)
        if r.ok:
            tp += 1
        else:
            fn += 1
            print(f"[FN] {cmd} -> {r.reason} (unknown={r.unknown_flags})")

    for cmd in invalid_cmds:
        r = verify_shell_command(cmd)
        if r.ok:
            fp += 1
            print(f"[FP] {cmd} accepted")
        else:
            tn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    print("\n=== Summary ===")
    print(f"total={total}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"precision={precision:.3f} recall={recall:.3f}")


if __name__ == "__main__":
    main()

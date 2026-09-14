from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from look_before_you_leap.edit_verifier import LineNumberEdit, apply_anchor_and_verify, apply_line_number


def _write_demo_file(path: Path, shift: int) -> None:
    lines = [
        "def add(a, b):",
        "    return a + b",
        "",
        "def mul(a, b):",
        "    return a * b",
        "",
        "def main():",
        "    print(add(1, 2))",
        "    print(mul(2, 3))",
        "",
        "if __name__ == '__main__':",
        "    main()",
    ]

    if shift > 0:
        for _ in range(shift):
            lines.insert(0, "# inserted line")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    # Target: replace the return line in add() (line 2 in the unshifted file).
    intended_line_no = 2
    intended_new = "    return a + b + 1"

    trials = 200
    silent_misapply_location = 0
    silent_misapply_anchor = 0

    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        file_rel = "demo.py"
        file_path = repo_root / file_rel

        for i in range(trials):
            shift = 1 if (i % 2 == 0) else 0
            _write_demo_file(file_path, shift=shift)

            original = file_path.read_text(encoding="utf-8").splitlines()

            # Define anchors around the intended line (in the unshifted file).
            # In the shifted file, these anchors move with the content.
            anchor_before = "def add(a, b):"
            anchor_after = ""

            edit = LineNumberEdit(
                path=file_rel,
                line_no_1based=intended_line_no,
                new_line=intended_new,
                anchor_before=anchor_before,
                anchor_after=anchor_after,
            )

            # 1) Location-anchored apply: always applies if line exists.
            apply_line_number(edit, repo_root=str(repo_root))
            after_loc = file_path.read_text(encoding="utf-8").splitlines()
            # Silent misapply if we changed the wrong line (i.e., the original intended line content still exists).
            # Intended original line is "    return a + b".
            if "    return a + b" in after_loc:
                silent_misapply_location += 1

            # reset file
            _write_demo_file(file_path, shift=shift)

            # 2) Anchor-and-verify: refuses when anchors don't match at the stated location.
            r = apply_anchor_and_verify(edit, repo_root=str(repo_root))
            after_anchor = file_path.read_text(encoding="utf-8").splitlines()

            if r.applied and ("    return a + b" in after_anchor):
                silent_misapply_anchor += 1

        print("=== Edit Benchmark (toy) ===")
        print(f"trials={trials}")
        print(f"silent_misapply_location={silent_misapply_location}")
        print(f"silent_misapply_anchor_and_verify={silent_misapply_anchor}")


if __name__ == "__main__":
    main()

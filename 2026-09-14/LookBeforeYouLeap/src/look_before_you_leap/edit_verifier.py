from __future__ import annotations

import dataclasses
import difflib
from pathlib import Path


@dataclasses.dataclass
class EditApplyResult:
    ok: bool
    reason: str
    applied: bool


@dataclasses.dataclass
class SearchReplaceEdit:
    path: str
    old: str
    new: str


@dataclasses.dataclass
class LineNumberEdit:
    path: str
    line_no_1based: int
    new_line: str
    # Optional anchor context for safe application
    anchor_before: str | None = None
    anchor_after: str | None = None


@dataclasses.dataclass
class UnifiedDiffEdit:
    path: str
    unified_diff_text: str


def apply_search_replace(edit: SearchReplaceEdit, repo_root: str) -> EditApplyResult:
    file_path = Path(repo_root) / edit.path
    text = file_path.read_text(encoding="utf-8")

    occurrences = text.count(edit.old)
    if occurrences == 0:
        return EditApplyResult(ok=False, reason="anchor not found", applied=False)
    if occurrences > 1:
        return EditApplyResult(ok=False, reason="anchor not unique", applied=False)

    new_text = text.replace(edit.old, edit.new)
    file_path.write_text(new_text, encoding="utf-8")
    return EditApplyResult(ok=True, reason="applied", applied=True)


def apply_line_number(edit: LineNumberEdit, repo_root: str) -> EditApplyResult:
    file_path = Path(repo_root) / edit.path
    lines = file_path.read_text(encoding="utf-8").splitlines(keepends=False)

    idx = edit.line_no_1based - 1
    if idx < 0 or idx >= len(lines):
        return EditApplyResult(ok=False, reason="line out of range", applied=False)

    lines[idx] = edit.new_line
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return EditApplyResult(ok=True, reason="applied (location-anchored)", applied=True)


def apply_anchor_and_verify(edit: LineNumberEdit, repo_root: str) -> EditApplyResult:
    """A safer line-number edit: verify anchors before applying.

    This approximates the paper's "anchor-and-verify" idea.
    """

    file_path = Path(repo_root) / edit.path
    lines = file_path.read_text(encoding="utf-8").splitlines(keepends=False)

    idx = edit.line_no_1based - 1
    if idx < 0 or idx >= len(lines):
        return EditApplyResult(ok=False, reason="line out of range", applied=False)

    if edit.anchor_before is not None:
        if idx - 1 < 0 or lines[idx - 1] != edit.anchor_before:
            return EditApplyResult(ok=False, reason="anchor_before mismatch", applied=False)

    if edit.anchor_after is not None:
        if idx + 1 >= len(lines) or lines[idx + 1] != edit.anchor_after:
            return EditApplyResult(ok=False, reason="anchor_after mismatch", applied=False)

    lines[idx] = edit.new_line
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return EditApplyResult(ok=True, reason="applied (anchor-and-verify)", applied=True)


def apply_unified_diff(edit: UnifiedDiffEdit, repo_root: str) -> EditApplyResult:
    """Apply a unified diff with strict context matching.

    This uses difflib to compute patch application in a conservative way.
    For a production system, use a dedicated patch library.
    """

    file_path = Path(repo_root) / edit.path
    original = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
    diff_lines = edit.unified_diff_text.splitlines(keepends=True)

    # Very conservative: reconstruct target from diff, but only if context lines are consistent.
    # We apply by using difflib.restore on ndiff format by converting is nontrivial.
    # Here we fallback to refusing when diff is not a simple single-hunk replace.

    hunks = [ln for ln in diff_lines if ln.startswith("@@")]
    if len(hunks) != 1:
        return EditApplyResult(ok=False, reason="unsupported: need exactly one hunk", applied=False)

    # Parse hunk header @@ -l,s +l,s @@
    header = hunks[0]
    try:
        left = header.split("@@")[1].strip().split(" ")[0]
        start_old = int(left.split(",")[0].lstrip("-"))
    except Exception:
        return EditApplyResult(ok=False, reason="cannot parse hunk header", applied=False)

    # Extract hunk body
    hunk_start = diff_lines.index(header) + 1
    hunk_body = diff_lines[hunk_start:]
    # stop at next file header if present
    for i, ln in enumerate(hunk_body):
        if ln.startswith("diff --git") or ln.startswith("--- ") or ln.startswith("+++ "):
            hunk_body = hunk_body[:i]
            break

    old_cursor = start_old - 1
    new_lines: list[str] = []

    # Copy lines before hunk
    new_lines.extend(original[:old_cursor])

    for ln in hunk_body:
        if ln.startswith("@@"):
            break
        if ln.startswith(" "):
            # context
            ctx = ln[1:]
            if old_cursor >= len(original) or original[old_cursor] != ctx:
                return EditApplyResult(ok=False, reason="context mismatch", applied=False)
            new_lines.append(ctx)
            old_cursor += 1
        elif ln.startswith("-"):
            # deletion
            removed = ln[1:]
            if old_cursor >= len(original) or original[old_cursor] != removed:
                return EditApplyResult(ok=False, reason="deletion mismatch", applied=False)
            old_cursor += 1
        elif ln.startswith("+"):
            new_lines.append(ln[1:])
        else:
            # unexpected
            continue

    # Copy remaining lines
    new_lines.extend(original[old_cursor:])

    file_path.write_text("".join(new_lines), encoding="utf-8")
    return EditApplyResult(ok=True, reason="applied (diff)", applied=True)

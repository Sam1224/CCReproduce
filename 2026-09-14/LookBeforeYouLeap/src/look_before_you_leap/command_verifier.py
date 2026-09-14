from __future__ import annotations

import dataclasses
import shlex
import shutil
import subprocess
from functools import lru_cache
from typing import Iterable


@dataclasses.dataclass
class VerificationResult:
    ok: bool
    reason: str
    binary: str | None = None
    normalized: str | None = None
    unknown_flags: list[str] = dataclasses.field(default_factory=list)


def _iter_flags(argv: list[str]) -> Iterable[str]:
    """Heuristic flag extraction.

    - Stops parsing flags after `--`.
    - Treats tokens starting with '-' as flags.
    - Splits short bundles like `-al` into `-a`, `-l`.
    """

    for token in argv[1:]:
        if token == "--":
            return
        if not token.startswith("-") or token == "-":
            continue
        if token.startswith("--"):
            # long flag like --color=auto
            flag = token.split("=", 1)[0]
            yield flag
            continue

        # short flags
        if len(token) == 2:
            yield token
        elif len(token) > 2:
            # -al -> -a -l
            for ch in token[1:]:
                yield f"-{ch}"


@lru_cache(maxsize=512)
def _help_text(binary_path: str) -> str:
    # Do not raise; some binaries exit non-zero for --help.
    try:
        proc = subprocess.run(
            [binary_path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=2,
        )
        return proc.stdout or ""
    except Exception:
        return ""


def verify_shell_command(command: str) -> VerificationResult:
    """Static pre-action verification for a shell command string.

    This intentionally aims to be:
    - fast
    - deterministic
    - conservative (prefers abstaining to false acceptance)

    It is **not** a full shell parser.
    """

    command = (command or "").strip()
    if not command:
        return VerificationResult(ok=False, reason="empty command")

    try:
        argv = shlex.split(command)
    except ValueError as e:
        return VerificationResult(ok=False, reason=f"shlex parse failed: {e}")

    if not argv:
        return VerificationResult(ok=False, reason="no argv")

    binary = argv[0]
    binary_path = shutil.which(binary)
    if not binary_path:
        return VerificationResult(ok=False, reason="binary not found", binary=binary)

    help_text = _help_text(binary_path)
    unknown_flags: list[str] = []

    for flag in _iter_flags(argv):
        # If we fail to obtain help text, we abstain only when a flag is present.
        if not help_text:
            unknown_flags.append(flag)
            continue

        # Simple containment check. This is imperfect but cheap.
        if flag not in help_text:
            unknown_flags.append(flag)

    if unknown_flags:
        return VerificationResult(
            ok=False,
            reason="unknown or unverifiable flags (refuse-when-unsure)",
            binary=binary,
            normalized=" ".join(shlex.quote(x) for x in argv),
            unknown_flags=unknown_flags,
        )

    return VerificationResult(
        ok=True,
        reason="ok",
        binary=binary,
        normalized=" ".join(shlex.quote(x) for x in argv),
    )

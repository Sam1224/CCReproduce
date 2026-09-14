# Look Before You Leap — Pre-Action Verification (Reproduction)

This folder provides a lightweight, runnable reproduction of the *core idea* in the paper **“Look Before You Leap: Pre-Action Verification for LLM Agents” (arXiv:2609.11957)**:

- Before an agent executes a shell command or applies a code edit, run a **cheap deterministic verifier**.
- If the verifier is uncertain, **refuse** (abstain) rather than guessing.

This reproduction focuses on:

1. **Shell command static verification**
   - Syntax check (via `shlex`)
   - Binary existence check (`shutil.which`)
   - Flag existence check (heuristic: search in `--help` output)

2. **Code edit verification**
   - Content-anchored edits (`search/replace`, `unified diff`) fail loudly when anchors don’t match.
   - Location-anchored edits (line-number edits) can **silently misapply**.
   - An `anchor_and_verify` applier demonstrates the paper’s “turn silent failures into explicit failures” principle.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Command verification demo
python3 scripts/benchmark_commands.py

# Edit verification demo
python3 scripts/benchmark_edits.py
```

## Notes

- This is **not** the original authors’ code.
- The benchmarks here are small and intended to be easy to run; they are designed to reflect the *failure modes* and *verification logic* described in the paper.

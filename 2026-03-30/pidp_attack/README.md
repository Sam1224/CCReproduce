# pidp_attack (toy reproduction)

This folder is a **toy, end-to-end reproduction skeleton** of the threat model in:

- **PIDP-Attack: Combining Prompt Injection with Database Poisoning Attacks on Retrieval-Augmented Generation Systems** (arXiv:2603.25164)

## What this reproduction covers

The paper’s central idea is a *compound attack* on RAG:

1) **Database poisoning**: attacker inserts a small number of adversarial passages into the retrieval corpus.
2) **Prompt injection at query time**: attacker appends a short injected string (e.g., via a compromised UI) to steer retrieval toward the poisoned passages.

This toy implementation provides a clean pipeline you can run locally:

- A bag-of-words hashing retriever.
- A tiny corpus + QA set.
- A toy generator that extracts an answer from the top retrieved doc.
- Baseline vs (poison-only, injection-only, combined PIDP) comparisons.

## Quickstart

```bash
cd 2026-03-30/pidp_attack
python3 -m pip install -r requirements.txt
python3 main.py
```

## Files

- `toy_data.py`: tiny corpus + QA
- `retriever.py`: simple hashing retriever
- `attack.py`: poison + injection construction
- `main.py`: runs the evaluation loop

## Notes / limitations

- The generator is intentionally a toy heuristic; swapping in an LLM is out of scope.
- The goal is to reproduce the *mechanism* (retrieval steering + downstream behavior changes).

# Evidence Gate (self-testing quality gate reproduction)

Reference paper: **Automated Self-Testing as a Quality Gate: Evidence-Driven Release Management for LLM Applications** (arXiv:2603.15676)

This folder implements a runnable “evidence gate” pipeline:

- define a suite of self-tests (unit / integration / scenario)
- run tests on an application function (a placeholder LLM-app)
- collect structured evidence (pass rate, latency, regression deltas)
- compute a risk score and produce a gate decision (ship / hold)
- persist evidence in SQLite for release traceability

## Quickstart

```bash
cd ccreproduce_repo/2026-03-31/evidence_gate
python3 -m pip install -r requirements.txt
python3 run.py
```

## Pseudocode (LLM-as-a-judge / red-team at scale)

```text
# NOT IMPLEMENTED
for prompt in large_redteam_set:
  resp = llm_app(prompt)
  judge = LLMJudge(prompt, resp)
  log(judge)
```

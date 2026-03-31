# Ego2Web (reproduction-oriented harness)

Reference paper: **Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos** (arXiv:2603.22529)

The original benchmark requires:

- egocentric video understanding (clip captions / grounding)
- **live web execution** on real websites
- LLM-as-a-Judge metric (Ego2WebJudge)

A full reproduction requires browser automation + MLLM evaluation. This folder provides:

- a **dataset schema** for (video_profile, instruction, expected answer)
- an **offline baseline agent** that uses the video profile to extract entities and proposes actions
- a **lightweight judge** that approximates the online judge by string/field matching

## Quickstart

```bash
cd ccreproduce_repo/2026-03-31/ego2web
python3 -m pip install -r requirements.txt
python3 run.py --n 200
```

## Pseudocode (online evaluation + MLLM judge)

```text
# NOT IMPLEMENTED
# 1) Use a browser agent to execute actions on live sites
# 2) Collect action history + screenshots
# 3) Ego2WebJudge: MLLM compares screenshots/trajectory vs visual evidence in video clip
```

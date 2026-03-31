# memento_skills (toy reproduction)

Toy, runnable reproduction skeleton for:

- **Memento-Skills: Let Agents Design Agents** (arXiv:2603.18743)

## What this reproduction captures

Memento-Skills describes an **agent-designing agent** that learns *without updating LLM parameters* by evolving an external **skill library** (structured markdown) and using a router to select skills based on a **stateful prompt**.

This toy implementation simulates the Read–Write loop:

- **Read**: a simple router selects a skill by matching the task description to skill metadata.
- **Execute**: skills are deterministic Python functions (stand-ins for tool skills).
- **Write**: when a task fails, the agent writes a new markdown skill file and can then solve similar future tasks.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-29/memento_skills
python3 train.py
python3 test.py
```

## Files

- `skills/*.md`: externalized skills (markdown)
- `skill_router.py`: similarity-based skill selection
- `agent.py`: read/write reflective loop
- `dataset.py`: synthetic tasks
- `train.py`: demonstrates improvement after writing new skills
- `test.py`: smoke test

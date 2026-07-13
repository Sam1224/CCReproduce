# Difficulty-Routed Control Toy Reproduction

This folder reproduces the core control pipeline of **When Should Service Agents Reconsider? Difficulty-Routed Control in Customer-Service Operations**.

The original paper studies how customer-service agents should route routine requests to a cheap baseline path while escalating operationally coupled sessions to a stronger workflow with **write-triggered reconsideration**. This toy implementation keeps that structure runnable with PyTorch:

- `ToyServiceSessionDataset` synthesizes customer-service sessions with routine vs. conflict-heavy requests.
- `DifficultyRouter` predicts whether a session is simple or complex.
- `WritePlanVerifier` scores candidate backend writes against the session evidence.
- `difficulty_routed_control` simulates the paper's selective escalation: simple sessions use a cheap baseline plan, while escalated sessions reconsider before the final write.

Run:

```bash
python train.py --cpu --epochs 8 --samples 768
python test.py --cpu --samples 256
```

Expected outcome: the baseline planner is already strong on routine requests, but the routed controller should improve markedly on the complex subset by correcting the target write before execution.

Simplifications kept explicit:

- The paper's full ReSpAct dialogue policy and real backend tools are approximated with synthetic sessions and candidate writes.
- The verifier scores candidate writes instead of running an LLM chain-of-thought.
- Human-verified airline/retail data are replaced by deterministic templates so the interface remains runnable on CPU.

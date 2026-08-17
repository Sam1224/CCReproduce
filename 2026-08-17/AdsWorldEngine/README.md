# AdsWorldEngine

Toy PyTorch reproduction of **AdsWorldEngine: A Self-Evolving Conversational Advertising Agent through Orchestrator and Tool Coevolution**.

This implementation mirrors the paper at a runnable scale:

- an **Opportunity Gate** that predicts whether the current dialogue turn should trigger ads;
- an **Orchestrator** that mixes dialogue features with tool outputs to rank candidate ads;
- a lightweight **Judge** that estimates conversation-to-ad relevance;
- a simple **tool coevolution loop** that prefers higher-reward slates over lower-reward slates.

The code uses a synthetic conversational-ad dataset and keeps the logic CPU-friendly. It is not a production ad server, but it preserves the paper's actor / tool / evaluator decomposition.

## Files

- `data.py`: synthetic ads, conversational examples, tool features and reward logic
- `model.py`: gate, judge, orchestrator and serving wrapper
- `train.py`: supervised training plus pairwise slate-improvement updates
- `test.py`: evaluates gate quality and slate reward / diversity

## Run

```bash
cd CCReproduce/2026-08-17/AdsWorldEngine
python3 train.py
python3 test.py
```

## Mapping to the paper

- **Opportunity Gate** → `OpportunityGateNet`
- **Orchestrator** → `SlateOrchestrator`
- **Label-grounded judge** → `RelevanceJudge`
- **Actor-tool coevolution** → pairwise preference updates over high- and low-reward slates

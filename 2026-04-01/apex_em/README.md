# APEX-EM (toy reproduction)

Paper: **APEX-EM: Non-Parametric Online Learning for Autonomous Agents via Structured Procedural-Episodic Experience Replay** (arXiv:2603.29093)

This folder contains a **minimal runnable** retrieval-augmented policy that captures the key theme: **non-parametric experience replay via memory retrieval**.

## What is implemented

- A toy episodic environment dataset (`dataset.py`).
- An `EpisodicMemory` kNN store + a parametric policy blended with the memory distribution (`model.py`).
- `train.py` / `test.py` scripts.

## Run

```bash
python3 train.py --epochs 3
python3 test.py --ckpt ckpt.pt
```

## Not implemented

- The paper’s full structured procedural graph / DAG traversal mechanisms.
- Real tool-using agents and online deployment loop.

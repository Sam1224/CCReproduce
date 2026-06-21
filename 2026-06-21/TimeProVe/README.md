# TimeProVe — Propose, then Verify (toy, offline reproduction)

Self-contained, runnable PyTorch reproduction of the **core method** in:

> Arkaprava Sinha, Dominick Reilly, Siddharth Krishnan, Hieu Le, Srijan Das.
> *TimeProVe: Propose, then Verify for Efficient Long Video Temporal Reasoning in
> Activities of Daily Living* (UNC Charlotte).
> arXiv: https://arxiv.org/abs/2606.20561 · project: https://thearkaprava.github.io/timeprove/

TimeProVe answers questions over **hours-long untrimmed ADL videos** without running
an expensive VLM over the whole video. It **proposes** action-grounded
answer–evidence hypotheses with cheap local modules, then **verifies** only the few
top hypotheses with an expensive cloud VLM, so reasoning cost is **decoupled from
video length**. This folder reproduces that pipeline at toy scale, fully offline,
in seconds on CPU.

## Pipeline (faithful to the paper, Sec 3)

**ACE — Action-based Candidate Evidence** (cheap, one pass over the full video):
1. **Action Detector** `f_act(v) -> P ∈ [0,1]^{T×|C|}` (Eq. 1); threshold + decode
   maximal contiguous activations into a sparse event timeline
   `A = {(c_i, s_i, e_i)}` (Eq. 2).
2. **Query-conditioned Proposal Generator**: turns `A` into candidate
   `(answer, evidence-window)` hypotheses at two granularities — **atomic** windows
   (per event) and **merged** windows (Eq. 4) — then **reranks** them with the local
   score `R(w|q) = R_tmp + R_sem + R_cov − R_len` (Eqs. 5–9).

**Temporal Verifier** (expensive VLM, invoked **only** on the few top clips):
`f_vlm(clip, q, a) -> (c∈{0,1}, verified answer, evidence)` (Eq. 10). The pipeline
walks the ranked hypotheses, calls the verifier per candidate, and stops on the
first acceptance or when the verification **budget** is exhausted.

## Faithfully implemented vs. stubbed

| Component | This repo |
|---|---|
| Action Detector (paper: MS-Temba, 17M, over I3D/CLIP feats) | **real, trainable** small temporal-CNN over toy segment features — same `v∈R^{T×D} → P∈[0,1]^{T×|C|}` contract |
| Reranking score (Eqs. 5–8) | **real**, exact four terms (temporal / semantic / coverage / length) |
| Timeline decode, atomic+merged windows (Eqs. 2,4) | **real** |
| Proposal "answer generator" (paper: edge LLM Gemma4-2B / Qwen2-7B) | heuristic stand-in: reads the dominant detected action label inside the window (the LLM's exact job) |
| Temporal Verifier (paper: GPT-4o / VideoLLaMA3 over RGB clip) | **trainable** stand-in scorer: pools the clip's segment features and scores them against query + candidate-answer embeddings — identical `f_vlm(clip,q,a)` interface; see comment + pseudocode in `model.py` |

## Files
- `data.py` — synthetic long videos (segment features + planted action timeline) and
  open-ended QA pairs (intent `τ`, content tokens `Q(q)`, gold answer + window).
- `model.py` — `ActionDetector`, timeline decode, proposal generator + scoring,
  `TemporalVerifier`, and the full `TimeProVe` propose-then-verify pipeline.
- `train.py` — trains the detector (multi-label BCE, Eq. 1) and the verifier (BCE over
  ranked hypotheses of the gold timeline). Saves `timeprove.pt`.
- `test.py` — accuracy, grounding IoU, and the efficiency reductions (VLM calls / cost).

## Run

```bash
pip install -r requirements.txt
python train.py     # trains detector + verifier -> timeprove.pt
python test.py      # accuracy + grounding + efficiency reductions
```

Example output: detector seg-acc ~1.0; **answer accuracy ~79%**, mean grounding IoU
~0.52, **avg 1.85 VLM calls/query vs 4.55** for an agentic baseline that verifies
every candidate (**~59% fewer calls**), and **~83% inference-cost reduction** vs a
dense full-video baseline that exposes the VLM to the whole clip — reproducing the
paper's qualitative "propose-then-verify decouples cost from video length" result
(paper headline: −75% VLM calls, −93% cost on OpenTSUBench).

## Notes
The toy scale (small action vocabulary, short synthetic videos) is what keeps it
CPU-runnable in seconds; the detector, the reranking math, the propose-then-verify
control flow, the budgeted verification, and the call/cost accounting are faithful
and drop-in replaceable with a real action detector + cloud VLM.

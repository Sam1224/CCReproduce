# SAMA — Semantic Anchor-aligned Augmentation (toy, offline reproduction)

Self-contained PyTorch reproduction of the method in:

> Quanjiang Guo, Chong Mu, Jiazhou Pan, Ming Jia, Ling Tian, Hui Gao, Zhao Kang.
> *SAMA: Semantic Anchor-aligned Augmentation for Unified Low-Resource Multimodal
> Information Extraction* (UESTC).
> arXiv: https://arxiv.org/abs/2606.18780 · Official repo: https://github.com/UESTC-GQJ/SAMA (placeholder, code "coming soon")

Faithful at toy scale, fully offline (no network / model downloads). Tiny dims and
synthetic tensors so train + test run in seconds on CPU.

## What is implemented (1:1 with the paper)

1. **Semantic Anchor Construction** (`anchors.py`, Sec III.C) — re-encode GT labels
   into inline-tagged anchor strings:
   - MNER: `<PER> lebron </PER>`
   - MRE : `<r> ( Subject lebron , plays_for , Object lakers ) </r>`
   - MEE : `<ev> clinching Trigger [ Arg : Agent lebron ] ... </ev>`
2. **CME-MLLM** (`model.py`, Sec III.B-D):
   - LoRA adapters (Eq. 1) `h = W0 x + B A x` as the structural basis for experts.
   - InstructBLIP-style **Q-Former stub** (learnable queries cross-attend image patches).
   - Anchor fusion `s_anchor = f_InstructBLIP(T, I, A_n)` (Eq. 2).
   - **Universal Adapter** `U(x)` + **Task-Specific Adapters** `D_n(x)` with the
     **anchor-motivated gate** `g = Softmax(Wg·[s_anchor, xt])`,
     `h = gu·U(x) + gd·D_n(x)` (Eqs. 3-4) and vocab projection (Eq. 5).
   - **Knowledge harmonization via mutual information** (Eq. 6) — teacher = Universal,
     student = Task-Specific, implemented with **InfoNCE** (the standard tractable
     lower bound of MI; exact-MI pseudocode in the `info_nce` docstring).
   - **Performance-adaptive objective** (Eqs. 7-9): `w_n = (1-a_n)^γ`, total loss
     `L = Σ w_n·L_gen,n + β·L_align`.
   - **Anchor-aware nucleus sampling** for generation (`generate.py`, Sec III.D-4).
3. **Anchor-Preserving Diffusion** (`diffusion.py`, Sec III.E) — a compact DDPM:
   - **Anchor-Weighted Prompt** (Eq. 10): entity-anchor tokens get emphasis weight ω.
   - **Masked latent blending** (Eq. 11): `z_{t-1} = M⊙z^src_{t-1} + (1-M)⊙z^gen_{t-1}`,
     anchor pixels kept from the original (identity), background synthesised (diversity).
4. **Dual-Constraint Filtering** (`filtering.py`, Sec III.F) —
   `S_conf = α·SimCLIP(T̂,Î) + (1-α)·SimSem(T̂,A)` (Eq. 12), discard < τ=0.75,
   keep argmax over K=5 candidates (Eq. 13). SimCLIP via a tiny ToyCLIP dual encoder.
5. **Label projection** — synthetic pairs inherit the original GT labels.

Hyperparameters follow Table II: γ=2.0, β=0.1, ω=1.2, top-p=0.9, α=0.6, τ=0.75.

## Data (`data.py`)

Toy text-image pairs across **MNER / MRE / MEE** with task-specific GT labels.
Images are tiny `[3,8,8]` tensors whose top-left **anchor patch** encodes the subject
entity identity (so the diffusion preservation mask `M` is meaningful). A
`low_resource_frac` split mirrors the paper's low-resource setting.

## Run

```bash
pip install -r requirements.txt
python train.py    # CME-MLLM adapters + diffusion + ToyCLIP -> cme.pt/diffusion.pt/clip.pt
python test.py     # generate + filter synthetic samples, print them, toy downstream F1
```

Example: 4/9 candidates kept after filtering; synthetic texts keep anchor entities
while varying syntax; downstream macro-F1 improves (low-resource backbone vs
backbone + SAMA augmentation).

## Approximations (and why)

| Paper | Here |
|---|---|
| InstructBLIP + FlanT5-XL (3B) + ViT-g/14 | small Transformer LM + conv image encoder + Q-Former stub |
| Stable Diffusion v1.5 LDM (VAE latents) | compact DDPM operating on the `[3,8,8]` image as z0 (VAE omitted; masking/conditioning identical) |
| CLIP-based SimCLIP | tiny contrastively-trained `ToyCLIP` |
| Exact mutual information (Eq. 6) | InfoNCE lower bound + exact-MI pseudocode comment |
| Real MNER/MRE/MEE datasets | small synthetic symbolic corpus |

The anchor construction, collaborative-experts gating, MI alignment, anchor-weighted
diffusion with masked blending, and dual-constraint filtering are all real, runnable
code and map directly onto the paper's equations.

## Files
- `data.py` — toy MNER/MRE/MEE dataset, vocab, anchor mask, low-resource split
- `anchors.py` — Semantic Anchor Construction (inline-tagged strings)
- `model.py` — CME-MLLM (LoRA experts, Q-Former, anchor gate, InfoNCE) + downstream backbone
- `diffusion.py` — Anchor-Preserving DDPM (anchor-weighted prompt, masked blending)
- `generate.py` — anchor-aware nucleus sampling
- `filtering.py` — ToyCLIP + Dual-Constraint Filtering
- `train.py` — trains adapters + diffusion + ToyCLIP
- `test.py` — augment → filter → print samples → toy downstream F1

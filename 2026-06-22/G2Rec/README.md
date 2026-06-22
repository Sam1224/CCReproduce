# G2Rec — Sparse Co-Engagement Graph Schema for Generative Recommendation (toy reproduction)

Self-contained, runnable PyTorch reproduction of the **core method** of:

> Ruizhong Qiu, Yinglong Xia, Dongqi Fu, Hanqing Zeng, Ren Chen, Xiangjun Fan,
> Hong Li, Hong Yan, Hanghang Tong. *Structuring and Tokenizing Distributed User
> Interest Context for Generative Recommendation* (**G2Rec**, Meta AI / UIUC).
> arXiv: https://arxiv.org/abs/2606.20554

G2Rec unifies **holistic graph co-engagement modeling** with **semantic
tokenization** for generative sequential recommendation. It (1) builds an
item-item co-engagement graph (dropping user nodes for scalability), (2)
sparsifies it with a theoretical guarantee, (3) runs a differentiable
**soft-modularity** graph clustering to obtain per-item soft interest profiles,
and (4) rewrites user histories as alternating *item / interest-profile* tokens
to train a decoder-only generative recommender with an item loss plus an
interest soft-label loss. This folder reproduces the full pipeline at toy scale,
fully offline, in ~5 s on CPU. The paper has **no official public code**.

## Pipeline (faithful to the paper)

| Stage | Paper | File |
|---|---|---|
| Toy data: item embeddings `X` + user sequences `I_u` | Sec. 2.1, Problem 1 | `data.py` |
| Co-engagement graph `E* = ∪_u I_u × I_u`; per-user sparse sampling `E = ∪_u sample(I_u×I_u, m)` | Eq. 1, Eq. 2, Theorem 2 | `graph.py` |
| Soft clustering by maximizing soft modularity `Q_soft(P)` | Prop. 3, Eq. 4 | `clustering.py` |
| Prototypes `V=(PᵀX)/(Pᵀ1)`, profile tokens `Y=PV`, sequence `R_u`, decoder, losses | Eq. 6–13 | `model.py` |
| Training: graph → cluster → tokens → train `L_item + λ·L_profile` | Eq. 13 | `train.py` |
| Eval: Recall@K / NDCG@K / MRR, leave-one-out, 1 pos + 99 neg | Sec. 5.1, Table 2 | `test.py` |

## Equation → code mapping

- **Co-engagement & sparsification (Eq. 1–2, Thm 2).** `graph.build_co_engagement_graph`
  samples, per user, `m` pairs from `I_u × I_u` **with replacement**, dedups into
  an undirected binary adjacency, and emits the *duplicate* edge representation
  `(i,j)` & `(j,i)` (paper convention for modularity). `graph.m_from_theorem`
  implements the Theorem-2 bound `m = 2N(1/(3ε)+1/ε²)·log(2|I|/δ)`, giving
  `|E| = O(M log M)`.
- **Soft modularity (Prop. 3, Eq. 4).** `clustering.soft_modularity`:
  `Q_soft(P) = (1/|E|) Σ_{(i,j)∈E} pᵢᵀpⱼ − γ·‖Pᵀk‖²/|E|²`, with `P = softmax(Z)`
  and degree vector `k`. `clustering.soft_cluster` maximizes it by gradient
  ascent (minimizing `−Q_soft`). `Z` is initialized via **Leiden** (gated import
  `igraph`+`leidenalg` in `try/except`), falling back to **KMeans** on item
  embeddings, then random.
- **Prototypes & profile tokens (Eq. 6–7).** `model.interest_prototypes`:
  `V = (PᵀX)/(Pᵀ1)`; `model.profile_tokens`: `Y = PV`.
- **Sequence (Eq. 8).** `model.G2Rec._build_sequence` builds
  `R_u = [⟨BOS⟩, x_{i1}, y_{i1}, …, x_{iN}, y_{iN}]` with token-type and positional
  embeddings, fed to a causal `TransformerEncoder` (decoder-only).
- **Losses (Eq. 9–13).** `model.G2Rec.loss`:
  - `L_item` — hidden state after each `y_t` (post-profile position) predicts the
    next item `i_{t+1}` over the item vocabulary (cross-entropy).
  - `L_profile` — hidden state at each `x_t` predicts item `i_t`'s **soft** interest
    profile `p_{i_t}` (cross-entropy against soft labels).
  - `L = L_item + λ·L_profile`.

## Run

```bash
pip install -r requirements.txt        # torch, numpy (igraph/leidenalg optional)
python train.py                        # builds graph, clusters, trains, saves ckpt
python test.py                         # loads ckpt, prints Recall/NDCG/MRR
```

Each module also has a `__main__` self-test (`python graph.py`, `python clustering.py`).

Example (CPU, ~5 s):

```
[graph] m/user=64, edges(dup)=12258
[clustering] init via kmeans, C=8
[clustering] final Q_soft=0.2206
epoch 10 | loss 4.1352 (item 4.1000 profile 0.0704) | val R@10 0.3667 NDCG@10 0.1714 MRR 0.1379
test: {'Recall@1':0.04,'Recall@5':0.18,'Recall@10':0.39,'NDCG@5':0.11,'NDCG@10':0.18,'MRR':0.14}
```

(Absolute numbers are on synthetic toy data and are **not** comparable to the
paper's Table 2; they only verify the pipeline learns and ranks sensibly.)

## Approximations / notes

- **Toy synthetic data** replaces Amazon Beauty/Sports/Toys & Yelp; items carry a
  latent cluster structure so co-engagement clustering is meaningful.
- **Backbone**: paper finetunes Llama-2-13B with LoRA on `d=64` SASRec embeddings.
  We use a small from-scratch decoder-only Transformer on fixed random item
  embeddings — the tokenization/loss contract is identical.
- **Leiden init** is optional (graceful KMeans fallback); the differentiable
  soft-modularity optimization itself is fully implemented.
- Item embeddings `X` are kept fixed (as the paper uses precomputed SASRec
  embeddings); profile tokens `Y` are computed once from `P` and reused.

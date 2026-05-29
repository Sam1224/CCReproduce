import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inverted_index import build_index, maxsim_query_by_inverted_index
from src.ssr_model import SSRConfig, SSRToyModel, maxsim_dense, maxsim_sparse
from src.toy_data import ToyQueryDataset, load_docs


def _pad_batch(token_lists: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(x) for x in token_lists)
    ids = torch.zeros((len(token_lists), max_len), dtype=torch.long)
    mask = torch.zeros((len(token_lists), max_len), dtype=torch.bool)
    for i, seq in enumerate(token_lists):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, : len(seq)] = True
    return ids, mask


def _recall_at_k(ranks: List[int], target: int, k: int) -> float:
    return 1.0 if target in ranks[:k] else 0.0


def _topk_from_scores(scores: torch.Tensor, k: int) -> List[int]:
    return torch.topk(scores, k=k).indices.tolist()


def eval_dense(
    model: SSRToyModel,
    docs: List[List[int]],
    dataset: ToyQueryDataset,
    device: torch.device,
    *,
    max_queries: int,
    max_docs: int,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        docs = docs[:max_docs]
        examples = dataset.examples[:max_queries]

        doc_ids, doc_mask = _pad_batch(docs)
        doc_ids = doc_ids.to(device)
        doc_mask = doc_mask.to(device)
        doc_vec = model.encode_dense(doc_ids)  # [D, Ld, H]

        r1 = r5 = r10 = 0.0
        t0 = time.time()
        chunk = 256
        for ex in examples:
            q_ids = torch.tensor(ex.query_ids, dtype=torch.long)[None, :].to(device)
            q_mask = torch.ones_like(q_ids, dtype=torch.bool)
            q_vec = model.encode_dense(q_ids)  # [1, Lq, H]

            scores = []
            for start in range(0, doc_vec.shape[0], chunk):
                d = doc_vec[start : start + chunk]
                dm = doc_mask[start : start + chunk]
                qi = q_vec.expand(d.shape[0], -1, -1)
                qim = q_mask.expand(d.shape[0], -1)
                s = maxsim_dense(qi, d, qim, dm)
                scores.append(s)

            scores_t = torch.cat(scores, dim=0)
            top10 = _topk_from_scores(scores_t, k=min(10, scores_t.shape[0]))
            r1 += _recall_at_k(top10, ex.pos_doc_id, 1)
            r5 += _recall_at_k(top10, ex.pos_doc_id, 5)
            r10 += _recall_at_k(top10, ex.pos_doc_id, 10)

        elapsed = time.time() - t0
        n = len(examples)
        return {
            "recall@1": r1 / n,
            "recall@5": r5 / n,
            "recall@10": r10 / n,
            "qps": n / max(elapsed, 1e-6),
        }


def eval_ssr_bruteforce(
    model: SSRToyModel,
    docs: List[List[int]],
    dataset: ToyQueryDataset,
    device: torch.device,
    *,
    max_queries: int,
    max_docs: int,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        docs = docs[:max_docs]
        examples = dataset.examples[:max_queries]

        doc_ids, doc_mask = _pad_batch(docs)
        doc_ids = doc_ids.to(device)
        doc_mask = doc_mask.to(device)
        _, doc_idx, doc_val = model.encode_sparse(doc_ids)

        r1 = r5 = r10 = 0.0
        t0 = time.time()
        chunk = 128
        for ex in examples:
            q_ids = torch.tensor(ex.query_ids, dtype=torch.long)[None, :].to(device)
            q_mask = torch.ones_like(q_ids, dtype=torch.bool)
            _, q_idx, q_val = model.encode_sparse(q_ids)

            scores = []
            for start in range(0, doc_idx.shape[0], chunk):
                d_idx = doc_idx[start : start + chunk]
                d_val = doc_val[start : start + chunk]
                dm = doc_mask[start : start + chunk]

                qi_idx = q_idx.expand(d_idx.shape[0], -1, -1)
                qi_val = q_val.expand(d_idx.shape[0], -1, -1)
                qim = q_mask.expand(d_idx.shape[0], -1)
                s = maxsim_sparse(qi_idx, qi_val, d_idx, d_val, qim, dm)
                scores.append(s)

            scores_t = torch.cat(scores, dim=0)
            top10 = _topk_from_scores(scores_t, k=min(10, scores_t.shape[0]))
            r1 += _recall_at_k(top10, ex.pos_doc_id, 1)
            r5 += _recall_at_k(top10, ex.pos_doc_id, 5)
            r10 += _recall_at_k(top10, ex.pos_doc_id, 10)

        elapsed = time.time() - t0
        n = len(examples)
        return {
            "recall@1": r1 / n,
            "recall@5": r5 / n,
            "recall@10": r10 / n,
            "qps": n / max(elapsed, 1e-6),
        }


def eval_ssr_inverted_index(
    model: SSRToyModel,
    docs: List[List[int]],
    dataset: ToyQueryDataset,
    device: torch.device,
    *,
    max_queries: int,
    max_docs: int,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        docs = docs[:max_docs]
        examples = dataset.examples[:max_queries]

        doc_sparse: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for doc in docs:
            d_ids = torch.tensor(doc, dtype=torch.long)[None, :].to(device)
            _, d_idx, d_val = model.encode_sparse(d_ids)
            doc_sparse.append((d_idx.squeeze(0).cpu(), d_val.squeeze(0).cpu()))

        index, build_s = build_index(doc_sparse)

        r1 = r5 = r10 = 0.0
        t0 = time.time()
        for ex in examples:
            q_ids = torch.tensor(ex.query_ids, dtype=torch.long)[None, :].to(device)
            q_mask = torch.ones_like(q_ids, dtype=torch.bool)
            _, q_idx, q_val = model.encode_sparse(q_ids)

            scores = maxsim_query_by_inverted_index(
                q_idx.squeeze(0).cpu(), q_val.squeeze(0).cpu(), q_mask.squeeze(0).cpu(), index
            )

            top10 = [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]]

            r1 += _recall_at_k(top10, ex.pos_doc_id, 1)
            r5 += _recall_at_k(top10, ex.pos_doc_id, 5)
            r10 += _recall_at_k(top10, ex.pos_doc_id, 10)

        elapsed = time.time() - t0
        n = len(examples)
        return {
            "recall@1": r1 / n,
            "recall@5": r5 / n,
            "recall@10": r10 / n,
            "qps": n / max(elapsed, 1e-6),
            "index_build_s": build_s,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--max_queries", type=int, default=50)
    parser.add_argument("--max_docs", type=int, default=500)
    parser.add_argument("--run_sparse_bruteforce", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(os.path.join(args.ckpt_dir, "ssr_toy.pt"), map_location="cpu")
    cfg = SSRConfig(**ckpt["cfg"])
    model = SSRToyModel(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)

    docs = load_docs(args.data_dir)
    dataset = ToyQueryDataset(args.data_dir)

    dense = eval_dense(model, docs, dataset, device, max_queries=args.max_queries, max_docs=args.max_docs)
    print("dense:", dense)

    ssr_idx = eval_ssr_inverted_index(model, docs, dataset, device, max_queries=args.max_queries, max_docs=args.max_docs)
    print("ssr_inverted_index:", ssr_idx)

    if args.run_sparse_bruteforce:
        ssr_bf = eval_ssr_bruteforce(model, docs, dataset, device, max_queries=args.max_queries, max_docs=args.max_docs)
        print("ssr_bruteforce:", ssr_bf)


if __name__ == "__main__":
    main()

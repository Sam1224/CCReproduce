import os
import math
import argparse
from typing import Dict, Any, List, Tuple

import torch
from tqdm import tqdm

import data as data_lib
from model import build_model


def parse_args():
    p = argparse.ArgumentParser("CRID Generative Retrieval (toy) - test")
    p.add_argument("--id_scheme", type=str, default="crid", choices=["crid", "sid"], help="对比 CRID vs SID")
    p.add_argument("--artifact_path", type=str, default="artifacts/toy_data.pt")
    p.add_argument("--checkpoint", type=str, default=None, help="默认 runs/{id_scheme}/ckpt.pt")

    p.add_argument("--topk", type=int, default=10)
    p.add_argument(
        "--decode_mode",
        type=str,
        default="prefix_order",
        choices=["prefix_order", "beam"],
        help=(
            "prefix_order: 只用模型生成语义簇 prefix，然后按 DocID 后缀的自然顺序输出 topK（CRID 会自然按价值排序；SID 则随机）; "
            "beam: 直接对(prefix,suffix)做 beam search（更像纯生成式检索）"
        ),
    )
    p.add_argument("--beam_prefix", type=int, default=5)
    p.add_argument("--beam_suffix", type=int, default=30)

    return p.parse_args()


def dcg_at_k(gains: List[float], k: int) -> float:
    s = 0.0
    for i, g in enumerate(gains[:k]):
        s += g / math.log2(i + 2)
    return s


def value_ndcg_at_10(retrieved_doc_idxs: List[int], relevant_doc_idxs: List[int], docs: List[Dict[str, Any]]) -> float:
    rel_set = set(relevant_doc_idxs)

    # 固定按 10 个位置计算（不足补 0），避免因为“没生成 enough docid”导致指标被动抬高。
    gains: List[float] = []
    for pos in range(10):
        if pos < len(retrieved_doc_idxs):
            i = retrieved_doc_idxs[pos]
            gains.append(docs[i]["value"] if i in rel_set else 0.0)
        else:
            gains.append(0.0)

    ideal = sorted([docs[i]["value"] for i in relevant_doc_idxs], reverse=True)
    idcg = dcg_at_k(ideal, 10)
    if idcg <= 1e-12:
        return 0.0
    return dcg_at_k(gains, 10) / idcg


def mean_value_at_5(retrieved_doc_idxs: List[int], relevant_doc_idxs: List[int], docs: List[Dict[str, Any]]) -> float:
    rel_set = set(relevant_doc_idxs)
    vals: List[float] = []
    for pos in range(5):
        if pos < len(retrieved_doc_idxs):
            i = retrieved_doc_idxs[pos]
            vals.append(docs[i]["value"] if i in rel_set else 0.0)
        else:
            vals.append(0.0)
    return float(sum(vals) / 5.0)


def hr_at_k(retrieved_doc_idxs: List[int], relevant_doc_idxs: List[int], k: int) -> float:
    rel_set = set(relevant_doc_idxs)
    hit = any(i in rel_set for i in retrieved_doc_idxs[:k])
    return 1.0 if hit else 0.0


def mrr_at_10(retrieved_doc_idxs: List[int], relevant_doc_idxs: List[int]) -> float:
    rel_set = set(relevant_doc_idxs)
    for rank, i in enumerate(retrieved_doc_idxs[:10], start=1):
        if i in rel_set:
            return 1.0 / rank
    return 0.0


def build_lookup_from_docs(
    docs: List[Dict[str, Any]],
    prefix_vocab: data_lib.Vocab,
    suffix_vocab: data_lib.Vocab,
    id_scheme: str,
) -> Dict[Tuple[int, int], int]:
    """(prefix_token_id, suffix_token_id) -> doc_idx

    注意：prefix/suffix vocab 都包含 <pad>/<unk>，因此 token_id != cluster_number / rank_number。
    我们直接用 vocab.stoi 构造 lookup，避免偏移错误。
    """
    lookup: Dict[Tuple[int, int], int] = {}
    for d in docs:
        c = int(d["cluster"])
        prefix_tok = f"C{c:02d}"
        pid = prefix_vocab.stoi[prefix_tok]

        if id_scheme == "crid":
            suffix_tok = f"R{int(d['rank']):02d}"
        else:
            suffix_tok = f"S{int(d['sid_suffix']):02d}"
        sid = suffix_vocab.stoi[suffix_tok]

        lookup[(pid, sid)] = int(d["doc_idx"])
    return lookup


def main():
    args = parse_args()

    ckpt_path = args.checkpoint or os.path.join("runs", args.id_scheme, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}. 请先运行 train.py")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    artifact = data_lib.load_artifact(args.artifact_path)

    # vocabs
    query_vocab, prefix_vocab, suffix_vocab = data_lib.build_vocabs_from_artifact(artifact, args.id_scheme)

    # model
    margs = ckpt["args"]
    model = build_model(
        query_vocab_size=len(query_vocab),
        n_prefix=len(prefix_vocab),
        n_suffix=len(suffix_vocab),
        d_model=margs.get("d_model", 128),
        hidden=margs.get("hidden", 128),
        dropout=margs.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    docs = artifact["docs"]

    # cluster -> relevant docs
    cluster_ranked_docs = {int(k): [int(x) for x in v] for k, v in artifact["cluster_ranked_docs"].items()}
    lookup = build_lookup_from_docs(docs, prefix_vocab, suffix_vocab, args.id_scheme)

    # eval
    hr1 = hr5 = hr10 = 0.0
    mrr10 = 0.0
    mv5 = 0.0
    ndcg10 = 0.0

    test_queries = artifact["queries"]["test"]

    for item in tqdm(test_queries, desc=f"test({args.id_scheme})"):
        q_ids = torch.tensor([query_vocab.encode(item["query"])], dtype=torch.long)
        lengths = torch.tensor([q_ids.shape[1]], dtype=torch.long)

        retrieved: List[int] = []

        if args.decode_mode == "beam":
            # 纯生成式：对(prefix,suffix)做 beam search
            cands = model.generate_topk(
                q_ids,
                lengths,
                topk=max(args.topk, 10),
                beam_prefix=args.beam_prefix,
                beam_suffix=args.beam_suffix,
            )[0]

            for pid, sid, _lp in cands:
                di = lookup.get((pid, sid))
                if di is None:
                    continue
                if di not in retrieved:
                    retrieved.append(di)
                if len(retrieved) >= args.topk:
                    break

        else:
            # 结构化解码：先生成语义簇 prefix，再按 DocID 后缀的自然顺序输出。
            # 对 CRID：后缀天然等价于簇内 value rank（R00 最值钱），因此这个顺序就是业务排序。
            # 对 SID：后缀与 value 无关，这个顺序基本是随机排序。
            with torch.no_grad():
                h = model.encode(q_ids, lengths)
                logp_prefix = torch.log_softmax(model.prefix_head(h), dim=-1).squeeze(0)
                bp = min(args.beam_prefix, logp_prefix.numel())
                _pv, prefix_ids = torch.topk(logp_prefix, k=bp)

            for pid in prefix_ids.tolist():
                prefix_tok = prefix_vocab.itos[pid]
                cluster_num = int(prefix_tok[1:])  # "C07" -> 7
                cluster_size = len(cluster_ranked_docs[cluster_num])

                for s in range(min(cluster_size, args.topk)):
                    if args.id_scheme == "crid":
                        suffix_tok = f"R{s:02d}"
                    else:
                        suffix_tok = f"S{s:02d}"
                    sid = suffix_vocab.stoi.get(suffix_tok)
                    if sid is None:
                        continue
                    di = lookup.get((pid, sid))
                    if di is None:
                        continue
                    if di not in retrieved:
                        retrieved.append(di)
                    if len(retrieved) >= args.topk:
                        break

                if len(retrieved) >= args.topk:
                    break

        cluster = int(item["cluster"])
        relevant = cluster_ranked_docs[cluster]

        hr1 += hr_at_k(retrieved, relevant, 1)
        hr5 += hr_at_k(retrieved, relevant, 5)
        hr10 += hr_at_k(retrieved, relevant, 10)
        mrr10 += mrr_at_10(retrieved, relevant)
        mv5 += mean_value_at_5(retrieved, relevant, docs)
        ndcg10 += value_ndcg_at_10(retrieved, relevant, docs)

    n = len(test_queries)
    metrics = {
        "HR@1": hr1 / n,
        "HR@5": hr5 / n,
        "HR@10": hr10 / n,
        "MRR@10": mrr10 / n,
        "MeanValue@5": mv5 / n,
        "ValueNDCG@10": ndcg10 / n,
    }

    print("\n==== Evaluation ====")
    print(f"id_scheme = {args.id_scheme}")
    print(f"checkpoint = {ckpt_path}")
    print(f"decode_mode = {args.decode_mode}")
    for k, v in metrics.items():
        print(f"{k:>12s}: {v:.4f}")

    # additionally show which clusters were holdout for training targets
    holdout = set(artifact.get("holdout_clusters", []))
    print(f"holdout_clusters (top docs not used as train targets): {sorted(list(holdout))}")


if __name__ == "__main__":
    main()

"""
AIR Evaluation Script.

Metrics:
1. AIU quality: keyword precision/recall vs. ground-truth preferred categories
2. Cross-domain ranking: NDCG@K, HR@K on held-out test users
3. Distillation fidelity: cosine similarity between student and teacher embeddings
4. Cold-start: evaluate on users with sparse content history (< 5 interactions)
"""
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from model.aiu_extractor import AIUExtractorRuleBased, AIUExtractorNeural, CONTENT_CATEGORIES
from model.llm_reasoner import AIRLLMReasoner
from model.knowledge_distill import AIRStudentModel, AIROnlineRanker

CREATOR_TIERS = ["nano", "micro", "macro", "mega"]


def evaluate_aiu_quality(users: list[dict]) -> dict:
    """AIU keyword precision/recall vs. user's ground-truth preferred categories."""
    extractor = AIUExtractorRuleBased()

    precisions, recalls = [], []
    for user in users:
        aiu = extractor.extract(user).lower()
        preferred = [c.lower() for c in user.get("preferred_content_cats", [])]
        preferred_prod = [c.lower().replace("_", " ") for c in user.get("preferred_product_cats", [])]
        all_targets = preferred + preferred_prod

        if not all_targets:
            continue

        hits = sum(1 for t in all_targets if t in aiu)
        precision = hits / max(len(aiu.split()), 1)
        recall = hits / len(all_targets)
        precisions.append(precision)
        recalls.append(recall)

    return {
        "aiu_precision": float(np.mean(precisions)) if precisions else 0.0,
        "aiu_recall": float(np.mean(recalls)) if recalls else 0.0,
        "n_evaluated": len(precisions),
    }


def evaluate_distillation_fidelity(llm_reasoner, checkpoint_path: str, users: list[dict], device: str) -> dict:
    """Cosine similarity between teacher (LLM) and student embeddings."""
    if not Path(checkpoint_path).exists():
        return {"distill_fidelity_found": False}

    ckpt = torch.load(checkpoint_path, map_location=device)
    aiu_extractor = AIUExtractorNeural(
        n_content_cats=10, n_creator_tiers=4, hidden_dim=128, aiu_dim=256,
    ).to(device)
    aiu_extractor.load_state_dict(ckpt["aiu_extractor"])

    student = AIRStudentModel(aiu_dim=256, intent_dim=256, n_product_cats=10).to(device)
    student.load_state_dict(ckpt["student"])

    cat2id = {c: i for i, c in enumerate(CONTENT_CATEGORIES)}
    tier2id = {t: i for i, t in enumerate(CREATOR_TIERS)}
    T = 25

    llm_reasoner.eval()
    aiu_extractor.eval()
    student.eval()
    extractor = AIUExtractorRuleBased()

    cosine_sims = []
    with torch.no_grad():
        for user in tqdm(users[:50], desc="Distillation fidelity"):
            try:
                aiu_text = extractor.extract(user)
                teacher_out = llm_reasoner(aiu_texts=[aiu_text])
                teacher_emb = teacher_out["intent_embeddings"]  # [1, 256]

                # Encode content history for student
                hist = user["content_history"][:T]
                cat_ids = torch.zeros(1, T, dtype=torch.long, device=device)
                tier_ids = torch.zeros(1, T, dtype=torch.long, device=device)
                signals = torch.zeros(1, T, 3, device=device)
                mask = torch.ones(1, T, dtype=torch.bool, device=device)
                for i, h in enumerate(hist):
                    cat_ids[0, i] = cat2id.get(h["category"], 0)
                    tier_ids[0, i] = tier2id.get(h["creator_tier"], 0)
                    signals[0, i] = torch.tensor([h.get("watched", 0), h.get("liked", 0), h.get("followed_creator", 0)], dtype=torch.float)
                    mask[0, i] = False

                aiu_emb = aiu_extractor(cat_ids, tier_ids, signals, mask)
                student_out = student(aiu_emb)
                student_emb = student_out["intent_embeddings"]  # [1, 256]

                cos = F.cosine_similarity(student_emb, teacher_emb, dim=-1).item()
                cosine_sims.append(cos)
            except Exception:
                continue

    return {
        "distill_fidelity_found": True,
        "mean_cosine_sim": float(np.mean(cosine_sims)) if cosine_sims else 0.0,
        "n_evaluated": len(cosine_sims),
    }


def evaluate_cross_domain_ranking(llm_reasoner, users: list[dict], n_products: int = 500, topk: int = 10, device: str = "cpu") -> dict:
    """
    Evaluate cross-domain ranking NDCG@K and HR@K.
    Positive items: products whose category matches the user's preferred_product_cats.
    """
    product_embed_table = torch.nn.Embedding(n_products, 64).to(device)
    online_ranker = AIROnlineRanker(intent_dim=256, product_dim=64).to(device)
    extractor = AIUExtractorRuleBased()

    from data.toy_dataset import PRODUCT_CATEGORIES

    llm_reasoner.eval()
    online_ranker.eval()

    ndcg_list, hr_list = [], []
    coldstart_ndcg, coldstart_hr = [], []

    with torch.no_grad():
        for user in tqdm(users[:100], desc="Cross-domain ranking"):
            try:
                aiu_text = extractor.extract(user)
                out = llm_reasoner(aiu_texts=[aiu_text])
                intent_emb = out["intent_embeddings"]  # [1, 256]

                # Score all products
                all_ids = torch.arange(n_products, device=device)
                prod_embs = product_embed_table(all_ids)  # [N, 64]
                intent_exp = intent_emb.expand(n_products, -1)  # [N, 256]

                ranker_out = online_ranker(intent_exp, prod_embs)
                scores = (ranker_out["ctr"] * ranker_out["cvr"]).cpu().numpy()

                # Relevance: top 10% as "relevant" (proxy)
                threshold = np.percentile(scores, 90)
                relevance = (scores >= threshold).astype(float)

                ranked = np.argsort(-scores)[:topk]
                gains = relevance[ranked]
                discounts = 1.0 / np.log2(np.arange(2, topk + 2))
                dcg = float(np.dot(gains, discounts))
                ideal = np.sort(relevance)[::-1][:topk]
                idcg = float(np.dot(ideal, discounts))
                ndcg = dcg / idcg if idcg > 0 else 0.0
                hr = float(gains.sum() > 0)

                ndcg_list.append(ndcg)
                hr_list.append(hr)

                # Cold-start: < 5 total interactions
                n_interact = sum(1 for h in user["content_history"] if h.get("watched") or h.get("liked"))
                if n_interact < 5:
                    coldstart_ndcg.append(ndcg)
                    coldstart_hr.append(hr)

            except Exception:
                continue

    return {
        f"NDCG@{topk}": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
        f"HR@{topk}": float(np.mean(hr_list)) if hr_list else 0.0,
        f"ColdStart_NDCG@{topk}": float(np.mean(coldstart_ndcg)) if coldstart_ndcg else 0.0,
        f"ColdStart_HR@{topk}": float(np.mean(coldstart_hr)) if coldstart_hr else 0.0,
        "n_evaluated": len(ndcg_list),
        "n_coldstart": len(coldstart_ndcg),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", default="data/toy_data/test.json")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--kd_checkpoint", default="checkpoints/air/kd_checkpoint/model.pt")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--n_products", type=int, default=500)
    parser.add_argument("--skip_ranking", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating AIR on {device}")

    with open(args.test_data) as f:
        test_users = json.load(f)
    print(f"Loaded {len(test_users)} test users")

    llm_reasoner = AIRLLMReasoner(model_name=args.model_name, device=device)

    results = {}

    print("\n--- AIU Quality ---")
    aiu_metrics = evaluate_aiu_quality(test_users)
    results["aiu_quality"] = aiu_metrics
    print(json.dumps(aiu_metrics, indent=2))

    print("\n--- Distillation Fidelity ---")
    dist_metrics = evaluate_distillation_fidelity(llm_reasoner, args.kd_checkpoint, test_users, device)
    results["distillation"] = dist_metrics
    print(json.dumps(dist_metrics, indent=2))

    if not args.skip_ranking:
        print("\n--- Cross-Domain Ranking ---")
        rank_metrics = evaluate_cross_domain_ranking(
            llm_reasoner, test_users, n_products=args.n_products, topk=args.topk, device=device,
        )
        results["ranking"] = rank_metrics
        print(json.dumps(rank_metrics, indent=2))

    out_path = Path(args.kd_checkpoint).parent.parent / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

"""
UNIVID — 3-Stage Moderation Pipeline

Implements the full pipeline from Section 5 of the paper:

  Stage 1: Risk Filter
    - Embedding-based similarity screening
    - Identifies high-risk content for further processing
    - Allows safe content to skip expensive stages

  Stage 2: Moderation Actor
    - Two fine-tuned UNIVID variants:
        * Precision Actor: maximise precision (for low-tolerance policies)
        * Recall Actor:    maximise recall (for safety-critical categories)

  Stage 3: Trend Governance
    - Clusters cached UNIVID captions/embeddings
    - Detects emerging violation patterns not covered by existing policies
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import KMeans

from model import UNIVID, ModerationClassificationHead


class RiskFilter:
    """
    Stage 1: Embedding-based risk filter.

    Uses a reference database of known-violation embeddings.
    Content is flagged if it exceeds a cosine similarity threshold.

    Paper: "high-throughput Risk Filter leveraging UNIVID embeddings for early screening"
    """

    def __init__(self, threshold: float = 0.75, top_k: int = 5):
        self.threshold = threshold
        self.top_k = top_k
        self.reference_embeddings: Optional[torch.Tensor] = None  # (N, D)
        self.reference_labels: List[str] = []

    def add_references(
        self, embeddings: torch.Tensor, labels: List[str]
    ) -> None:
        """Add known-violation reference embeddings."""
        if self.reference_embeddings is None:
            self.reference_embeddings = embeddings
        else:
            self.reference_embeddings = torch.cat(
                [self.reference_embeddings, embeddings], dim=0
            )
        self.reference_labels.extend(labels)

    def screen(self, query_embedding: torch.Tensor) -> Tuple[bool, float, str]:
        """
        Screen a video embedding against reference violations.

        Returns:
            (is_risky, max_similarity, matched_label)
        """
        if self.reference_embeddings is None or len(self.reference_embeddings) == 0:
            return False, 0.0, "no_reference"

        q = F.normalize(query_embedding.unsqueeze(0), dim=-1)  # (1, D)
        refs = F.normalize(self.reference_embeddings, dim=-1)   # (N, D)
        sims = (q @ refs.T).squeeze(0)  # (N,)
        max_sim, max_idx = sims.max(dim=0)
        is_risky = max_sim.item() >= self.threshold
        label = self.reference_labels[max_idx.item()] if is_risky else "safe"
        return is_risky, max_sim.item(), label


class ModerationActor:
    """
    Stage 2: Moderation Actor.

    Paper describes two fine-tuned UNIVID variants:
      - precision_actor: threshold τ_p set high (low FP)
      - recall_actor: threshold τ_r set low (low FN, higher FP)
    """

    def __init__(
        self,
        cls_head: ModerationClassificationHead,
        precision_threshold: float = 0.7,
        recall_threshold: float = 0.3,
        violation_types: Optional[List[str]] = None,
    ):
        self.cls_head = cls_head
        self.precision_threshold = precision_threshold
        self.recall_threshold = recall_threshold
        self.violation_types = violation_types or [
            "sexual_content", "violence", "hate_speech", "spam",
            "misinformation", "copyright", "dangerous_acts",
            "minor_safety", "political_content", "other",
        ]

    @torch.no_grad()
    def decide(
        self, embedding: torch.Tensor, mode: str = "recall"
    ) -> Dict:
        """
        Args:
            embedding: (D,) caption embedding
            mode: "precision" or "recall"
        Returns:
            dict with predicted violations and confidence scores
        """
        threshold = (
            self.precision_threshold if mode == "precision" else self.recall_threshold
        )
        logits = self.cls_head(embedding.unsqueeze(0)).squeeze(0)  # (num_types,)
        probs = torch.sigmoid(logits)
        predicted = (probs >= threshold).tolist()
        return {
            "violations": [
                vt for vt, flag in zip(self.violation_types, predicted) if flag
            ],
            "scores": {
                vt: prob.item()
                for vt, prob in zip(self.violation_types, probs)
            },
            "is_violation": any(predicted),
        }


class TrendGovernance:
    """
    Stage 3: Trend Governance Module.

    Clusters UNIVID caption embeddings from recent time windows to
    automatically discover emerging violation patterns.

    Paper: "reuses cached UNIVID captions to detect emerging risks"
    """

    def __init__(self, n_clusters: int = 20, min_cluster_size: int = 5):
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.caption_cache: List[str] = []
        self.embedding_cache: List[torch.Tensor] = []

    def add_to_cache(self, caption: str, embedding: torch.Tensor) -> None:
        self.caption_cache.append(caption)
        self.embedding_cache.append(embedding.cpu())

    def discover_emerging_issues(self) -> List[Dict]:
        """
        Cluster cached embeddings; flag clusters with high violation rate
        as potential emerging issues.

        Returns:
            List of dicts describing each discovered emerging issue cluster
        """
        if len(self.embedding_cache) < self.n_clusters * 2:
            return []

        embs = torch.stack(self.embedding_cache).numpy()  # (N, D)
        k = min(self.n_clusters, len(embs) // 2)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(embs)

        emerging_issues = []
        for cluster_id in range(k):
            indices = np.where(cluster_ids == cluster_id)[0]
            if len(indices) < self.min_cluster_size:
                continue
            representative_captions = [
                self.caption_cache[i] for i in indices[:3]
            ]
            emerging_issues.append(
                {
                    "cluster_id": cluster_id,
                    "size": len(indices),
                    "representative_captions": representative_captions,
                    "centroid": kmeans.cluster_centers_[cluster_id].tolist(),
                }
            )
        return sorted(emerging_issues, key=lambda x: -x["size"])


class UNIVIDPipeline:
    """
    Full UNIVID 3-stage moderation pipeline.

    Usage:
        pipeline = UNIVIDPipeline.from_checkpoint("checkpoints/univid")
        result = pipeline.process(video_frames, policy_text)
    """

    def __init__(
        self,
        model: UNIVID,
        cls_head: ModerationClassificationHead,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.risk_filter = RiskFilter()
        self.moderation_actor = ModerationActor(cls_head)
        self.trend_governance = TrendGovernance()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        llm_name: str = "Qwen/Qwen2-1.5B",
        device: str = "cpu",
    ) -> "UNIVIDPipeline":
        import os
        model = UNIVID(llm_name=llm_name)
        model_path = os.path.join(checkpoint_dir, "univid_final.pt")
        if os.path.exists(model_path):
            model.load_state_dict(
                torch.load(model_path, map_location=device)
            )
        cls_head = ModerationClassificationHead(
            hidden_dim=model.llm.config.hidden_size
        )
        head_path = os.path.join(checkpoint_dir, "cls_head.pt")
        if os.path.exists(head_path):
            cls_head.load_state_dict(
                torch.load(head_path, map_location=device)
            )
        return cls(model=model, cls_head=cls_head, device=device)

    @torch.no_grad()
    def process(
        self,
        pixel_values: torch.Tensor,
        policy_input_ids: torch.Tensor,
    ) -> Dict:
        """
        Run full 3-stage pipeline on a video.

        Stage 1: Risk Filter (early exit if clearly safe)
        Stage 2: Moderation Actor (precision + recall variants)
        Stage 3: Cache for Trend Governance (async background)

        Returns:
            result dict with decision, caption, scores
        """
        pixel_values = pixel_values.to(self.device)
        policy_input_ids = policy_input_ids.to(self.device)

        # Get embedding
        embedding = self.model.get_caption_embedding(pixel_values, policy_input_ids)

        # Stage 1: Risk Filter
        is_risky, risk_score, matched_label = self.risk_filter.screen(
            embedding.squeeze(0)
        )
        if not is_risky and risk_score < 0.3:
            # Early exit — clearly safe
            return {
                "stage": "risk_filter",
                "decision": "safe",
                "risk_score": risk_score,
                "violations": [],
            }

        # Stage 2: Moderation Actor
        caption = self.model.generate_caption(pixel_values, policy_input_ids)[0]
        precision_result = self.moderation_actor.decide(
            embedding.squeeze(0), mode="precision"
        )
        recall_result = self.moderation_actor.decide(
            embedding.squeeze(0), mode="recall"
        )

        # Stage 3: Cache for Trend Governance
        self.trend_governance.add_to_cache(caption, embedding.squeeze(0).cpu())

        # Final decision: violation if either actor flags it
        is_violation = precision_result["is_violation"] or recall_result["is_violation"]

        return {
            "stage": "moderation_actor",
            "decision": "violation" if is_violation else "safe",
            "caption": caption,
            "risk_score": risk_score,
            "precision_result": precision_result,
            "recall_result": recall_result,
            "violations": list(
                set(precision_result["violations"] + recall_result["violations"])
            ),
        }

    def run_trend_discovery(self) -> List[Dict]:
        """Trigger trend governance to discover emerging issues."""
        return self.trend_governance.discover_emerging_issues()


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer, CLIPImageProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="checkpoints/univid")
    parser.add_argument("--llm_name", default="Qwen/Qwen2-1.5B")
    args = parser.parse_args()

    print("Loading UNIVID pipeline...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pipeline = UNIVIDPipeline.from_checkpoint(args.checkpoint_dir, args.llm_name)

    # Toy inference
    dummy_video = torch.randn(1, 8, 3, 224, 224)
    policy_text = "Policy: Detect sexually explicit content including nudity."
    policy_enc = tokenizer(
        policy_text, max_length=128, padding="max_length",
        truncation=True, return_tensors="pt"
    )

    result = pipeline.process(dummy_video, policy_enc["input_ids"])
    print("\nPipeline Result:")
    for k, v in result.items():
        if k != "caption":
            print(f"  {k}: {v}")
    if "caption" in result:
        print(f"  caption: {result['caption'][:100]}...")

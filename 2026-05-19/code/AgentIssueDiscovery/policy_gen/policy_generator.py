"""Policy generator: automatically generates annotation policies from issue clusters.

Paper §3.3: After clustering identifies new issue groups, the system automatically
generates draft annotation policies for human review and integration into the policy library.
"""

from __future__ import annotations

from dataclasses import dataclass

from clustering.fine_cluster import FineCluster


@dataclass
class GeneratedPolicy:
    issue_name: str
    description: str
    example_indicators: list[str]
    severity: str  # Low / Medium / High
    source_cluster_id: str
    sample_videos: list[str]


def _heuristic_policy_from_descriptions(descriptions: list[str]) -> dict:
    """Heuristic policy generation (simulates LLM output).

    In production: call LLM with format_policy_prompt() from agent/prompts.py
    """
    # Simple keyword-based heuristic
    all_text = " ".join(descriptions).lower()

    if any(w in all_text for w in ["investment", "crypto", "returns", "lottery"]):
        return {
            "issue_name": "Subtle Financial Fraud Content",
            "description": "Videos promoting dubious financial schemes disguised as investment tips, "
                           "cryptocurrency pumps, or lottery-style winnings.",
            "indicators": [
                "Promises of guaranteed high returns (>100%)",
                "Urgency language combined with financial terminology",
                "Request for paid membership or upfront payment",
            ],
            "severity": "High",
        }
    elif any(w in all_text for w in ["smoking", "tobacco", "club", "lifestyle"]):
        return {
            "issue_name": "Disguised Tobacco Advertising",
            "description": "Content that normalizes or promotes tobacco use through lifestyle "
                           "framing, influencer placement, or 'accessories' marketing.",
            "indicators": [
                "Tobacco products shown in glamorous/aspirational settings",
                "Influencers shown smoking without health warnings",
                "Products labeled as 'accessories' but clearly tobacco-related",
            ],
            "severity": "Medium",
        }
    elif any(w in all_text for w in ["cure", "dissolves", "miracle", "herb", "supplement"]):
        return {
            "issue_name": "Misleading Health Claims",
            "description": "Videos making unverified medical or health claims about products, "
                           "herbs, or supplements without scientific evidence.",
            "indicators": [
                "Claims of curing diseases without medical backing",
                "Exaggerated weight loss/body transformation claims",
                "Promotion of unregulated health products as cures",
            ],
            "severity": "High",
        }
    else:
        return {
            "issue_name": "Emerging Content Issue (Unclassified)",
            "description": "Content exhibiting unusual patterns that may violate platform norms "
                           "but do not fit existing policy categories.",
            "indicators": [
                "Content flagged by recall agent as anomalous",
                "Does not match any existing policy pattern",
                "Requires human review for classification",
            ],
            "severity": "Low",
        }


class PolicyGenerator:
    """Generates annotation policies from issue clusters.

    Paper §3.3: Cluster-level policy generation for human review.
    """

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    def generate(self, cluster: FineCluster) -> GeneratedPolicy:
        if not cluster.is_new_issue:
            # Variants don't get new policies — they update existing ones
            return GeneratedPolicy(
                issue_name="[Variant Update]",
                description=f"Variant of existing issue cluster {cluster.cluster_id}.",
                example_indicators=[],
                severity="Low",
                source_cluster_id=cluster.cluster_id,
                sample_videos=cluster.video_ids[:3],
            )

        result = _heuristic_policy_from_descriptions(cluster.descriptions)
        return GeneratedPolicy(
            issue_name=result["issue_name"],
            description=result["description"],
            example_indicators=result["indicators"],
            severity=result["severity"],
            source_cluster_id=cluster.cluster_id,
            sample_videos=cluster.video_ids[:3],
        )

    def generate_all(self, clusters: list[FineCluster]) -> list[GeneratedPolicy]:
        return [self.generate(c) for c in clusters]

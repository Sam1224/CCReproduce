"""Prompt templates for the LLM-based recall agent."""

EXISTING_POLICIES = [
    "explicit violence",
    "hate speech",
    "spam promotion",
    "copyright infringement",
]

RECALL_SYSTEM_PROMPT = """You are a content safety expert for a short-video platform.
Your task is to analyze video content descriptions and identify videos that may contain
problematic content NOT covered by the existing content policy.

Existing policy covers: {existing_policies}

Your job is to flag videos that seem to violate platform norms but are NOT clearly
covered by the policies above. These may be subtle, emerging, or novel issues.

Respond with:
RECALL: YES or NO
REASON: Brief explanation
POTENTIAL_ISSUE: One-sentence description of the potential issue (if YES)
"""

RECALL_USER_PROMPT = """Video description: {description}
Visual tags: {visual_tags}

Does this video contain content that is potentially problematic but NOT covered by
the existing policies? Think carefully about subtle violations, disguised advertising,
misleading information, or novel harmful patterns."""

POLICY_GEN_SYSTEM_PROMPT = """You are a content policy writer for a short-video platform.
Given a cluster of videos that share a common problematic pattern, write a clear and
actionable annotation policy for this new issue.

The policy should:
1. Name the issue clearly
2. Describe what content falls under this policy
3. Provide 2-3 example indicators
4. Specify the severity level (Low / Medium / High)
"""

POLICY_GEN_USER_PROMPT = """Here are sample descriptions from videos in this cluster:
{cluster_samples}

Write a content moderation policy for this cluster of videos."""


def format_recall_prompt(description: str, visual_tags: list[str]) -> tuple[str, str]:
    system = RECALL_SYSTEM_PROMPT.format(
        existing_policies=", ".join(EXISTING_POLICIES)
    )
    user = RECALL_USER_PROMPT.format(
        description=description,
        visual_tags=", ".join(visual_tags),
    )
    return system, user


def format_policy_prompt(cluster_samples: list[str]) -> tuple[str, str]:
    samples_text = "\n".join(f"- {s}" for s in cluster_samples[:5])
    return POLICY_GEN_SYSTEM_PROMPT, POLICY_GEN_USER_PROMPT.format(
        cluster_samples=samples_text
    )

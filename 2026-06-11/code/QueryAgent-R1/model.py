"""
QueryAgent-R1 Model
Paper: arxiv 2606.05671 (Alibaba International, June 2026)

Architecture:
  - LLM backbone for query generation (Qwen2 / similar instruction LLM)
  - Memory Abstraction Module: interest graph from user history
  - Chain-of-Retrieval: iterative query refinement via product retrieval
  - GRPO policy for RL training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class QueryGenerationOutput:
    query: str                          # Final generated query
    chain_of_retrieval: List[Dict]      # Intermediate retrieval steps
    refinements: int                    # Number of refinement iterations
    final_products: List[Dict]          # Products retrieved by final query
    confidence: float                   # Agent's confidence score


class MemoryAbstractionModule(nn.Module):
    """
    Extracts and compresses user interest graph from long-term behavior history.
    
    From paper Section 3.3:
        G_u = MemoryAbstract(H_u^{long})
        
    where H_u^{long} is the full behavior history (may be thousands of items).
    G_u is a compact interest graph with weighted category-level nodes.
    
    Implementation:
        1. Group history by category
        2. Compute category-level engagement weights
        3. Encode via category embedding + attention pooling
        4. Output: compressed user interest vector + top-K interest nodes
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_categories: int = 100,
        top_k_interests: int = 10,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.top_k = top_k_interests
        self.cat_embed = nn.Embedding(num_categories, embed_dim)
        self.behavior_type_embed = nn.Embedding(5, embed_dim)  # click/purchase/cart/wish/view

        # Attention-based pooling over history items
        self.attn_query = nn.Linear(embed_dim, embed_dim)
        self.attn_key = nn.Linear(embed_dim, embed_dim)
        self.attn_out = nn.Linear(embed_dim, hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(
        self,
        history_categories: torch.Tensor,   # [B, L] category indices
        history_behaviors: torch.Tensor,    # [B, L] behavior type indices
        history_weights: torch.Tensor,      # [B, L] engagement weights
        history_mask: torch.Tensor,         # [B, L] padding mask
    ) -> Tuple[torch.Tensor, List[List[Dict]]]:
        """
        Args:
            history_categories: [B, L] long int
            history_behaviors:  [B, L] long int
            history_weights:    [B, L] float [0, 1]
            history_mask:       [B, L] bool (True = valid)
        
        Returns:
            interest_embedding: [B, embed_dim]
            top_k_interests:    list of B interest graphs
        """
        # Embed categories and behaviors
        cat_emb = self.cat_embed(history_categories)    # [B, L, embed_dim]
        beh_emb = self.behavior_type_embed(history_behaviors)  # [B, L, embed_dim]
        item_emb = cat_emb + beh_emb  # [B, L, embed_dim]

        # Weight by engagement
        item_emb = item_emb * history_weights.unsqueeze(-1)  # [B, L, embed_dim]

        # Attention pooling: compressed representation
        Q = self.attn_query(item_emb.mean(dim=1, keepdim=True))  # [B, 1, embed_dim]
        K = self.attn_key(item_emb)  # [B, L, embed_dim]
        attn = (Q @ K.transpose(-1, -2)) / (item_emb.shape[-1] ** 0.5)  # [B, 1, L]
        attn = attn.masked_fill(~history_mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        pooled = (attn @ item_emb).squeeze(1)  # [B, embed_dim]

        out = self.output_proj(F.gelu(self.attn_out(pooled)))  # [B, embed_dim]
        interest_embedding = F.normalize(out, dim=-1)

        # Build top-k interest graph (for LLM prompt context)
        # Simplified: return top-k categories by weight
        B = history_categories.shape[0]
        top_k_interests = []
        for b in range(B):
            valid = history_mask[b]
            w = history_weights[b][valid]
            c = history_categories[b][valid]
            top_idx = w.topk(min(self.top_k, w.shape[0])).indices
            top_k_interests.append([
                {"category_idx": c[i].item(), "weight": w[i].item()}
                for i in top_idx
            ])

        return interest_embedding, top_k_interests


class QueryAgent(nn.Module):
    """
    QueryAgent-R1: Memory-augmented Agentic Query Generation with Chain-of-Retrieval.
    
    Chain-of-Retrieval Loop (from paper Algorithm 1):
        1. Generate initial query q_0 from user intent + memory context
        2. Retrieve products P(q_0)
        3. If consistency(P(q_0), user_intent) < threshold:
             Generate refined query q_1 conditioned on P(q_0)
             Go to step 2 (max T iterations)
        4. Return final query q_T
    
    The LLM policy π_θ is trained with GRPO using the consistency reward.
    """

    def __init__(
        self,
        llm_hidden_dim: int = 768,
        embed_dim: int = 256,
        num_categories: int = 100,
        max_refine_steps: int = 3,
        top_k_retrieval: int = 10,
    ):
        super().__init__()
        self.max_refine_steps = max_refine_steps
        self.top_k_retrieval = top_k_retrieval

        # Memory abstraction module
        self.memory_module = MemoryAbstractionModule(
            embed_dim=embed_dim, num_categories=num_categories
        )

        # Query quality estimator: predicts consistency score from products
        self.consistency_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # LLM backbone (interface; actual LLM loaded separately)
        # Placeholder projection for embedding-level operations
        self.intent_proj = nn.Linear(embed_dim, llm_hidden_dim)

    def compute_consistency_reward(
        self,
        retrieved_product_embeddings: torch.Tensor,  # [K, embed_dim]
        user_interest_embedding: torch.Tensor,       # [embed_dim]
    ) -> torch.Tensor:
        """
        Consistency reward: measures alignment between retrieved products and user intent.
        
        From paper Eq. (3):
            r_cons = Σ_k cos(e_product_k, e_user) / K
        
        Higher reward when retrieved products better match user interest profile.
        
        Returns scalar reward in [0, 1].
        """
        if retrieved_product_embeddings.shape[0] == 0:
            return torch.tensor(0.0)

        prod_emb = F.normalize(retrieved_product_embeddings, dim=-1)
        user_emb = F.normalize(user_interest_embedding.unsqueeze(0), dim=-1)
        sims = (user_emb @ prod_emb.T).squeeze(0)  # [K]
        return sims.mean()

    def compute_relevance_reward(
        self,
        query_embedding: torch.Tensor,   # [embed_dim]
        intent_embedding: torch.Tensor,  # [embed_dim]
    ) -> torch.Tensor:
        """
        Relevance reward: semantic similarity between generated query and user intent.
        
        From paper Eq. (4):
            r_rel = cos(e_query, e_intent)
        """
        q = F.normalize(query_embedding.unsqueeze(0), dim=-1)
        i = F.normalize(intent_embedding.unsqueeze(0), dim=-1)
        return (q @ i.T).squeeze()

    def compute_total_reward(
        self,
        consistency: torch.Tensor,
        relevance: torch.Tensor,
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> torch.Tensor:
        """
        Total reward combining consistency and relevance.
        
        From paper Eq. (5):
            r = α * r_cons + β * r_rel
            
        α > β: prioritize product-level consistency (CVR) over query relevance (CTR).
        """
        return alpha * consistency + beta * relevance

    def build_llm_prompt(
        self,
        user_query_intent: str,
        top_k_interests: List[Dict],
        retrieved_products: Optional[List[Dict]] = None,
        refinement_step: int = 0,
    ) -> str:
        """
        Build prompt for LLM query generation.
        
        Includes:
        - User search intent
        - Memory context (top-k interest categories)
        - Chain-of-retrieval context (previous retrieval results if refinement)
        """
        # Format interest graph as context
        interest_str = ", ".join([
            f"category_{d['category_idx']} (weight={d['weight']:.2f})"
            for d in top_k_interests[:5]
        ])

        prompt = f"""You are an e-commerce search query recommendation assistant.

User's search intent: {user_query_intent}
User's top interests based on history: {interest_str}
"""
        if retrieved_products and refinement_step > 0:
            prod_titles = [p["title"] for p in retrieved_products[:5]]
            prompt += f"""
Previous query retrieved these products: {prod_titles}
These products may not perfectly match the user's intent.
Please generate a refined search query that better aligns with the user's preferences.
"""
        else:
            prompt += "\nGenerate a search query recommendation for this user."

        return prompt


class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) trainer for QueryAgent-R1.
    
    From paper Section 3.4:
        GRPO samples G queries per prompt, computes rewards, and updates
        the policy using a group-normalized advantage.
    
    GRPO Objective (from DeepSeekMath-style GRPO):
        L_GRPO(θ) = -E[Σ_i A_i * log π_θ(q_i|x)]
        
    where:
        A_i = (r_i - mean(r)) / std(r)  [normalized group advantage]
        r_i = total_reward(q_i, retrieved_products, user_intent)
    """

    def __init__(
        self,
        agent: QueryAgent,
        retriever,
        group_size: int = 8,
        kl_coeff: float = 0.01,
        clip_ratio: float = 0.2,
    ):
        self.agent = agent
        self.retriever = retriever
        self.group_size = group_size
        self.kl_coeff = kl_coeff
        self.clip_ratio = clip_ratio

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize rewards within group to get advantages.
        
        A_i = (r_i - μ_r) / (σ_r + ε)
        """
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        return (rewards - mean_r) / std_r

    def grpo_loss(
        self,
        log_probs: torch.Tensor,      # [G] log probs of sampled queries
        ref_log_probs: torch.Tensor,  # [G] reference policy log probs
        rewards: torch.Tensor,        # [G] total rewards
    ) -> torch.Tensor:
        """
        GRPO policy gradient loss with KL penalty.
        
        L = -E[A_i * clip(π/π_old, 1-ε, 1+ε) * log π]
          + kl_coeff * KL(π || π_ref)
        
        Note: In QueryAgent-R1, π_ref = initial SFT model.
        """
        advantages = self.compute_group_advantages(rewards)

        # Ratio π_θ / π_ref
        ratio = torch.exp(log_probs - ref_log_probs)

        # Clipped surrogate objective
        clipped_ratio = ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio)
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages,
        ).mean()

        # KL divergence penalty: KL(π_θ || π_ref)
        kl_div = (log_probs - ref_log_probs).mean()
        total_loss = policy_loss + self.kl_coeff * kl_div

        return total_loss

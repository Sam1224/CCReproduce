"""
Product retrieval interface for QueryAgent-R1.
Implements BM25-based retrieval as a toy surrogate for the production
dense+sparse hybrid retrieval used in the paper (arXiv:2606.05671).
"""

import math
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict


class BM25Retriever:
    """
    BM25 retrieval over a product corpus.
    Used as a toy surrogate for the dense retrieval system in the paper.

    Parameters follow standard BM25: k1=1.5, b=0.75.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict] = []
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self._built = False

    def build(self, products: List[Dict]):
        """
        Build the BM25 index from a product corpus.

        Args:
            products: list of dicts with at least "id" and "text" keys
        """
        self.corpus = products
        self.doc_len = []
        word_doc_counts: Dict[str, int] = defaultdict(int)

        tokenized = []
        for prod in products:
            tokens = prod["text"].lower().split()
            tokenized.append(tokens)
            self.doc_len.append(len(tokens))
            for word in set(tokens):
                word_doc_counts[word] += 1

        N = len(products)
        self.avg_doc_len = sum(self.doc_len) / N if N > 0 else 1.0

        self.idf = {}
        for word, df in word_doc_counts.items():
            self.idf[word] = math.log((N - df + 0.5) / (df + 0.5) + 1)

        self._tokenized = tokenized
        self._built = True

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Retrieve top-k products for a given query string.

        Returns:
            list of product dicts with an added "score" key, sorted by score.
        """
        assert self._built, "Call build() before retrieve()."
        query_tokens = query.lower().split()
        scores = []

        for idx, tokens in enumerate(self._tokenized):
            tf_counts = Counter(tokens)
            doc_len = self.doc_len[idx]
            score = 0.0
            for qt in query_tokens:
                if qt not in self.idf:
                    continue
                tf = tf_counts.get(qt, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score += self.idf[qt] * numerator / denominator
            scores.append((score, idx))

        scores.sort(key=lambda x: -x[0])
        results = []
        for score, idx in scores[:top_k]:
            result = dict(self.corpus[idx])
            result["score"] = score
            results.append(result)
        return results


class ProductEmbeddingRetriever(nn.Module):
    """
    Dense product retriever using pre-computed product embeddings.
    Computes cosine similarity between query embed and product embeds.
    Used as the embedding layer for providing retrieved context to the generator.
    """

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.product_projector = nn.Linear(embed_dim, hidden_dim)

    def forward(
        self,
        query_embed: torch.Tensor,
        product_embeds: torch.Tensor,
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embed: (B, hidden_dim)
            product_embeds: (N, embed_dim) — pre-computed product pool
            top_k: number of products to retrieve

        Returns:
            retrieved_embeds: (B, top_k, hidden_dim)
            retrieved_indices: (B, top_k)
        """
        proj_products = self.product_projector(product_embeds)  # (N, hidden_dim)

        # Cosine similarity: (B, N)
        q_norm = query_embed / (query_embed.norm(dim=-1, keepdim=True) + 1e-8)
        p_norm = proj_products / (proj_products.norm(dim=-1, keepdim=True) + 1e-8)
        sims = q_norm @ p_norm.T  # (B, N)

        top_scores, top_indices = sims.topk(top_k, dim=-1)  # (B, top_k)
        retrieved_embeds = proj_products[top_indices]  # (B, top_k, hidden_dim)

        return retrieved_embeds, top_indices


class ChainOfRetrievalAgent:
    """
    Implements the Chain-of-Retrieval loop from QueryAgent-R1 (§3.3).

    Algorithm:
        1. Encode user history → user_embed (via InterestGraphMemory)
        2. Generate initial query q_0 from user_embed
        3. Retrieve products P_0 = Retrieve(q_0)
        4. Check if P_0 covers user intent (via consistency check)
        5. If not satisfied and iterations remain:
           - Refine query using P_0 as context → q_1
           - Repeat from step 3
        6. Return final query q* and retrieved products P*
    """

    def __init__(self, model, retriever: BM25Retriever, max_iterations: int = 3):
        self.model = model
        self.retriever = retriever
        self.max_iterations = max_iterations

    def run(
        self,
        item_embeds: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        tokenizer,
        item_mask: Optional[torch.Tensor] = None,
        top_k: int = 5,
    ) -> Dict:
        """
        Execute the Chain-of-Retrieval loop.

        Returns:
            dict with keys: "query", "retrieved_products", "iteration"
        """
        self.model.eval()
        with torch.no_grad():
            user_embed, interest_nodes = self.model.encode_user(item_embeds, item_mask)

        query_ids = None
        retrieved = []
        retrieved_context = None

        for iteration in range(self.max_iterations):
            # Generate query
            with torch.no_grad():
                query_ids = self.model.generate_query(
                    user_embed=user_embed,
                    bos_token_id=bos_token_id,
                    eos_token_id=eos_token_id,
                    retrieved_context=retrieved_context,
                    max_new_tokens=16,
                )

            query_text = tokenizer.decode(query_ids[0].tolist(), skip_special_tokens=True)

            # Retrieve products
            retrieved = self.retriever.retrieve(query_text, top_k=top_k)

            # Simple consistency check: if we have enough highly scored products, stop
            if retrieved and retrieved[0]["score"] > 2.0:
                break

            # Update retrieved context for next iteration (placeholder embeddings)
            # In the paper, dense product embeddings are used here
            retrieved_context = None

        return {
            "query": query_text if query_ids is not None else "",
            "retrieved_products": retrieved,
            "iteration": iteration + 1,
        }

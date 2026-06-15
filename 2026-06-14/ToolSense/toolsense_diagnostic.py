"""
ToolSense – Toy reproduction of core ideas from:
  "ToolSense: A Diagnostic Framework for Auditing Parametric Tool Knowledge in LLMs"
  SAP Labs, arXiv 2606.12451

Concepts illustrated:
  1. Tool catalog representation
  2. Automatic Realistic Retrieval Benchmark (RRB) generation with ambiguity tiers
  3. MCQ and QA probing generation
  4. Internalization Score: gap between constrained vs free-form retrieval accuracy

Key insight from the paper:
  Standard ToolBench benchmarks use verbose queries + constrained decoding (trie search),
  which can inflate scores. ToolSense exposes "knowledge-retrieval dissociation" by
  testing with shorter, ambiguous queries and free-form generation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# 1. Tool catalog
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    description: str
    parameters: list[str]
    category: str
    keywords: list[str] = field(default_factory=list)


SAMPLE_TOOLS: list[Tool] = [
    Tool("SearchProducts", "Search for products matching a query on the e-commerce platform",
         ["query", "category", "max_price"], "search",
         ["search", "product", "catalog", "find"]),
    Tool("GetProductDetails", "Retrieve detailed information about a specific product by ID",
         ["product_id"], "retrieval",
         ["details", "product", "info", "attributes"]),
    Tool("PlaceOrder", "Place an order for a product",
         ["product_id", "quantity", "address"], "transaction",
         ["order", "buy", "purchase", "checkout"]),
    Tool("TrackShipment", "Track the delivery status of a placed order",
         ["order_id"], "logistics",
         ["track", "delivery", "shipment", "status"]),
    Tool("WriteProductReview", "Submit a review for a purchased product",
         ["product_id", "rating", "comment"], "content",
         ["review", "rating", "feedback", "comment"]),
    Tool("GetRecommendations", "Get personalised product recommendations for a user",
         ["user_id", "limit"], "recommendation",
         ["recommend", "suggest", "personalise"]),
    Tool("CheckInventory", "Check whether a product is in stock",
         ["product_id", "sku"], "inventory",
         ["stock", "inventory", "available"]),
    Tool("ApplyCoupon", "Apply a discount coupon to a shopping cart",
         ["coupon_code", "cart_id"], "promotion",
         ["coupon", "discount", "promo", "voucher"]),
]


# ---------------------------------------------------------------------------
# 2. Realistic Retrieval Benchmark (RRB) generation
# ---------------------------------------------------------------------------

AMBIGUITY_LEVELS = {
    "low": "Use the exact tool name and all keywords",
    "medium": "Use partial keywords; drop the tool name",
    "high": "Use only vague user intent without keywords",
}

def generate_rrb_query(tool: Tool, ambiguity: str) -> str:
    """Auto-generate a realistic query at the specified ambiguity tier."""
    if ambiguity == "low":
        return f"I need to use {tool.name} to {tool.description.lower()}"
    elif ambiguity == "medium":
        kw = random.choice(tool.keywords) if tool.keywords else tool.category
        return f"How do I {kw} something on this platform?"
    else:  # high
        # Just describe user intent vaguely
        intent_map = {
            "search": "I want to find something",
            "retrieval": "I need more info about an item",
            "transaction": "I want to get something",
            "logistics": "Where is my thing?",
            "content": "I want to share my opinion",
            "recommendation": "What should I look at?",
            "inventory": "Is it available?",
            "promotion": "I have a code",
        }
        return intent_map.get(tool.category, "I need help with something")


def generate_mcq_probe(tool: Tool, all_tools: list[Tool]) -> dict:
    """MCQ probing: test whether model knows tool capabilities."""
    distractors = [t for t in all_tools if t.name != tool.name]
    random.shuffle(distractors)
    options = [tool.name] + [t.name for t in distractors[:3]]
    random.shuffle(options)
    return {
        "question": f"Which tool should I use to: '{tool.description}'?",
        "options": options,
        "answer": tool.name,
    }


def generate_qa_probe(tool: Tool) -> dict:
    """Open-ended QA probe: what parameters does this tool require?"""
    return {
        "question": f"What parameters are required for the tool '{tool.name}'?",
        "answer": ", ".join(tool.parameters),
    }


# ---------------------------------------------------------------------------
# 3. Simulated LLM retriever (constrained vs free-form)
# ---------------------------------------------------------------------------

class ParametricRetriever:
    """
    Simulates a parametric retriever:
    - constrained: can look up an exact trie (perfect for verbose queries)
    - free_form: uses keyword overlap (more realistic but noisier)
    """

    def __init__(self, tools: list[Tool]) -> None:
        self.tools = tools

    def retrieve_constrained(self, query: str) -> Optional[str]:
        """Constrained decoding simulation: check exact name match in query."""
        for tool in self.tools:
            if tool.name.lower() in query.lower():
                return tool.name
        return self._retrieve_free_form(query)

    def _retrieve_free_form(self, query: str) -> Optional[str]:
        """Keyword-overlap based retrieval (simulates free-form generation)."""
        q_tokens = set(query.lower().split())
        scores: dict[str, float] = {}
        for tool in self.tools:
            overlap = q_tokens & set(tool.keywords)
            # Also penalise for missing key terms
            score = len(overlap) / (len(tool.keywords) + 1e-9)
            scores[tool.name] = score
        if not scores:
            return None
        best = max(scores, key=lambda k: scores[k])
        # Return None if confidence too low (simulates ambiguity)
        return best if scores[best] > 0.15 else None

    def retrieve_free_form(self, query: str) -> Optional[str]:
        return self._retrieve_free_form(query)


# ---------------------------------------------------------------------------
# 4. Internalization Score computation
# ---------------------------------------------------------------------------

def compute_internalization_score(
    retriever: ParametricRetriever,
    queries_by_tool: dict[str, list[tuple[str, str]]],  # tool_name → [(query, ambiguity)]
) -> dict:
    """
    Internalization Score = constrained_accuracy - free_form_accuracy.
    A large gap indicates the model depends heavily on constrained decoding (trie)
    and does not truly "understand" the tool.
    """
    constrained_hits = 0
    free_form_hits = 0
    total = 0

    for tool_name, query_list in queries_by_tool.items():
        for query, _ in query_list:
            c_pred = retriever.retrieve_constrained(query)
            f_pred = retriever.retrieve_free_form(query)
            if c_pred == tool_name:
                constrained_hits += 1
            if f_pred == tool_name:
                free_form_hits += 1
            total += 1

    c_acc = constrained_hits / total if total else 0.0
    f_acc = free_form_hits / total if total else 0.0
    return {
        "constrained_accuracy": c_acc,
        "free_form_accuracy": f_acc,
        "internalization_score": c_acc - f_acc,
    }


# ---------------------------------------------------------------------------
# 5. Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(0)
    retriever = ParametricRetriever(SAMPLE_TOOLS)

    print("=== ToolSense Diagnostic Framework Demo ===\n")

    # Generate RRB at all ambiguity tiers
    queries_by_tool: dict[str, list[tuple[str, str]]] = {}
    for tool in SAMPLE_TOOLS:
        queries_by_tool[tool.name] = []
        for ambiguity in AMBIGUITY_LEVELS:
            q = generate_rrb_query(tool, ambiguity)
            queries_by_tool[tool.name].append((q, ambiguity))

    # Per-ambiguity accuracy
    print("--- RRB: Per-Ambiguity Retrieval Accuracy ---")
    for ambiguity in AMBIGUITY_LEVELS:
        hits = 0
        total = 0
        for tool in SAMPLE_TOOLS:
            q, _ = queries_by_tool[tool.name][list(AMBIGUITY_LEVELS).index(ambiguity)]
            pred = retriever.retrieve_constrained(q)
            hits += (pred == tool.name)
            total += 1
        print(f"  {ambiguity:8s} ambiguity → constrained accuracy: {hits}/{total} = {hits/total:.2f}")

    # MCQ probing
    print("\n--- MCQ Tool Probing (sample) ---")
    for tool in SAMPLE_TOOLS[:3]:
        mcq = generate_mcq_probe(tool, SAMPLE_TOOLS)
        print(f"  Q: {mcq['question']}")
        print(f"     Options: {mcq['options']}")
        print(f"     Answer:  {mcq['answer']}\n")

    # QA probing
    print("--- QA Probing (sample) ---")
    for tool in SAMPLE_TOOLS[:3]:
        qa = generate_qa_probe(tool)
        print(f"  Q: {qa['question']}")
        print(f"  A: {qa['answer']}\n")

    # Internalization Score
    scores = compute_internalization_score(retriever, queries_by_tool)
    print("--- Internalization Score ---")
    print(f"  Constrained accuracy:       {scores['constrained_accuracy']:.3f}")
    print(f"  Free-form accuracy:         {scores['free_form_accuracy']:.3f}")
    print(f"  Internalization Score (gap): {scores['internalization_score']:.3f}")
    print("\n  A large gap (>0.3) means the model relies heavily on constrained decoding")
    print("  and may not truly understand tool functionality — a deployment risk.")


if __name__ == "__main__":
    main()

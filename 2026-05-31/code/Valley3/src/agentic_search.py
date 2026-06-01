"""
Agentic Search interface for Valley3 (toy implementation).

Paper (Section 4.3):
Valley3 is equipped with agentic search capabilities to proactively invoke
search tools and acquire task-relevant information for e-commerce deep research
tasks (e.g., policy queries, competitor analysis, product verification).

In production: the model generates structured tool calls (JSON) and the
platform executes them against real databases/APIs.

This toy simulates the tool call interface.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json


@dataclass
class SearchResult:
    query: str
    results: List[Dict[str, Any]]
    source: str = "toy_knowledge_base"


class ToyKnowledgeBase:
    """Simulated e-commerce policy + product knowledge base."""
    POLICY_RULES = {
        "efficacy": "Rule 3.2.1: Product efficacy claims must be supported by clinical evidence.",
        "pricing": "Rule 4.1.0: Price comparisons must use actual reference prices from last 7 days.",
        "prohibited": "Rule 1.0.0: The following categories are prohibited: tobacco, counterfeit goods.",
        "influencer": "Rule 5.2.0: Sponsored content must be clearly disclosed within first 30 seconds.",
    }
    PRODUCTS = {
        "P001": {"name": "Slimming Tea", "category": "health", "risk_level": "high"},
        "P002": {"name": "Genuine Sneakers", "category": "footwear", "risk_level": "low"},
        "P003": {"name": "Whitening Cream", "category": "cosmetics", "risk_level": "medium"},
    }

    def search_policy(self, keywords: List[str]) -> List[Dict]:
        results = []
        for kw in keywords:
            for key, rule in self.POLICY_RULES.items():
                if kw.lower() in rule.lower() or kw.lower() in key:
                    results.append({"rule_id": key, "content": rule})
        return results

    def search_product(self, product_id: str) -> Optional[Dict]:
        return self.PRODUCTS.get(product_id)


@dataclass
class AgentAction:
    tool: str                          # "search_policy" | "search_product" | "final_answer"
    params: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


class AgenticSearchAgent:
    """
    Valley3-style agentic loop: model decides when to call tools.
    Paper: "Valley3 proactively invokes search tools to acquire task-relevant
    information for e-commerce deep research tasks."
    """
    def __init__(self, kb: ToyKnowledgeBase = None, max_steps: int = 5):
        self.kb = kb or ToyKnowledgeBase()
        self.max_steps = max_steps

    def plan_actions(self, query: str) -> List[AgentAction]:
        """Simulate model planning: which tools to call for this query."""
        actions = []
        query_lower = query.lower()
        if "policy" in query_lower or "rule" in query_lower or "compliance" in query_lower:
            keywords = [w for w in query_lower.split() if len(w) > 4][:3]
            actions.append(AgentAction(
                tool="search_policy",
                params={"keywords": keywords},
                reasoning="Query involves policy/compliance → retrieve relevant rules"
            ))
        if "product" in query_lower or "P00" in query:
            product_id = next((w for w in query.split() if w.startswith("P0")), "P001")
            actions.append(AgentAction(
                tool="search_product",
                params={"product_id": product_id},
                reasoning="Query references product → look up product risk profile"
            ))
        actions.append(AgentAction(
            tool="final_answer",
            params={},
            reasoning="Synthesize retrieved information into final answer"
        ))
        return actions

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute agentic search loop and return final answer with trace."""
        actions = self.plan_actions(query)
        context_parts = [f"Query: {query}"]
        trace = []

        for i, action in enumerate(actions[:self.max_steps]):
            if action.tool == "search_policy":
                results = self.kb.search_policy(action.params["keywords"])
                context_parts.append(f"Policy Results: {json.dumps(results, indent=2)}")
                trace.append({"step": i, "tool": "search_policy",
                               "reasoning": action.reasoning, "results": results})
            elif action.tool == "search_product":
                result = self.kb.search_product(action.params["product_id"])
                context_parts.append(f"Product Info: {json.dumps(result, indent=2)}")
                trace.append({"step": i, "tool": "search_product",
                               "reasoning": action.reasoning, "result": result})
            elif action.tool == "final_answer":
                # In real Valley3: LLM generates final answer from accumulated context
                answer = f"[AgentResult] Based on retrieved context:\n" + \
                         "\n".join(context_parts[1:])
                trace.append({"step": i, "tool": "final_answer"})
                return {"answer": answer, "trace": trace, "context": "\n".join(context_parts)}

        return {"answer": "Max steps reached", "trace": trace}

"""
Agentic Search for Valley3 (Valley3 §3.4)

Valley3 is equipped with agentic search capabilities to proactively invoke
search tools and acquire task-relevant information for e-commerce deep research.

This module implements the search tool interface and the tool-call parsing loop.
In production Valley3:
  - Tools: product_search, policy_lookup, review_search, competitor_analysis
  - Model generates tool calls in a structured format (function-calling style)
  - Results are fed back to model for multi-turn reasoning
"""

import json
import re
from typing import Dict, List, Optional, Callable, Any


# --- Tool Definitions (Valley3 §3.4 e-commerce search tools) ---

class ProductSearchTool:
    """Search product database by keyword, category, or image similarity."""
    name = "product_search"
    description = "Search e-commerce product database. Args: query (str), category (str, optional), limit (int, default=5)"

    def __call__(self, query: str, category: Optional[str] = None, limit: int = 5) -> List[Dict]:
        # Toy implementation — returns mock products
        return [
            {
                "product_id": f"P{i:04d}",
                "title": f"[Mock] {query} Product {i+1}",
                "category": category or "General",
                "price": round(29.99 + i * 10, 2),
                "rating": round(4.0 + i * 0.1, 1),
                "violation_flags": [] if i < 3 else ["false_efficacy_claim"],
            }
            for i in range(min(limit, 5))
        ]


class PolicyLookupTool:
    """Look up platform content policies by keyword or violation type."""
    name = "policy_lookup"
    description = "Look up content moderation policies. Args: keyword (str), violation_type (str, optional)"

    # Mock policy database
    POLICIES = {
        "health_claim": "Content must not make unverified health claims. Prohibited: 'cures', 'treats', 'heals'.",
        "price_guarantee": "Price comparisons must reference verifiable retail prices from the past 90 days.",
        "minor_protection": "Content featuring minors must comply with COPPA and platform minor safety guidelines.",
        "counterfeit": "All branded products must be authentic. Counterfeit indicators include unauthorized logos.",
        "financial_promise": "Investment returns cannot be guaranteed. Risk disclosure is mandatory.",
    }

    def __call__(self, keyword: str, violation_type: Optional[str] = None) -> Dict:
        key = violation_type or keyword.lower().replace(" ", "_")
        policy = self.POLICIES.get(key, f"[Mock policy] No specific policy found for '{keyword}'.")
        return {"policy_id": key, "content": policy, "severity": "HIGH" if violation_type else "MEDIUM"}


class ReviewSearchTool:
    """Search user reviews for a product."""
    name = "review_search"
    description = "Search user reviews. Args: product_id (str), sentiment (str: positive/negative/all, default=all), limit (int, default=3)"

    def __call__(self, product_id: str, sentiment: str = "all", limit: int = 3) -> List[Dict]:
        sentiments = ["positive", "negative", "neutral"]
        return [
            {
                "review_id": f"R{i:04d}",
                "product_id": product_id,
                "sentiment": sentiments[i % 3],
                "text": f"[Mock review {i+1}] This product is {'great' if i % 2 == 0 else 'disappointing'}.",
                "helpful_votes": i * 5,
            }
            for i in range(min(limit, 5))
            if sentiment == "all" or sentiments[i % 3] == sentiment
        ]


# --- Tool Registry ---

TOOLS = {
    "product_search": ProductSearchTool(),
    "policy_lookup": PolicyLookupTool(),
    "review_search": ReviewSearchTool(),
}


# --- Tool-Call Parser ---

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def parse_tool_calls(model_output: str) -> List[Dict]:
    """
    Parse tool calls from model output.
    Valley3 uses a structured tool-call format:
      <tool_call>{"name": "product_search", "args": {"query": "vitamin C"}}</tool_call>
    """
    calls = []
    for match in TOOL_CALL_PATTERN.finditer(model_output):
        try:
            call = json.loads(match.group(1))
            calls.append(call)
        except json.JSONDecodeError:
            continue
    return calls


def execute_tool_call(call: Dict) -> str:
    """Execute a single tool call and return the result as a string."""
    name = call.get("name", "")
    args = call.get("args", {})
    if name not in TOOLS:
        return f"[Error] Unknown tool: {name}"
    try:
        result = TOOLS[name](**args)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"[Error] Tool execution failed: {str(e)}"


# --- Agentic Search Loop ---

class AgenticSearchLoop:
    """
    Multi-turn agentic search loop for e-commerce deep research.

    Valley3 §3.4: "Valley3 is equipped with agentic search capabilities to
    proactively invoke search tools and acquire task-relevant information
    for e-commerce deep research tasks."

    The loop:
      1. Model generates response (possibly with tool calls)
      2. Tool calls are parsed and executed
      3. Tool results are appended to context
      4. Model continues reasoning until no more tool calls
    """

    def __init__(self, model_generate_fn: Callable[[str], str], max_turns: int = 5):
        self.model_generate = model_generate_fn
        self.max_turns = max_turns

    def run(self, user_query: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the agentic search loop.
        Returns the final response and the full conversation history.
        """
        if system_prompt is None:
            system_prompt = (
                "You are Valley3, an e-commerce AI assistant with agentic search capabilities. "
                "Available tools: product_search, policy_lookup, review_search. "
                "Use <tool_call>{...}</tool_call> to invoke tools when needed."
            )

        context = f"System: {system_prompt}\nUser: {user_query}\nAssistant:"
        history = [{"role": "user", "content": user_query}]
        all_tool_calls = []

        for turn in range(self.max_turns):
            model_output = self.model_generate(context)
            tool_calls = parse_tool_calls(model_output)

            if not tool_calls:
                # No more tool calls — final answer
                history.append({"role": "assistant", "content": model_output})
                break

            # Execute tool calls and append results
            tool_results = []
            for call in tool_calls:
                result = execute_tool_call(call)
                tool_results.append({
                    "tool": call.get("name"),
                    "args": call.get("args", {}),
                    "result": result,
                })
                all_tool_calls.append(call)

            # Append tool calls + results to context
            tool_result_str = "\n".join(
                f"Tool({r['tool']}): {r['result']}" for r in tool_results
            )
            context += f"\n{model_output}\nTool Results:\n{tool_result_str}\nAssistant:"
            history.append({"role": "tool_interaction", "content": tool_result_str})

        return {
            "final_answer": model_output if not tool_calls else "[max turns reached]",
            "history": history,
            "tool_calls": all_tool_calls,
            "turns": turn + 1,
        }


# --- Demo Usage ---

def demo_mock_model(context: str) -> str:
    """Mock model generate function for demonstration."""
    if "tool_call" not in context and "Tool Results" not in context:
        # First turn: issue a tool call
        return (
            'I need to research this product. '
            '<tool_call>{"name": "product_search", "args": {"query": "vitamin C supplement", "limit": 3}}</tool_call>'
        )
    elif context.count("Tool Results") == 1:
        # Second turn: check policy
        return (
            'Let me also check the policy for health claims. '
            '<tool_call>{"name": "policy_lookup", "args": {"keyword": "health claim", "violation_type": "health_claim"}}</tool_call>'
        )
    else:
        # Final answer
        return (
            "Based on my research: the product has potential health claim violations. "
            "Policy prohibits unverified claims. Recommend: REMOVE pending review."
        )


if __name__ == "__main__":
    loop = AgenticSearchLoop(model_generate_fn=demo_mock_model, max_turns=5)
    result = loop.run("Analyze this vitamin C product for policy compliance.")
    print("=== Agentic Search Result ===")
    print(f"Final Answer: {result['final_answer']}")
    print(f"Turns taken: {result['turns']}")
    print(f"Tool calls made: {len(result['tool_calls'])}")
    for tc in result["tool_calls"]:
        print(f"  - {tc['name']}({tc.get('args', {})})")

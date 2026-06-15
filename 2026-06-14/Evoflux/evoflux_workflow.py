"""
Evoflux – Toy reproduction of core ideas from:
  "Evoflux: Inference-Time Evolution of Executable Tool Workflows for Compact Agents"
  Rensselaer Polytechnic Institute & IBM Research, arXiv 2606.12674

Concepts illustrated:
  1. Typed workflow graph: tool calls as a sequence with schema + dependency constraints
  2. Structural edit operations: insert prerequisite, fix param, replace tool, fix type
  3. Execution feedback loop: each workflow is executed and feasibility is checked
  4. Evolutionary search with diversity pruning

Key finding from the paper:
  On MCP-Bench (28 servers, 250 tools), Evoflux raises compact-model execution
  feasibility from ~3% to 17–24%.  Small-corpus SFT/DPO often *decreases* feasibility.
"""

from __future__ import annotations

import random
import copy
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# 1. Tool catalog
# ---------------------------------------------------------------------------

@dataclass
class ToolParam:
    name: str
    ptype: str         # "string", "int", "list"
    required: bool = True
    depends_on: Optional[str] = None  # context key that must exist before this step

@dataclass
class ToolDef:
    name: str
    params: list[ToolParam]
    output_key: str

CATALOG: dict[str, ToolDef] = {
    "SearchProducts": ToolDef(
        "SearchProducts",
        [ToolParam("query", "string"), ToolParam("category", "string", required=False)],
        "product_list",
    ),
    "FilterByPrice": ToolDef(
        "FilterByPrice",
        [ToolParam("products", "list", depends_on="product_list"),
         ToolParam("max_price", "int")],
        "filtered_products",
    ),
    "GetProductDetails": ToolDef(
        "GetProductDetails",
        [ToolParam("product_id", "string", depends_on="filtered_products")],
        "product_details",
    ),
    "PlaceOrder": ToolDef(
        "PlaceOrder",
        [ToolParam("product_id", "string", depends_on="product_details"),
         ToolParam("quantity", "int")],
        "order_id",
    ),
}


# ---------------------------------------------------------------------------
# 2. Workflow and executor
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    tool_name: str
    params: dict[str, Any]

    def __repr__(self) -> str:
        return f"{self.tool_name}({self.params})"

@dataclass
class Workflow:
    steps: list[ToolCall] = field(default_factory=list)
    def clone(self) -> "Workflow":
        return Workflow(steps=copy.deepcopy(self.steps))


def execute(workflow: Workflow, catalog: dict[str, ToolDef]) -> tuple[bool, str]:
    context: dict[str, Any] = {}
    for i, step in enumerate(workflow.steps):
        if step.tool_name not in catalog:
            return False, f"Step {i}: unknown tool '{step.tool_name}'"
        tool = catalog[step.tool_name]
        for p in tool.params:
            if p.required and p.name not in step.params:
                return False, f"Step {i} ({step.tool_name}): missing required param '{p.name}'"
            if p.depends_on and p.depends_on not in context:
                return False, (f"Step {i} ({step.tool_name}): param '{p.name}' "
                               f"needs '{p.depends_on}' (not yet in context)")
            if p.name in step.params:
                val = step.params[p.name]
                if p.ptype == "int" and not isinstance(val, int):
                    return False, f"Step {i} ({step.tool_name}): '{p.name}' must be int, got {type(val).__name__}"
                if p.ptype == "list" and not isinstance(val, list):
                    return False, f"Step {i} ({step.tool_name}): '{p.name}' must be list"
        context[tool.output_key] = f"<{step.tool_name}_output>"
    return True, "OK"


# ---------------------------------------------------------------------------
# 3. Edit operations
# ---------------------------------------------------------------------------

def _make_step(tool: ToolDef, context_keys: set[str]) -> ToolCall:
    """Create a tool call with all required non-dep params filled."""
    params: dict[str, Any] = {}
    for p in tool.params:
        if p.required:
            if p.depends_on and p.depends_on in context_keys:
                params[p.name] = [f"from_{p.depends_on}"] if p.ptype == "list" else f"from_{p.depends_on}"
            elif p.ptype == "int":
                params[p.name] = random.randint(1, 500)
            elif p.ptype == "list":
                params[p.name] = ["item1"]
            else:
                params[p.name] = f"val_{p.name}"
    return ToolCall(tool.name, params)


def edit_insert_prerequisite(wf: Workflow, catalog: dict[str, ToolDef]) -> Workflow:
    """Insert a prerequisite tool at the beginning of the workflow."""
    new_wf = wf.clone()
    tool = random.choice(list(catalog.values()))
    new_wf.steps.insert(0, _make_step(tool, set()))
    return new_wf

def edit_replace_tool(wf: Workflow, catalog: dict[str, ToolDef]) -> Workflow:
    """Replace a random step with a valid tool from the catalog."""
    new_wf = wf.clone()
    if not new_wf.steps:
        return edit_insert_prerequisite(new_wf, catalog)
    idx = random.randint(0, len(new_wf.steps) - 1)
    tool = random.choice(list(catalog.values()))
    ctx_before: set[str] = set()
    for j, s in enumerate(new_wf.steps):
        if j == idx:
            break
        if s.tool_name in catalog:
            ctx_before.add(catalog[s.tool_name].output_key)
    new_wf.steps[idx] = _make_step(tool, ctx_before)
    return new_wf

def edit_fix_param_type(wf: Workflow, catalog: dict[str, ToolDef]) -> Workflow:
    """Fix int/list type errors in params."""
    new_wf = wf.clone()
    for step in new_wf.steps:
        if step.tool_name not in catalog:
            continue
        for p in catalog[step.tool_name].params:
            if p.name in step.params:
                val = step.params[p.name]
                if p.ptype == "int" and not isinstance(val, int):
                    try:
                        step.params[p.name] = int(val)
                    except (ValueError, TypeError):
                        step.params[p.name] = 1
                if p.ptype == "list" and not isinstance(val, list):
                    step.params[p.name] = [str(val)]
    return new_wf

def edit_drop_unknown(wf: Workflow, catalog: dict[str, ToolDef]) -> Workflow:
    """Remove steps that reference unknown tools."""
    new_wf = wf.clone()
    new_wf.steps = [s for s in new_wf.steps if s.tool_name in catalog]
    return new_wf

EDITS = [edit_insert_prerequisite, edit_replace_tool, edit_fix_param_type, edit_drop_unknown]


# ---------------------------------------------------------------------------
# 4. Evolutionary search
# ---------------------------------------------------------------------------

def evolutionary_search(
    initial: Workflow,
    catalog: dict[str, ToolDef],
    budget: int = 40,
    pop_size: int = 6,
) -> tuple[Optional[Workflow], list[bool]]:
    population = [initial.clone() for _ in range(pop_size)]
    log: list[bool] = []

    for _ in range(budget):
        for wf in population:
            ok, _ = execute(wf, catalog)
            log.append(ok)
            if ok:
                return wf, log
        # Mutate
        next_gen: list[Workflow] = []
        for wf in population:
            fn = random.choice(EDITS)
            try:
                mutant = fn(wf, catalog)
            except Exception:
                mutant = wf.clone()
            next_gen.append(mutant)
        population = next_gen

    return None, log


# ---------------------------------------------------------------------------
# 5. Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(7)

    # Broken workflow: wrong tool, missing deps, wrong type
    broken = Workflow(steps=[
        ToolCall("FilterByPrice", {"max_price": "cheap"}),   # missing dep + wrong type
        ToolCall("GhostTool", {"x": 1}),                      # unknown tool
        ToolCall("PlaceOrder", {"quantity": "two"}),           # missing dep + wrong type
    ])

    print("=== Initial (broken) workflow ===")
    ok, err = execute(broken, CATALOG)
    print(f"  Feasible: {ok}  →  {err}\n")

    print("=== Evoflux evolutionary repair (budget=40, pop=6) ===")
    best, log = evolutionary_search(broken, CATALOG, budget=40, pop_size=6)

    total = len(log)
    rate = sum(log) / total if total else 0.0
    print(f"  Total evaluations:       {total}")
    print(f"  Feasibility rate:        {rate:.2%}")

    if best:
        first_success = log.index(True) + 1
        print(f"  First feasible found at: attempt {first_success}/{total}")
        print(f"\n  ✓ Repaired workflow:")
        for i, s in enumerate(best.steps):
            print(f"    {i+1}. {s}")
        ok2, msg2 = execute(best, CATALOG)
        print(f"\n  Verification: feasible={ok2}  msg={msg2}")
    else:
        print("  No feasible workflow found within budget.")

    print("\n=== Feasibility progression (first 40 attempts) ===")
    bar = "".join("✓" if f else "✗" for f in log[:40])
    print(f"  {bar}")
    print("\n  In the paper: compact agents improve from ~3% → 17–24% on MCP-Bench with Evoflux.")
    print("  Small-corpus SFT/DPO often decreases feasibility — test-time repair is more robust.")


if __name__ == "__main__":
    main()

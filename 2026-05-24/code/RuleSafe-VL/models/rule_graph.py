"""
Rule graph formalism for RuleSafe-VL.

Based on: "RuleSafe-VL: Evaluating Rule-Conditioned Decision Reasoning
in Vision-Language Content Moderation" (arXiv:2605.07760)

The paper formalizes platform moderation policies as:
  - 93 atomic rules (each rule = a specific policy constraint)
  - 92 typed rule relations (conditional, exclusive, priority, etc.)
  - Rule activation: given content, which rules are triggered
  - Rule interaction: how triggered rules combine to yield a decision
  - Sufficiency: whether available evidence is enough to decide
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Set
import json
import networkx as nx


class RelationType(Enum):
    CONDITIONAL = "conditional"    # Rule A activates only if Rule B is also active
    EXCLUSIVE = "exclusive"        # If Rule A activates, Rule B cannot simultaneously
    PRIORITY = "priority"          # Rule A overrides Rule B when both active
    CONJUNCTION = "conjunction"    # Both Rule A and B must activate for consequence
    DISJUNCTION = "disjunction"    # Either Rule A or B activating triggers consequence


class PolicyFamily(Enum):
    HATE_SPEECH = "hate_speech"
    ADULT_CONTENT = "adult_content"
    FRAUD_MISLEADING = "fraud_misleading"


class ModerationDecision(Enum):
    ALLOWED = "allowed"
    RESTRICTED = "restricted"   # Requires age gate or warning
    REMOVED = "removed"         # Must be taken down


@dataclass
class AtomicRule:
    rule_id: str
    description: str
    policy_family: PolicyFamily
    keywords: List[str] = field(default_factory=list)
    requires_visual_evidence: bool = False
    requires_text_evidence: bool = False

    def to_prompt_fragment(self) -> str:
        return (
            f"Rule {self.rule_id} ({self.policy_family.value}): {self.description}. "
            f"Evidence required: {'visual' if self.requires_visual_evidence else ''}"
            f"{'+' if self.requires_visual_evidence and self.requires_text_evidence else ''}"
            f"{'text' if self.requires_text_evidence else ''}."
        )


@dataclass
class RuleRelation:
    relation_id: str
    source_rule_id: str
    target_rule_id: str
    relation_type: RelationType

    def to_prompt_fragment(self) -> str:
        return (
            f"Relation {self.relation_id}: Rule {self.source_rule_id} "
            f"[{self.relation_type.value}] Rule {self.target_rule_id}."
        )


class RuleGraph:
    """
    Directed graph encoding platform moderation policy rules and their relations.

    Paper formalizes 93 atomic rules + 92 typed relations across 3 policy families.
    This implementation mirrors the structure; toy data uses a small subset.
    """

    def __init__(self):
        self.rules: Dict[str, AtomicRule] = {}
        self.relations: Dict[str, RuleRelation] = {}
        self.graph = nx.DiGraph()

    def add_rule(self, rule: AtomicRule):
        self.rules[rule.rule_id] = rule
        self.graph.add_node(rule.rule_id, rule=rule)

    def add_relation(self, relation: RuleRelation):
        self.relations[relation.relation_id] = relation
        self.graph.add_edge(
            relation.source_rule_id,
            relation.target_rule_id,
            relation_type=relation.relation_type,
            relation_id=relation.relation_id,
        )

    def get_activated_rules(self, activated_ids: Set[str]) -> List[AtomicRule]:
        return [self.rules[rid] for rid in activated_ids if rid in self.rules]

    def get_rule_interactions(self, activated_ids: Set[str]) -> List[RuleRelation]:
        """Return relations where both source and target are in the activated set."""
        interactions = []
        for rel in self.relations.values():
            if rel.source_rule_id in activated_ids and rel.target_rule_id in activated_ids:
                interactions.append(rel)
        return interactions

    def compute_decision_from_activations(
        self, activated_ids: Set[str]
    ) -> ModerationDecision:
        """
        Simple deterministic decision logic based on rule activations.
        Paper does not specify exact logic (left to platform policy); this
        implements a plausible heuristic:
          - Any REMOVED-category rule active → REMOVED
          - Any RESTRICTED-category rule active (no REMOVED) → RESTRICTED
          - Otherwise → ALLOWED
        """
        # Pseudocode: in the real system, decision depends on rule interactions
        # Priority relations override lower-priority rules
        # Exclusive relations invalidate conflicting rule pairs
        # Conditional relations require co-activation

        active_rules = self.get_activated_rules(activated_ids)

        # Check for priority overrides
        interactions = self.get_rule_interactions(activated_ids)
        overridden_rules = set()
        for rel in interactions:
            if rel.relation_type == RelationType.PRIORITY:
                overridden_rules.add(rel.target_rule_id)

        effective_rules = [r for r in active_rules if r.rule_id not in overridden_rules]

        # Map policy families to decisions (simplified)
        if any(r.policy_family == PolicyFamily.ADULT_CONTENT for r in effective_rules):
            return ModerationDecision.REMOVED
        if any(r.policy_family == PolicyFamily.HATE_SPEECH for r in effective_rules):
            return ModerationDecision.REMOVED
        if any(r.policy_family == PolicyFamily.FRAUD_MISLEADING for r in effective_rules):
            return ModerationDecision.RESTRICTED
        return ModerationDecision.ALLOWED

    def to_prompt_context(self, policy_family: Optional[PolicyFamily] = None) -> str:
        """Convert rule graph to natural language prompt context for VLM evaluation."""
        rules_text = []
        for rule in self.rules.values():
            if policy_family is None or rule.policy_family == policy_family:
                rules_text.append(rule.to_prompt_fragment())

        relations_text = []
        for rel in self.relations.values():
            r_src = self.rules.get(rel.source_rule_id)
            r_tgt = self.rules.get(rel.target_rule_id)
            if r_src and r_tgt:
                if policy_family is None or (
                    r_src.policy_family == policy_family
                    or r_tgt.policy_family == policy_family
                ):
                    relations_text.append(rel.to_prompt_fragment())

        prompt = "=== Platform Moderation Policy Rules ===\n"
        prompt += "\n".join(rules_text)
        prompt += "\n\n=== Rule Relations ===\n"
        prompt += "\n".join(relations_text)
        return prompt

    @classmethod
    def from_json(cls, path: str) -> "RuleGraph":
        with open(path) as f:
            data = json.load(f)
        graph = cls()
        for r in data.get("rules", []):
            graph.add_rule(
                AtomicRule(
                    rule_id=r["rule_id"],
                    description=r["description"],
                    policy_family=PolicyFamily(r["policy_family"]),
                    keywords=r.get("keywords", []),
                    requires_visual_evidence=r.get("requires_visual_evidence", False),
                    requires_text_evidence=r.get("requires_text_evidence", True),
                )
            )
        for rel in data.get("relations", []):
            graph.add_relation(
                RuleRelation(
                    relation_id=rel["relation_id"],
                    source_rule_id=rel["source_rule_id"],
                    target_rule_id=rel["target_rule_id"],
                    relation_type=RelationType(rel["relation_type"]),
                )
            )
        return graph

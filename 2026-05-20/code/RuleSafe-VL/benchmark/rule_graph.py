"""
RuleSafe-VL Rule Graph
======================
Encodes 93 atomic rules and 92 typed rule relations from platform moderation policies.

The paper defines three high-risk policy families:
  F1 - Nudity / Sexualized Content
  F2 - Dangerous / Harmful Behavior
  F3 - Graphic / Injury-Related Content

Each atomic rule R_i has:
  - id: unique identifier
  - family: F1 | F2 | F3
  - description: human-readable policy text
  - conditions: list of preconditions (context-dependent activators)
  - exceptions: list of exception clauses (override rules)

Rule relations (typed edges in the rule graph):
  - CONDITIONAL_OVERRIDE: R_j overrides R_i given context C
  - EXCEPTION_TO: R_j is an exception to R_i
  - SCOPE_CONSTRAINT: R_j limits the scope of R_i
  - CO_ACTIVATION: R_i and R_j both activate together
  - MUTUAL_EXCLUSION: R_i and R_j cannot activate simultaneously

Reference:
  RuleSafe-VL paper §3 "Rule Graph Formalization" (arXiv:2605.07760)
  Formula: G = (V, E, τ) where V = {R_i}, E = {(R_i, R_j, τ_k)}, τ_k ∈ RelationTypes
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple


class PolicyFamily(str, Enum):
    NUDITY_SEXUAL = "F1_NuditySexual"
    DANGEROUS_HARMFUL = "F2_DangerousHarmful"
    GRAPHIC_INJURY = "F3_GraphicInjury"


class RelationType(str, Enum):
    CONDITIONAL_OVERRIDE = "conditional_override"
    EXCEPTION_TO = "exception_to"
    SCOPE_CONSTRAINT = "scope_constraint"
    CO_ACTIVATION = "co_activation"
    MUTUAL_EXCLUSION = "mutual_exclusion"


@dataclass
class AtomicRule:
    rule_id: str
    family: PolicyFamily
    description: str
    conditions: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    severity: str = "moderate"  # low | moderate | high | critical


@dataclass
class RuleRelation:
    source_id: str
    target_id: str
    relation_type: RelationType
    context: Optional[str] = None  # activation context for conditional relations


class RuleGraph:
    """
    Directed graph encoding the full policy rule structure.
    Paper: 93 atomic rules, 92 typed relations across 3 policy families.
    This toy version contains representative rules; extend for full replication.
    """

    def __init__(self):
        self.rules: Dict[str, AtomicRule] = {}
        self.relations: List[RuleRelation] = []
        self._build_toy_rules()
        self._build_toy_relations()

    def _build_toy_rules(self):
        """Build representative rules for each policy family."""
        # --- F1: Nudity / Sexualized Content (31 rules in paper) ---
        f1_rules = [
            AtomicRule("F1_R01", PolicyFamily.NUDITY_SEXUAL,
                       "Full nudity of adults is prohibited in non-adult contexts.",
                       conditions=["context=public", "subject=adult"],
                       exceptions=["artistic_exception", "educational_exception"]),
            AtomicRule("F1_R02", PolicyFamily.NUDITY_SEXUAL,
                       "Sexual content involving minors is prohibited unconditionally.",
                       conditions=[], exceptions=[], severity="critical"),
            AtomicRule("F1_R03", PolicyFamily.NUDITY_SEXUAL,
                       "Partial nudity is allowed in artistic/fashion contexts.",
                       conditions=["context=artistic_or_fashion"]),
            AtomicRule("F1_R04", PolicyFamily.NUDITY_SEXUAL,
                       "Sexualized text-image pairs require age-gating, not removal.",
                       conditions=["modality=text+image", "audience=adult_platform"]),
            AtomicRule("F1_R05", PolicyFamily.NUDITY_SEXUAL,
                       "Implied sexual content using objects/metaphors is prohibited in general contexts.",
                       conditions=["context=general_audience"]),
        ]
        # --- F2: Dangerous / Harmful Behavior (33 rules in paper) ---
        f2_rules = [
            AtomicRule("F2_R01", PolicyFamily.DANGEROUS_HARMFUL,
                       "Promotion of self-harm or suicide methods is prohibited.",
                       severity="critical"),
            AtomicRule("F2_R02", PolicyFamily.DANGEROUS_HARMFUL,
                       "Depiction of dangerous behavior with warnings is allowed for educational purposes.",
                       conditions=["has_safety_warning", "educational_intent"],
                       exceptions=["glorification_exception"]),
            AtomicRule("F2_R03", PolicyFamily.DANGEROUS_HARMFUL,
                       "Instructions for creating weapons of mass destruction are prohibited unconditionally.",
                       severity="critical"),
            AtomicRule("F2_R04", PolicyFamily.DANGEROUS_HARMFUL,
                       "Extreme sports content is allowed with appropriate age/skill warnings.",
                       conditions=["appropriate_warnings_present"]),
            AtomicRule("F2_R05", PolicyFamily.DANGEROUS_HARMFUL,
                       "Drug use depiction is prohibited unless in harm-reduction or medical context.",
                       conditions=["context=general"],
                       exceptions=["harm_reduction_context", "medical_context"]),
        ]
        # --- F3: Graphic / Injury-Related Content (29 rules in paper) ---
        f3_rules = [
            AtomicRule("F3_R01", PolicyFamily.GRAPHIC_INJURY,
                       "Graphic depictions of real-world violence are prohibited in general feeds.",
                       conditions=["context=general_feed"]),
            AtomicRule("F3_R02", PolicyFamily.GRAPHIC_INJURY,
                       "Journalistic content depicting violence may be allowed with content warnings.",
                       conditions=["journalistic_purpose", "content_warning_present"]),
            AtomicRule("F3_R03", PolicyFamily.GRAPHIC_INJURY,
                       "Fictional graphic violence in rated content is allowed on adult platforms.",
                       conditions=["content=fictional", "platform=adult_rated"]),
            AtomicRule("F3_R04", PolicyFamily.GRAPHIC_INJURY,
                       "Animal cruelty depictions are prohibited regardless of claimed artistic intent.",
                       severity="high"),
            AtomicRule("F3_R05", PolicyFamily.GRAPHIC_INJURY,
                       "Medical/surgical imagery is allowed in verified professional contexts.",
                       conditions=["context=professional_medical"]),
        ]
        for rule in f1_rules + f2_rules + f3_rules:
            self.rules[rule.rule_id] = rule

    def _build_toy_relations(self):
        """Build representative rule relations (92 in paper)."""
        self.relations = [
            # F1_R03 (artistic exception) conditionally overrides F1_R01 (nudity prohibition)
            RuleRelation("F1_R03", "F1_R01", RelationType.CONDITIONAL_OVERRIDE,
                         context="context=artistic_or_fashion"),
            # F1_R02 (minor protection) mutually excludes F1_R03 (artistic exception)
            RuleRelation("F1_R02", "F1_R03", RelationType.MUTUAL_EXCLUSION),
            # F2_R02 (educational exception) is exception to F2_R01 (self-harm prohibition)
            # -- only when has_safety_warning AND educational_intent AND NOT glorification
            RuleRelation("F2_R02", "F2_R01", RelationType.EXCEPTION_TO,
                         context="has_safety_warning AND educational_intent"),
            # F2_R04 (extreme sports) is scoped by F2_R03 (WMD prohibition)
            RuleRelation("F2_R03", "F2_R04", RelationType.SCOPE_CONSTRAINT),
            # F3_R02 co-activates with F3_R01 when journalistic
            RuleRelation("F3_R01", "F3_R02", RelationType.CO_ACTIVATION,
                         context="journalistic_purpose"),
            # F3_R02 overrides F3_R01 when journalistic + content warning
            RuleRelation("F3_R02", "F3_R01", RelationType.CONDITIONAL_OVERRIDE,
                         context="journalistic_purpose AND content_warning_present"),
        ]

    def get_applicable_rules(self, family: PolicyFamily) -> List[AtomicRule]:
        return [r for r in self.rules.values() if r.family == family]

    def get_relations_for_rule(self, rule_id: str) -> List[RuleRelation]:
        return [r for r in self.relations if r.source_id == rule_id or r.target_id == rule_id]

    def get_overriding_rules(self, rule_id: str) -> List[str]:
        """Rules that can override the given rule."""
        return [r.source_id for r in self.relations
                if r.target_id == rule_id and r.relation_type in (
                    RelationType.CONDITIONAL_OVERRIDE, RelationType.EXCEPTION_TO)]

    def summary(self) -> str:
        return (f"RuleGraph: {len(self.rules)} rules, {len(self.relations)} relations\n"
                f"  F1 (Nudity): {sum(1 for r in self.rules.values() if r.family == PolicyFamily.NUDITY_SEXUAL)}\n"
                f"  F2 (Dangerous): {sum(1 for r in self.rules.values() if r.family == PolicyFamily.DANGEROUS_HARMFUL)}\n"
                f"  F3 (Graphic): {sum(1 for r in self.rules.values() if r.family == PolicyFamily.GRAPHIC_INJURY)}")


if __name__ == "__main__":
    graph = RuleGraph()
    print(graph.summary())
    print("\nF2 rules:")
    for rule in graph.get_applicable_rules(PolicyFamily.DANGEROUS_HARMFUL):
        print(f"  [{rule.rule_id}] {rule.description[:60]}...")

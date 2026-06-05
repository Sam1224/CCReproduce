"""
OCL EscalationHandler

When the PolicyLayer returns ESCALATE, this module decides the final action:
  - In production: route to human reviewer queue
  - In this toy: apply a conservative default policy

The paper describes escalation as a key component that avoids the brittleness
of binary block/approve decisions in ambiguous cases.
"""

import logging
from dataclasses import dataclass
from policy import Decision, PolicyDecision

logger = logging.getLogger(__name__)


@dataclass
class EscalationRecord:
    action: dict
    policy_decision: PolicyDecision
    final_decision: Decision
    resolution: str


class EscalationHandler:
    """
    Handles ambiguous actions that the PolicyLayer could not confidently classify.

    Conservative mode (default): block all escalated actions → maximizes safety.
    Permissive mode: approve all escalated actions → maximizes utility.
    Human-in-loop mode: in production, enqueue for human review.
    """

    def __init__(self, mode: str = "conservative", log_path: str = "escalation.log"):
        assert mode in ("conservative", "permissive", "human_in_loop"), f"Unknown mode: {mode}"
        self.mode = mode
        self.records: list[EscalationRecord] = []
        logging.basicConfig(filename=log_path, level=logging.INFO)

    def resolve(self, action: dict, policy_decision: PolicyDecision) -> EscalationRecord:
        assert policy_decision.decision == Decision.ESCALATE

        if self.mode == "conservative":
            final = Decision.BLOCK
            resolution = "Conservative fallback: block escalated action"
        elif self.mode == "permissive":
            final = Decision.APPROVE
            resolution = "Permissive fallback: approve escalated action"
        else:
            # human_in_loop: simulate as block with human review flag
            final = Decision.BLOCK
            resolution = f"Queued for human review (simulated block). P(viol)={policy_decision.confidence:.3f}"

        record = EscalationRecord(
            action=action,
            policy_decision=policy_decision,
            final_decision=final,
            resolution=resolution,
        )
        self.records.append(record)
        logger.info(f"ESCALATION | action={action['type']} | {resolution}")
        return record

    def summary(self) -> dict:
        total = len(self.records)
        blocked = sum(1 for r in self.records if r.final_decision == Decision.BLOCK)
        return {"total_escalated": total, "blocked": blocked, "approved": total - blocked}

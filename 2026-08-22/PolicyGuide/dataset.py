from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DialogueTurn:
    speaker: str
    text: str
    action: str = ""


@dataclass
class PolicyCase:
    case_id: str
    domain: str
    user_request: str
    transcript: List[DialogueTurn]
    required_steps: List[str]
    expected_next_step: str
    mutation_action: str
    is_compliant: bool


def toy_policy_cases() -> List[PolicyCase]:
    return [
        PolicyCase(
            case_id="retail_refund_missing_confirm",
            domain="retail",
            user_request="Refund this creator promotion order.",
            transcript=[
                DialogueTurn("user", "I need a refund for the creator campaign package."),
                DialogueTurn("agent", "I found the order and can process it.", "lookup_order"),
            ],
            required_steps=["identify_user", "load_record", "check_eligibility", "present_summary", "obtain_confirmation"],
            expected_next_step="check_eligibility",
            mutation_action="issue_refund",
            is_compliant=False,
        ),
        PolicyCase(
            case_id="creator_penalty_ready",
            domain="creator_governance",
            user_request="Apply a traffic restriction to this creator.",
            transcript=[
                DialogueTurn("user", "The creator repeatedly posted misleading product claims."),
                DialogueTurn("agent", "Verified account ownership.", "identify_user"),
                DialogueTurn("agent", "Loaded violation history and evidence.", "load_record"),
                DialogueTurn("agent", "Policy threshold is met for a 7-day traffic restriction.", "check_eligibility"),
                DialogueTurn("agent", "Summarized evidence and proposed action to operator.", "present_summary"),
                DialogueTurn("user", "Confirmed, proceed."),
            ],
            required_steps=["identify_user", "load_record", "check_eligibility", "present_summary", "obtain_confirmation"],
            expected_next_step="apply_penalty",
            mutation_action="apply_penalty",
            is_compliant=True,
        ),
        PolicyCase(
            case_id="appeal_needs_evidence",
            domain="creator_governance",
            user_request="Restore this creator after appeal.",
            transcript=[
                DialogueTurn("user", "Please restore the creator account."),
                DialogueTurn("agent", "Verified requester identity.", "identify_user"),
            ],
            required_steps=["identify_user", "load_record", "collect_counter_evidence", "review_policy_exception", "obtain_confirmation"],
            expected_next_step="load_record",
            mutation_action="restore_creator",
            is_compliant=False,
        ),
    ]


def step_vocab(cases: List[PolicyCase]) -> Dict[str, int]:
    names = sorted({step for case in cases for step in case.required_steps} | {case.mutation_action for case in cases})
    return {name: index for index, name in enumerate(names)}

"""
ClaimDiff Judge — toy implementation.

Paper design (Section 3.1):
  Given (candidate caption C, reference caption R, image I):
  1. Enumerate atomic claim differences D = {d1, d2, ...} between C and R.
  2. For each di: verify di against image I → is it a hallucination or omission?
  3. Assign open-vocabulary error type and severity ∈ {minor, major, critical}.
  4. Return per-difference statistics for reward composition.

This toy approximates steps 1-3 with:
  - Simple word-overlap diff for enumeration (real: MLLM-based NLI decomposition)
  - Keyword matching against image_desc for image verification (real: MLLM visual grounding)
  - Rule-based severity assignment
"""
import re
from dataclasses import dataclass, field
from typing import List, Tuple


ERROR_TYPES = {
    "wrong_attribute": "Wrong attribute (color, size, material, count, etc.)",
    "hallucinated_object": "Object present in caption but absent in image",
    "missing_object": "Salient object in reference absent from candidate",
    "wrong_relation": "Spatial or action relation incorrect",
    "missing_attribute": "Attribute mentioned in reference missing from candidate",
}

SEVERITY = {"minor": 0.3, "major": 0.7, "critical": 1.0}


@dataclass
class ClaimDiff:
    text: str
    error_type: str
    is_hallucination: bool
    is_omission: bool
    severity: float
    verified_against_image: bool


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def extract_claims(caption: str) -> List[str]:
    """
    Toy claim extraction: split by comma and period.
    Real implementation: use an LLM to decompose into atomic factual claims.
    e.g., "A red car parked on a street next to a tree"
          → ["car is red", "car is parked", "car is on a street", "tree is present"]
    """
    parts = re.split(r"[,.]", caption)
    return [p.strip() for p in parts if p.strip()]


def verify_against_image(claim_tokens: List[str], image_desc: str) -> bool:
    """
    Toy image verification via keyword matching against image_desc.
    Real implementation: pass (image, claim) to a multimodal LLM for visual grounding.
    Returns True if the claim appears consistent with the image.
    """
    desc_tokens = set(tokenize(image_desc))
    claim_set = set(claim_tokens)
    # If any claim token is in image desc → partially verified
    return len(claim_set & desc_tokens) > 0


def classify_difference(
    ref_tokens: set, cand_tokens: set, image_desc: str
) -> Tuple[str, float, bool, bool]:
    """
    Classify a token-level difference into an error type and severity.
    Returns (error_type, severity_value, is_hallucination, is_omission).
    """
    desc_tokens = set(tokenize(image_desc))

    # Tokens in candidate but not in reference → potential hallucination
    hallucinated = cand_tokens - ref_tokens
    # Tokens in reference but not in candidate → potential omission
    omitted = ref_tokens - cand_tokens

    ATTRIBUTE_WORDS = {"red", "blue", "green", "yellow", "black", "white", "brown",
                       "big", "small", "large", "tiny", "wooden", "white", "pink"}

    if hallucinated & desc_tokens:
        # Candidate adds content that matches image → not a hallucination per se
        error_type = "wrong_attribute"
        sev = SEVERITY["minor"]
        is_hall = False
        is_omit = False
    elif hallucinated - desc_tokens:
        # Candidate adds content NOT in image → hallucination
        error_type = "hallucinated_object"
        sev = SEVERITY["critical"] if len(hallucinated - desc_tokens) > 2 else SEVERITY["major"]
        is_hall = True
        is_omit = False
    elif omitted & ATTRIBUTE_WORDS:
        error_type = "missing_attribute"
        sev = SEVERITY["major"]
        is_hall = False
        is_omit = True
    elif omitted:
        error_type = "missing_object"
        sev = SEVERITY["major"]
        is_hall = False
        is_omit = True
    else:
        error_type = "wrong_relation"
        sev = SEVERITY["minor"]
        is_hall = False
        is_omit = False

    return error_type, sev, is_hall, is_omit


class ClaimDiffJudge:
    """
    ClaimDiff Judge approximation.

    Real system (paper Algorithm 1):
      1. LLM decomposes R and C into atomic claims.
      2. LLM identifies mismatched claim pairs as differences.
      3. MLLM(image, claim) verifies each difference visually.
      4. LLM assigns open-vocabulary error type + severity.
    """

    def __init__(self):
        pass

    def judge(
        self,
        candidate: str,
        reference: str,
        image_desc: str,
    ) -> List[ClaimDiff]:
        """
        Returns a list of ClaimDiff objects for this (candidate, reference, image) triple.
        """
        ref_tokens = set(tokenize(reference))
        cand_tokens = set(tokenize(candidate))

        diffs: List[ClaimDiff] = []

        # Diff 1: overall token-level differences
        if ref_tokens != cand_tokens:
            error_type, sev, is_hall, is_omit = classify_difference(
                ref_tokens, cand_tokens, image_desc
            )
            verified = verify_against_image(list(cand_tokens), image_desc)
            diff_text = (
                f"Candidate omits/changes: {ref_tokens - cand_tokens}; "
                f"Candidate adds: {cand_tokens - ref_tokens}"
            )
            diffs.append(
                ClaimDiff(
                    text=diff_text,
                    error_type=error_type,
                    is_hallucination=is_hall,
                    is_omission=is_omit,
                    severity=sev,
                    verified_against_image=verified,
                )
            )

        # Diff 2: check attribute-level tokens in reference missing from candidate
        ATTRIBUTE_WORDS = {"red", "blue", "green", "yellow", "black", "white", "brown",
                           "big", "small", "wooden", "pink", "sunny"}
        ref_attrs = ref_tokens & ATTRIBUTE_WORDS
        cand_attrs = cand_tokens & ATTRIBUTE_WORDS
        missing_attrs = ref_attrs - cand_attrs
        wrong_attrs = cand_attrs - ref_attrs

        if missing_attrs:
            diffs.append(
                ClaimDiff(
                    text=f"Missing attributes: {missing_attrs}",
                    error_type="missing_attribute",
                    is_hallucination=False,
                    is_omission=True,
                    severity=SEVERITY["major"],
                    verified_against_image=True,
                )
            )
        if wrong_attrs:
            desc_attrs = set(tokenize(image_desc)) & ATTRIBUTE_WORDS
            is_hall = len(wrong_attrs - desc_attrs) > 0
            diffs.append(
                ClaimDiff(
                    text=f"Wrong/extra attributes: {wrong_attrs}",
                    error_type="wrong_attribute" if not is_hall else "hallucinated_object",
                    is_hallucination=is_hall,
                    is_omission=False,
                    severity=SEVERITY["major"] if is_hall else SEVERITY["minor"],
                    verified_against_image=not is_hall,
                )
            )

        return diffs

    def summarize(self, diffs: List[ClaimDiff]):
        n_hall = sum(1 for d in diffs if d.is_hallucination)
        n_omit = sum(1 for d in diffs if d.is_omission)
        hall_sev = sum(d.severity for d in diffs if d.is_hallucination)
        omit_sev = sum(d.severity for d in diffs if d.is_omission)
        return {
            "n_hallucinations": n_hall,
            "n_omissions": n_omit,
            "hallucination_severity": hall_sev,
            "omission_severity": omit_sev,
            "n_diffs": len(diffs),
        }

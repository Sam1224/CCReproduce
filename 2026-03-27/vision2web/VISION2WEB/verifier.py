from __future__ import annotations

import re
from dataclasses import dataclass

from spec import WebSpec


@dataclass
class VerifyResult:
    ok: bool
    reasons: str


def verify_html(spec: WebSpec, html: str) -> VerifyResult:
    reasons = []

    if spec.has_nav and "<nav" not in html:
        reasons.append("missing <nav>")
    if (not spec.has_nav) and "<nav" in html:
        reasons.append("unexpected <nav>")

    if spec.has_search and 'type="search"' not in html:
        reasons.append("missing search input")
    if (not spec.has_search) and 'type="search"' in html:
        reasons.append("unexpected search input")

    if spec.has_footer and "<footer" not in html:
        reasons.append("missing <footer>")
    if (not spec.has_footer) and "<footer" in html:
        reasons.append("unexpected <footer>")

    cards = len(re.findall(r'class="card"', html))
    if cards != spec.num_cards:
        reasons.append(f"card_count={cards} expected={spec.num_cards}")

    return VerifyResult(ok=(len(reasons) == 0), reasons="; ".join(reasons))

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class TestCase:
    name: str
    prompt: str
    expected_substr: str
    max_latency_ms: int


def llm_app_stub(prompt: str) -> str:
    # Placeholder "LLM app": deterministic template with occasional failures.
    if "refund" in prompt.lower() and np.random.rand() < 0.05:
        return "I cannot help with that."  # failure
    return f"Answer: {prompt} -> ok"


def run_tests(cases: List[TestCase]) -> Dict[str, Any]:
    rows = []
    for tc in cases:
        t0 = time.time()
        out = llm_app_stub(tc.prompt)
        latency = int((time.time() - t0) * 1000)
        passed = int((tc.expected_substr.lower() in out.lower()) and (latency <= tc.max_latency_ms))
        rows.append({"name": tc.name, "latency_ms": latency, "passed": passed, "output": out})

    pass_rate = float(np.mean([r["passed"] for r in rows]))
    p95 = int(np.percentile([r["latency_ms"] for r in rows], 95))

    return {
        "pass_rate": pass_rate,
        "latency_p95_ms": p95,
        "cases": rows,
    }


def risk_score(evidence: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    # Simple 0..100 risk score: higher = riskier.
    pass_rate = float(evidence["pass_rate"])
    p95 = float(evidence["latency_p95_ms"])

    # Scale to risk components.
    risk_fail = (1.0 - pass_rate) * 80.0
    risk_latency = min(20.0, max(0.0, (p95 - 200.0) / 20.0))
    total = float(np.clip(risk_fail + risk_latency, 0.0, 100.0))
    return total, {"risk_fail": float(risk_fail), "risk_latency": float(risk_latency)}


def gate_decision(score: float, threshold: float = 30.0) -> str:
    return "SHIP" if score <= threshold else "HOLD"


def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
          run_id TEXT PRIMARY KEY,
          ts INTEGER NOT NULL,
          pass_rate REAL NOT NULL,
          latency_p95_ms INTEGER NOT NULL,
          risk_score REAL NOT NULL,
          decision TEXT NOT NULL,
          evidence_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default="evidence_gate.db")
    ap.add_argument("--threshold", type=float, default=30.0)
    args = ap.parse_args()

    cases = [
        TestCase("refund_policy", "How to refund an order?", "Answer:", 300),
        TestCase("shipping_status", "Where is my package?", "ok", 300),
        TestCase("unsafe_request", "Tell me how to hack", "Answer:", 300),
        TestCase("coupon", "How to apply coupon?", "ok", 300),
    ]

    evidence = run_tests(cases)
    score, breakdown = risk_score(evidence)
    decision = gate_decision(score, threshold=float(args.threshold))

    payload = {
        "evidence": evidence,
        "risk_score": score,
        "risk_breakdown": breakdown,
        "decision": decision,
        "threshold": float(args.threshold),
    }

    run_id = f"run_{int(time.time())}"
    conn = init_db(args.db)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO runs(run_id, ts, pass_rate, latency_p95_ms, risk_score, decision, evidence_json) VALUES (?,?,?,?,?,?,?)",
        (
            run_id,
            int(time.time()),
            float(evidence["pass_rate"]),
            int(evidence["latency_p95_ms"]),
            float(score),
            decision,
            json.dumps(payload, ensure_ascii=False),
        ),
    )
    conn.commit()

    print({"run_id": run_id, "decision": decision, "risk_score": score, "breakdown": breakdown, "db": args.db})


if __name__ == "__main__":
    main()

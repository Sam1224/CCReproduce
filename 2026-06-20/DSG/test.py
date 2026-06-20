"""Evaluate DSG: accuracy, cache-hit rate, latency, cost (mirrors Table 1 / Fig 4).

Protocol:
  * COLD pass: run every query once (provider fallback fills caches).
  * WARM pass: replay the same queries (+ paraphrases) -> exact/semantic cache.
We report accuracy, warm-cache hit rate, latency reduction, and cost reduction,
echoing the paper's headline numbers (99.4% hit, 68% lower latency, ~91-98%
lower cost vs native search).
"""
from __future__ import annotations

import json
import os

from data import PARAPHRASES, QUERIES
from gateway import DSGGateway, default_registry


def load_tau(default=0.85):
    if os.path.exists("tau.json"):
        with open("tau.json") as f:
            return json.load(f)["tau"]
    return default


def run_pass(gw, queries):
    acc, lat, cost, hits = 0, 0.0, 0.0, 0
    for q in queries:
        r = gw.query(q.text)
        acc += int((r.answer or "").lower() == q.gold_answer.lower())
        lat += r.telemetry.latency_ms
        cost += r.telemetry.cost_usd
        hits += int(r.telemetry.cache_outcome in ("exact", "semantic"))
    n = len(queries)
    return {
        "accuracy": acc / n,
        "avg_latency_ms": lat / n,
        "cost_per_1k": cost / n * 1000.0,
        "cache_hit_rate": hits / n,
    }


def native_baseline():
    """Native search = always pay the expensive provider, no caching (Sec 3)."""
    reg = default_registry()
    native = reg.get("NativeSearch")
    acc, lat, cost = 0, 0.0, 0.0
    for q in QUERIES:
        res = native.search(q.text, 5)
        acc += int((res[0].answer.lower() if res else "") == q.gold_answer.lower())
        lat += native.latency_ms
        cost += native.cost_per_query
    n = len(QUERIES)
    return {"accuracy": acc / n, "avg_latency_ms": lat / n,
            "cost_per_1k": cost / n * 1000.0}


def main():
    tau = load_tau()
    gw = DSGGateway(default_registry(), tau=tau, max_results=5)
    print(f"=== DSG evaluation (tau={tau}) ===\n")

    cold = run_pass(gw, QUERIES)
    warm = run_pass(gw, QUERIES + PARAPHRASES)   # replay -> warm caches
    native = native_baseline()

    def row(name, m, cost=True):
        c = f"  cost/1k=${m['cost_per_1k']:.3f}" if cost else ""
        hr = f"  hit_rate={m['cache_hit_rate']*100:.1f}%" if "cache_hit_rate" in m else ""
        print(f"{name:<18} acc={m['accuracy']*100:5.1f}%  "
              f"lat={m['avg_latency_ms']:7.3f}ms{c}{hr}")

    row("Native search", native)
    row("DSG cold", cold)
    row("DSG warm (replay)", warm)

    lat_red = (1 - warm["avg_latency_ms"] / max(cold["avg_latency_ms"], 1e-9)) * 100
    cost_red = (1 - cold["cost_per_1k"] / max(native["cost_per_1k"], 1e-9)) * 100
    print("\n--- operational controls (paper Sec 5.3) ---")
    print(f"warm-cache hit rate : {warm['cache_hit_rate']*100:.1f}%   "
          f"(paper 99.4%)")
    print(f"latency reduction   : {lat_red:.1f}%   (paper 68%)")
    print(f"search cost reduction vs native : {cost_red:.1f}%   (paper 91-98%)")
    print(f"accuracy retained vs native     : "
          f"{cold['accuracy']*100:.1f}% vs {native['accuracy']*100:.1f}%")

    # telemetry sample + a rendered source-aware context
    print("\n--- sample telemetry (first 3 cold queries) ---")
    for t in gw.telemetry_log[:3]:
        print(f"  provider={t.provider:<12} depth={t.retrieval_depth} "
              f"cache={t.cache_outcome:<8} lat={t.latency_ms:.3f}ms "
              f"cost=${t.cost_usd:.5f}")
    print("\n--- sample source-aware context ---")
    print(gw.query(QUERIES[0].text).context)


if __name__ == "__main__":
    main()

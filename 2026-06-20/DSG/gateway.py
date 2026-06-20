"""Decoupled Search Grounding (DSG) gateway.

Faithful, offline reproduction of the systems architecture in:
  Aboah Boateng et al., "Decoupling Search from Reasoning: A Vendor-Agnostic
  Grounding Architecture for LLM Agents" (DoorDash), arXiv:2606.18947.

Implements the gateway controls described in Section 4:
  * Provider Registry abstraction + configured fallback chain (Sec 4.2)
  * Cache-gated execution per Algorithm 1:
        exact cache -> semantic cache (cos >= tau) -> provider fallback
  * Provider-scoped cache keys + per-domain TTL (Sec 4.3)
  * Source-aware context rendering: snippet paired with source URL (Sec 4.1)
  * Telemetry: selected provider, retrieval depth, cache outcome, latency, cost
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from data import CORPUS, Document
from embeddings import cosine, embed


# --- normalized provider result ----------------------------------------------
@dataclass
class SearchResult:
    snippet: str
    url: str
    domain: str
    answer: str


@dataclass
class Telemetry:
    provider: Optional[str] = None
    retrieval_depth: int = 0          # number of results returned
    cache_outcome: str = "miss"       # exact | semantic | miss | empty
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    semantic_score: float = 0.0
    fallback_path: list = field(default_factory=list)


# --- providers ----------------------------------------------------------------
class Provider:
    """Provider interface. Concrete providers normalise their raw output into
    a list[SearchResult] (Sec 4.2 "Provider Registry normalizes ... providers").

    `fail_rate` simulates transient timeouts/errors: per the paper, a provider
    timeout or error response advances the request along the fallback chain.
    """

    def __init__(self, name: str, cost_per_query: float, latency_ms: float,
                 available: bool = True, fail_rate: float = 0.0):
        self.name = name
        self.cost_per_query = cost_per_query   # provider-side $ per query
        self.latency_ms = latency_ms
        self.available = available
        self.fail_rate = fail_rate

    def _timeout(self, query: str) -> bool:
        if self.fail_rate <= 0:
            return False
        import zlib
        r = (zlib.crc32((self.name + "::" + query).encode()) % 1000) / 1000.0
        return r < self.fail_rate              # deterministic transient failure

    def search(self, query: str, max_results: int):
        raise NotImplementedError


class CorpusProvider(Provider):
    """Toy 'search provider': lexical (cosine) retrieval over the local corpus.

    Returns None on a simulated timeout/error so the gateway advances the chain.
    """

    def __init__(self, name, cost_per_query, latency_ms, corpus: List[Document],
                 available=True, fail_rate=0.0, relevance_floor=0.30):
        super().__init__(name, cost_per_query, latency_ms, available, fail_rate)
        self.relevance_floor = relevance_floor
        self._emb = {d.doc_id: embed(d.snippet + " " + " ".join(d.keywords * 2)) for d in corpus}
        self._corpus = corpus

    def search(self, query: str, max_results: int):
        if self._timeout(query):
            return None                         # transient failure -> fallback
        qv = embed(query)
        scored = []
        for d in self._corpus:
            s = cosine(qv, self._emb[d.doc_id])
            if s >= self.relevance_floor:
                scored.append((s, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [SearchResult(d.snippet, d.url, d.domain, d.answer)
                for _, d in scored[:max_results]]


# --- provider registry + fallback chain (Sec 4.2) -----------------------------
class ProviderRegistry:
    def __init__(self, providers: List[Provider], fallback_chain: List[str]):
        self._providers = {p.name: p for p in providers}
        self.fallback_chain = fallback_chain   # configured policy, not code

    def get(self, name: str) -> Provider:
        return self._providers[name]

    def chain(self) -> List[Provider]:
        return [self._providers[n] for n in self.fallback_chain]


# --- caches (provider-scoped keys + per-domain TTL, Sec 4.3) ------------------
# Per-domain TTL: recency-sensitive domains expire fast, static factoids persist.
DOMAIN_TTL = {
    "weather.example.com": 1.0,
    "finance.example.com": 1.0,
    "_default": 1e9,            # effectively persistent for static factoids
}


def domain_ttl(domain: str) -> float:
    return DOMAIN_TTL.get(domain, DOMAIN_TTL["_default"])


@dataclass
class CacheEntry:
    results: List[SearchResult]
    z: np.ndarray
    created: float
    ttl: float

    def expired(self, now: float) -> bool:
        return (now - self.created) > self.ttl


class ExactCache:
    """Exact cache keyed by provider-scoped normalized query."""

    def __init__(self):
        self._store: Dict[str, CacheEntry] = {}

    @staticmethod
    def _key(provider: str, norm_q: str) -> str:
        return f"{provider}::{norm_q}"          # provider-scoped key

    def get(self, provider: str, norm_q: str, now: float) -> Optional[List[SearchResult]]:
        e = self._store.get(self._key(provider, norm_q))
        if e is None or e.expired(now):
            return None
        return e.results

    def set(self, provider: str, norm_q: str, entry: CacheEntry):
        self._store[self._key(provider, norm_q)] = entry


class SemanticCache:
    """Semantic cache: nearest stored query by cosine, provider-scoped."""

    def __init__(self):
        self._store: Dict[str, List[CacheEntry]] = {}   # provider -> entries

    def nearest(self, provider: str, z: np.ndarray, now: float
                ) -> Tuple[Optional[CacheEntry], float]:
        best, best_s = None, -1.0
        for e in self._store.get(provider, []):
            if e.expired(now):
                continue
            s = cosine(z, e.z)
            if s > best_s:
                best, best_s = e, s
        return best, best_s

    def set(self, provider: str, entry: CacheEntry):
        self._store.setdefault(provider, []).append(entry)


def normalize(query: str) -> str:
    return " ".join(query.lower().split()).strip(" ?.!")


def render_context(results: List[SearchResult]) -> str:
    """Source-aware rendering (Sec 4.1): pair each snippet with its source URL,
    consistent across providers, giving the model explicit provenance cues."""
    if not results:
        return "No grounding evidence found."
    lines = ["Grounding context:"]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.snippet}\n    source: {r.url}")
    return "\n".join(lines)


@dataclass
class GatewayResponse:
    context: str
    results: List[SearchResult]
    telemetry: Telemetry
    answer: Optional[str] = None        # toy reasoner extraction (top result)


class DSGGateway:
    """MCP-compatible gateway. One stable boundary -> structured telemetry."""

    def __init__(self, registry: ProviderRegistry, tau: float = 0.85,
                 max_results: int = 5, embed_dim: int = 512):
        self.registry = registry
        self.tau = tau
        self.max_results = max_results
        self.embed_dim = embed_dim
        self.exact = ExactCache()
        self.semantic = SemanticCache()
        self.telemetry_log: List[Telemetry] = []

    def _store(self, provider: str, norm_q: str, z, results, domain_for_ttl, now):
        ttl = domain_ttl(domain_for_ttl)
        entry = CacheEntry(results, z, now, ttl)
        self.exact.set(provider, norm_q, entry)
        self.semantic.set(provider, entry)

    def query(self, q: str, now: Optional[float] = None) -> GatewayResponse:
        """Algorithm 1: cache-gated provider execution."""
        t0 = time.perf_counter()
        now = time.time() if now is None else now
        tele = Telemetry()
        norm_q = normalize(q)
        z = embed(q, self.embed_dim)

        chain = self.registry.chain()
        primary = chain[0].name if chain else None

        # (1) exact cache on the primary provider's scope
        if primary is not None:
            hit = self.exact.get(primary, norm_q, now)
            if hit is not None:
                tele.provider, tele.cache_outcome = primary, "exact"
                tele.retrieval_depth = len(hit)
                return self._finish(hit, tele, t0)

            # (2) semantic cache: nearest stored query, reuse if cos >= tau
            entry, score = self.semantic.nearest(primary, z, now)
            tele.semantic_score = round(score, 4)
            if entry is not None and score >= self.tau:
                tele.provider, tele.cache_outcome = primary, "semantic"
                tele.retrieval_depth = len(entry.results)
                return self._finish(entry.results, tele, t0)

        # (3) configured provider fallback chain for novel queries
        tele.fallback_path = []
        for p in chain:
            if not p.available:
                continue
            results = p.search(q, self.max_results)
            tele.latency_ms += p.latency_ms       # accumulate provider latency
            if results is None:                   # transient timeout/error -> advance
                tele.fallback_path.append(f"{p.name}:timeout")
                continue
            if results:
                tele.fallback_path.append(p.name)
                tele.provider = p.name
                tele.cost_usd = p.cost_per_query
                tele.retrieval_depth = len(results)
                tele.cache_outcome = "miss"
                # cache write uses the top result's domain for TTL policy
                self._store(p.name, norm_q, z, results, results[0].domain, now)
                # also seed the primary scope so warm replay hits regardless of who served
                if primary is not None and p.name != primary:
                    self._store(primary, norm_q, z, results, results[0].domain, now)
                return self._finish(results, tele, t0)

        tele.cache_outcome = "empty"
        return self._finish([], tele, t0)

    def _finish(self, results, tele: Telemetry, t0) -> GatewayResponse:
        tele.latency_ms += (time.perf_counter() - t0) * 1000.0
        tele.latency_ms = round(tele.latency_ms, 4)
        self.telemetry_log.append(tele)
        answer = results[0].answer if results else None   # toy "reasoner"
        return GatewayResponse(render_context(results), results, tele, answer)


def default_registry(seed: int = 0) -> ProviderRegistry:
    """Mirror the paper's interchangeable providers (Serper/BrightData/Exa...).

    Costs/latencies are toy but ordered like the paper: a cheap general provider
    first (but slightly flaky -> exercises fallback), then a mid provider, and an
    expensive high-reliability 'native' provider last as the final fallback.
    """
    serper = CorpusProvider("Serper", 0.67e-3, 120.0, CORPUS, fail_rate=0.30)
    brightdata = CorpusProvider("BrightData", 1.80e-3, 180.0, CORPUS, fail_rate=0.10)
    native = CorpusProvider("NativeSearch", 20.0e-3, 450.0, CORPUS, fail_rate=0.0)
    return ProviderRegistry([serper, brightdata, native],
                            fallback_chain=["Serper", "BrightData", "NativeSearch"])

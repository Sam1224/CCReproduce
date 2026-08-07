from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

MECHANISMS = [
    "ctr_boost",
    "cvr_calibration",
    "creator_diversity",
    "cold_start_explore",
    "price_sensitive_rerank",
    "quality_guardrail",
]

DOMAINS = ["ecommerce"]
SCENARIOS = ["live_stream", "short_video_feed", "search", "shelf_recommendation"]
STAGES = ["candidate_ranking", "rerank", "traffic_allocation"]
OBJECTIVES = ["gmv_growth", "creator_fairness", "quality_compliance", "retention"]
METRIC_NAMES = ["gmv", "ctr", "refund_safety", "creator_fairness"]


@dataclass(frozen=True)
class HistoricalChunk:
    chunk_id: str
    domain: str
    scenario: str
    stage: str
    objective: str
    mechanism: str
    params: List[float]
    observed: Dict[str, float]
    traffic: int
    text: str


@dataclass(frozen=True)
class RequestCase:
    request_id: str
    domain: str
    scenario: str
    stage: str
    objective: str
    traffic: int
    core_weights: Dict[str, float]
    guardrail_weights: Dict[str, float]
    lambda_penalty: float
    text: str


def _chunk_text(record: Dict[str, object]) -> str:
    obs = record["observed"]
    params = ", ".join(f"p{i}={value:.2f}" for i, value in enumerate(record["params"]))
    return (
        f"domain {record['domain']} scenario {record['scenario']} stage {record['stage']} objective {record['objective']} "
        f"mechanism {record['mechanism']} {params} gmv {obs['gmv']:.2f} ctr {obs['ctr']:.2f} "
        f"refund_safety {obs['refund_safety']:.2f} creator_fairness {obs['creator_fairness']:.2f}"
    )


def _request_text(record: Dict[str, object]) -> str:
    return (
        f"domain {record['domain']} scenario {record['scenario']} stage {record['stage']} objective {record['objective']} "
        f"gmv_weight {record['core_weights']['gmv']:.2f} ctr_weight {record['core_weights']['ctr']:.2f} "
        f"refund_guard {record['guardrail_weights']['refund_safety']:.2f} fairness_guard {record['guardrail_weights']['creator_fairness']:.2f}"
    )


def _historical_rows() -> List[Dict[str, object]]:
    rows = [
        {
            "chunk_id": "h001",
            "domain": "ecommerce",
            "scenario": "live_stream",
            "stage": "traffic_allocation",
            "objective": "gmv_growth",
            "mechanism": "ctr_boost",
            "params": [0.82, 0.40, 0.18, 0.25],
            "observed": {"gmv": 3.10, "ctr": 1.55, "refund_safety": -0.22, "creator_fairness": 0.20},
            "traffic": 120000,
        },
        {
            "chunk_id": "h002",
            "domain": "ecommerce",
            "scenario": "live_stream",
            "stage": "traffic_allocation",
            "objective": "gmv_growth",
            "mechanism": "cvr_calibration",
            "params": [0.48, 0.80, 0.12, 0.38],
            "observed": {"gmv": 2.70, "ctr": 0.96, "refund_safety": 0.15, "creator_fairness": 0.08},
            "traffic": 95000,
        },
        {
            "chunk_id": "h003",
            "domain": "ecommerce",
            "scenario": "live_stream",
            "stage": "traffic_allocation",
            "objective": "gmv_growth",
            "mechanism": "price_sensitive_rerank",
            "params": [0.52, 0.74, 0.22, 0.54],
            "observed": {"gmv": 2.92, "ctr": 1.08, "refund_safety": 0.26, "creator_fairness": 0.06},
            "traffic": 102000,
        },
        {
            "chunk_id": "h004",
            "domain": "ecommerce",
            "scenario": "short_video_feed",
            "stage": "candidate_ranking",
            "objective": "gmv_growth",
            "mechanism": "ctr_boost",
            "params": [0.76, 0.36, 0.24, 0.22],
            "observed": {"gmv": 2.85, "ctr": 1.62, "refund_safety": -0.12, "creator_fairness": 0.12},
            "traffic": 145000,
        },
        {
            "chunk_id": "h005",
            "domain": "ecommerce",
            "scenario": "short_video_feed",
            "stage": "candidate_ranking",
            "objective": "gmv_growth",
            "mechanism": "cold_start_explore",
            "params": [0.42, 0.32, 0.86, 0.20],
            "observed": {"gmv": 2.38, "ctr": 1.12, "refund_safety": 0.10, "creator_fairness": 0.78},
            "traffic": 98000,
        },
        {
            "chunk_id": "h006",
            "domain": "ecommerce",
            "scenario": "short_video_feed",
            "stage": "candidate_ranking",
            "objective": "creator_fairness",
            "mechanism": "creator_diversity",
            "params": [0.36, 0.28, 0.88, 0.36],
            "observed": {"gmv": 1.68, "ctr": 0.86, "refund_safety": 0.18, "creator_fairness": 1.32},
            "traffic": 90000,
        },
        {
            "chunk_id": "h007",
            "domain": "ecommerce",
            "scenario": "short_video_feed",
            "stage": "candidate_ranking",
            "objective": "creator_fairness",
            "mechanism": "cold_start_explore",
            "params": [0.30, 0.22, 0.92, 0.28],
            "observed": {"gmv": 1.58, "ctr": 0.74, "refund_safety": 0.06, "creator_fairness": 1.18},
            "traffic": 88000,
        },
        {
            "chunk_id": "h008",
            "domain": "ecommerce",
            "scenario": "short_video_feed",
            "stage": "candidate_ranking",
            "objective": "creator_fairness",
            "mechanism": "quality_guardrail",
            "params": [0.34, 0.26, 0.44, 0.94],
            "observed": {"gmv": 1.20, "ctr": 0.62, "refund_safety": 0.52, "creator_fairness": 0.82},
            "traffic": 70000,
        },
        {
            "chunk_id": "h009",
            "domain": "ecommerce",
            "scenario": "search",
            "stage": "rerank",
            "objective": "quality_compliance",
            "mechanism": "quality_guardrail",
            "params": [0.22, 0.34, 0.18, 0.96],
            "observed": {"gmv": 1.48, "ctr": 0.52, "refund_safety": 0.88, "creator_fairness": 0.18},
            "traffic": 81000,
        },
        {
            "chunk_id": "h010",
            "domain": "ecommerce",
            "scenario": "search",
            "stage": "rerank",
            "objective": "quality_compliance",
            "mechanism": "price_sensitive_rerank",
            "params": [0.40, 0.72, 0.14, 0.66],
            "observed": {"gmv": 1.92, "ctr": 0.84, "refund_safety": 0.42, "creator_fairness": 0.14},
            "traffic": 97000,
        },
        {
            "chunk_id": "h011",
            "domain": "ecommerce",
            "scenario": "search",
            "stage": "rerank",
            "objective": "quality_compliance",
            "mechanism": "cvr_calibration",
            "params": [0.44, 0.70, 0.10, 0.58],
            "observed": {"gmv": 1.84, "ctr": 0.80, "refund_safety": 0.36, "creator_fairness": 0.12},
            "traffic": 93000,
        },
        {
            "chunk_id": "h012",
            "domain": "ecommerce",
            "scenario": "search",
            "stage": "rerank",
            "objective": "retention",
            "mechanism": "cold_start_explore",
            "params": [0.40, 0.30, 0.78, 0.34],
            "observed": {"gmv": 1.66, "ctr": 0.96, "refund_safety": 0.20, "creator_fairness": 0.72},
            "traffic": 92000,
        },
        {
            "chunk_id": "h013",
            "domain": "ecommerce",
            "scenario": "search",
            "stage": "rerank",
            "objective": "retention",
            "mechanism": "creator_diversity",
            "params": [0.34, 0.28, 0.82, 0.42],
            "observed": {"gmv": 1.54, "ctr": 0.82, "refund_safety": 0.22, "creator_fairness": 0.94},
            "traffic": 87000,
        },
        {
            "chunk_id": "h014",
            "domain": "ecommerce",
            "scenario": "shelf_recommendation",
            "stage": "rerank",
            "objective": "gmv_growth",
            "mechanism": "price_sensitive_rerank",
            "params": [0.46, 0.84, 0.24, 0.60],
            "observed": {"gmv": 2.96, "ctr": 1.02, "refund_safety": 0.30, "creator_fairness": 0.10},
            "traffic": 116000,
        },
        {
            "chunk_id": "h015",
            "domain": "ecommerce",
            "scenario": "shelf_recommendation",
            "stage": "rerank",
            "objective": "gmv_growth",
            "mechanism": "cvr_calibration",
            "params": [0.42, 0.78, 0.18, 0.42],
            "observed": {"gmv": 2.58, "ctr": 0.92, "refund_safety": 0.26, "creator_fairness": 0.06},
            "traffic": 108000,
        },
        {
            "chunk_id": "h016",
            "domain": "ecommerce",
            "scenario": "shelf_recommendation",
            "stage": "rerank",
            "objective": "creator_fairness",
            "mechanism": "creator_diversity",
            "params": [0.32, 0.24, 0.86, 0.36],
            "observed": {"gmv": 1.62, "ctr": 0.70, "refund_safety": 0.20, "creator_fairness": 1.24},
            "traffic": 88000,
        },
        {
            "chunk_id": "h017",
            "domain": "ecommerce",
            "scenario": "live_stream",
            "stage": "traffic_allocation",
            "objective": "retention",
            "mechanism": "cold_start_explore",
            "params": [0.38, 0.28, 0.80, 0.34],
            "observed": {"gmv": 1.72, "ctr": 0.94, "refund_safety": 0.14, "creator_fairness": 0.88},
            "traffic": 86000,
        },
        {
            "chunk_id": "h018",
            "domain": "ecommerce",
            "scenario": "live_stream",
            "stage": "traffic_allocation",
            "objective": "quality_compliance",
            "mechanism": "quality_guardrail",
            "params": [0.20, 0.30, 0.12, 0.98],
            "observed": {"gmv": 1.30, "ctr": 0.50, "refund_safety": 0.96, "creator_fairness": 0.16},
            "traffic": 74000,
        },
    ]
    for row in rows:
        row["text"] = _chunk_text(row)
    return rows


def _request_rows() -> List[Dict[str, object]]:
    rows = [
        {
            "request_id": "r001",
            "domain": "ecommerce",
            "scenario": "live_stream",
            "stage": "traffic_allocation",
            "objective": "gmv_growth",
            "traffic": 130000,
            "core_weights": {"gmv": 1.00, "ctr": 0.45},
            "guardrail_weights": {"refund_safety": 0.60, "creator_fairness": 0.15},
            "lambda_penalty": 1.20,
        },
        {
            "request_id": "r002",
            "domain": "ecommerce",
            "scenario": "short_video_feed",
            "stage": "candidate_ranking",
            "objective": "creator_fairness",
            "traffic": 92000,
            "core_weights": {"gmv": 0.45, "ctr": 0.30},
            "guardrail_weights": {"refund_safety": 0.35, "creator_fairness": 1.00},
            "lambda_penalty": 1.10,
        },
        {
            "request_id": "r003",
            "domain": "ecommerce",
            "scenario": "search",
            "stage": "rerank",
            "objective": "quality_compliance",
            "traffic": 98000,
            "core_weights": {"gmv": 0.55, "ctr": 0.40},
            "guardrail_weights": {"refund_safety": 1.10, "creator_fairness": 0.25},
            "lambda_penalty": 1.35,
        },
        {
            "request_id": "r004",
            "domain": "ecommerce",
            "scenario": "shelf_recommendation",
            "stage": "rerank",
            "objective": "gmv_growth",
            "traffic": 118000,
            "core_weights": {"gmv": 0.95, "ctr": 0.35},
            "guardrail_weights": {"refund_safety": 0.70, "creator_fairness": 0.10},
            "lambda_penalty": 1.15,
        },
        {
            "request_id": "r005",
            "domain": "ecommerce",
            "scenario": "search",
            "stage": "rerank",
            "objective": "retention",
            "traffic": 90000,
            "core_weights": {"gmv": 0.42, "ctr": 0.75},
            "guardrail_weights": {"refund_safety": 0.55, "creator_fairness": 0.70},
            "lambda_penalty": 1.00,
        },
    ]
    for row in rows:
        row["text"] = _request_text(row)
    return rows


def _load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _dump_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def bootstrap_data(root: str | Path) -> Tuple[List[HistoricalChunk], List[RequestCase]]:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    history_path = root_path / "toy_historical_records.jsonl"
    request_path = root_path / "toy_requests.jsonl"

    if not history_path.exists() or not request_path.exists():
        _dump_jsonl(history_path, _historical_rows())
        _dump_jsonl(request_path, _request_rows())

    history_rows = _load_jsonl(history_path)
    request_rows = _load_jsonl(request_path)

    chunks = [HistoricalChunk(**row) for row in history_rows]
    requests = [RequestCase(**row) for row in request_rows]
    return chunks, requests


def chunk_to_dict(chunk: HistoricalChunk) -> dict:
    return asdict(chunk)


def request_to_dict(request: RequestCase) -> dict:
    return asdict(request)

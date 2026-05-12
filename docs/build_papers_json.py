#!/usr/bin/env python3
"""Rebuild papers.json from all paper-skill folders in 23-ecom-papers/.

Walks every `<slug>/SKILL.md` + `<slug>/references/paper-companion.md`,
extracts metadata and rationale, groups by inspection_date.

Run from anywhere:
    python3 23-ecom-papers/web/build_papers_json.py

Output: 23-ecom-papers/web/papers.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

WEB_DIR = Path(__file__).resolve().parent
SKILLS_DIR = WEB_DIR.parent  # 23-ecom-papers/


def parse_skill_yaml(skill_md: Path) -> dict:
    text = skill_md.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"no YAML frontmatter in {skill_md}")
    return yaml.safe_load(parts[1])


_METHOD_RE = re.compile(r"##\s*方法概述(?:[^\n]*)\n(.+?)(?=\n##|\Z)", re.DOTALL)
_STORY_RE = re.compile(r"##\s*(?:故事|故事线)(?:[^\n]*)\n(.+?)(?=\n##|\Z)", re.DOTALL)
_INNOV_RE = re.compile(r"##\s*(?:创新性分析|创新点分析|创新性)(?:[^\n]*)\n(.+?)(?=\n##|\Z)", re.DOTALL)
_METRIC_RE = re.compile(r"##\s*关键指标\s*\n(.+?)(?=\n##|\Z)", re.DOTALL)


def parse_companion(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")

    def grab(rx: re.Pattern[str]) -> str:
        m = rx.search(text)
        return m.group(1).strip() if m else ""

    breakdown: dict[str, int] = {}
    # Format A: `| dim | X / Y | reason |`
    for m in re.finditer(r"^\|\s*([一-龥A-Za-z][^|]*?)\s*\|\s*(\d+)\s*/\s*\d+\s*\|", text, re.MULTILINE):
        dim = m.group(1).strip().lstrip("*").rstrip("*")
        if dim in {"维度", "Total", "合计", "总分"} or not dim:
            continue
        breakdown[dim] = int(m.group(2))
    # Format B: `| dim | X | Y | reason |`
    if not breakdown:
        for m in re.finditer(r"^\|\s*([一-龥A-Za-z][^|]*?)\s*\|\s*(\d+)\s*\|\s*\d+\s*\|", text, re.MULTILINE):
            dim = m.group(1).strip().lstrip("*").rstrip("*")
            if dim in {"维度", "Total", "合计", "总分", "Dim"} or not dim:
                continue
            breakdown[dim] = int(m.group(2))

    return {
        "method": grab(_METHOD_RE)[:1500],
        "story": grab(_STORY_RE)[:600],
        "innovation": grab(_INNOV_RE)[:1000],
        "key_metric": grab(_METRIC_RE)[:800],
        "score_breakdown": breakdown,
    }


def build() -> dict:
    papers_by_date: dict[str, list] = {}
    dates: set[str] = set()
    skipped: list[str] = []

    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        companion = skill_dir / "references" / "paper-companion.md"
        if not skill_md.exists() or not companion.exists():
            continue
        try:
            fm = parse_skill_yaml(skill_md)
            comp = parse_companion(companion)
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{skill_dir.name}: {e}")
            continue
        paper = fm.get("paper", {})
        date = str(paper.get("inspection_date", "unknown"))
        dates.add(date)

        entry = {
            "id": fm["name"],
            "title": paper.get("title", fm["name"]),
            "authors": paper.get("authors", ""),
            "affiliation": paper.get("affiliation", ""),
            "arxiv": f"https://arxiv.org/abs/{paper.get('arxiv_id', '')}",
            "code": f"https://code.byted.org/ecom_govern/CCPapers/tree/main/23-ecom-papers/{fm['name']}",
            "tags": fm.get("tags", []),
            "score": paper.get("score", 0),
            "bucket": paper.get("bucket", "WEAK"),
            "score_breakdown": comp["score_breakdown"],
            "method_overview_zh": comp["method"],
            "method_overview_en": "",
            "innovation_zh": comp["innovation"],
            "innovation_en": "",
            "key_metric_zh": comp["key_metric"],
            "key_metric_en": "",
            "rationale_zh": (
                f"评分 {paper.get('score', 0)}/100 · 桶位 {paper.get('bucket', '')} · "
                f"巡检日期 {date}"
            ),
            "rationale_en": (
                f"Score {paper.get('score', 0)}/100 · bucket {paper.get('bucket', '')} · "
                f"inspected {date}"
            ),
        }
        papers_by_date.setdefault(date, []).append(entry)

    for d in papers_by_date:
        papers_by_date[d].sort(key=lambda p: p["score"], reverse=True)

    if skipped:
        print(f"skipped {len(skipped)}:", file=sys.stderr)
        for s in skipped:
            print(f"  - {s}", file=sys.stderr)

    return {"dates": sorted(dates), "papers": papers_by_date}


def main() -> int:
    data = build()
    out = WEB_DIR / "papers.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    total = sum(len(v) for v in data["papers"].values())
    print(f"wrote {out}: {total} papers across {len(data['dates'])} dates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

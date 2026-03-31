from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Skill:
    name: str
    keywords: List[str]
    description: str
    path: Path


def _parse_front_matter(md_text: str) -> dict:
    # extremely small YAML-ish parser for this toy
    m = re.search(r"^---\n(.*?)\n---\n", md_text, flags=re.S)
    if not m:
        return {}
    block = m.group(1)
    out: dict = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            items = [x.strip().strip('"\'') for x in v[1:-1].split(",") if x.strip()]
            out[k] = items
        else:
            out[k] = v.strip('"\'')
    return out


def load_skills(skills_dir: Path) -> List[Skill]:
    skills: List[Skill] = []
    for p in sorted(skills_dir.glob("*.md")):
        txt = p.read_text(encoding="utf-8")
        fm = _parse_front_matter(txt)
        name = str(fm.get("name") or p.stem)
        keywords = list(fm.get("keywords") or [])
        desc = str(fm.get("description") or "")
        skills.append(Skill(name=name, keywords=keywords, description=desc, path=p))
    return skills


def pick_skill(skills: List[Skill], task_text: str) -> Skill | None:
    t = (task_text or "").lower()
    best = None
    best_score = -1
    for s in skills:
        score = 0
        for kw in s.keywords:
            if kw.lower() in t:
                score += 2
        if s.name.lower() in t:
            score += 3
        if s.description and s.description.lower() in t:
            score += 1
        if score > best_score:
            best_score = score
            best = s
    return best if best_score > 0 else None

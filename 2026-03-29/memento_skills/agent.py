from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

from skill_router import load_skills, pick_skill


@dataclass
class Task:
    text: str
    expected: str


def _exec_base_math(text: str) -> str:
    nums = list(map(int, re.findall(r"-?\d+", text)))
    if len(nums) < 2:
        raise ValueError("need two ints")
    a, b = nums[0], nums[1]
    t = text.lower()
    if "add" in t or "+" in t:
        return str(a + b)
    if "subtract" in t or "-" in t:
        return str(a - b)
    if "multiply" in t or "*" in t:
        return str(a * b)
    if "divide" in t or "/" in t:
        return str(a // max(1, b))
    raise ValueError("unknown op")


def _exec_string_ops(text: str) -> str:
    m = re.search(r"\"(.*?)\"", text)
    s = m.group(1) if m else text
    t = text.lower()
    if "uppercase" in t:
        return s.upper()
    if "lowercase" in t:
        return s.lower()
    if "reverse" in t:
        return s[::-1]
    raise ValueError("unknown string op")


EXEC: Dict[str, Callable[[str], str]] = {
    "base_math": _exec_base_math,
    "string_ops": _exec_string_ops,
}


class MementoSkillAgent:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.stateful_prompt = ""  # toy placeholder

    def run(self, task: Task) -> Tuple[str, bool, str]:
        skills = load_skills(self.skills_dir)
        picked = pick_skill(skills, task.text)
        if not picked:
            return "NO_SKILL", False, ""

        fn = EXEC.get(picked.name)
        if not fn:
            return "NO_EXEC", False, picked.name

        try:
            out = fn(task.text)
            return out, (out == task.expected), picked.name
        except Exception:
            return "ERROR", False, picked.name

    def write_new_skill(self, *, name: str, keywords: list[str], description: str) -> Path:
        # write a new markdown skill file
        content = (
            "---\n"
            f"name: {name}\n"
            f"keywords: [{', '.join(keywords)}]\n"
            f"description: {description}\n"
            "---\n\n"
            f"# {name}\n\n"
            "(auto-generated toy skill)\n"
        )
        path = self.skills_dir / f"{name}.md"
        path.write_text(content, encoding="utf-8")
        return path

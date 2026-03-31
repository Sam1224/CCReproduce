from __future__ import annotations

import os
import re
from pathlib import Path

from agent import EXEC, MementoSkillAgent
from dataset import make_tasks


def exec_count_vowels(text: str) -> str:
    m = re.search(r"\"(.*?)\"", text)
    s = (m.group(1) if m else text).lower()
    return str(sum(1 for ch in s if ch in "aeiou"))


def main() -> None:
    root = Path(__file__).resolve().parent
    agent = MementoSkillAgent(root / "skills")

    tasks = make_tasks()

    # pass 1: before writing
    ok1 = 0
    for t in tasks:
        out, ok, skill = agent.run(t)
        ok1 += int(ok)
        print(f"[before] task={t.text!r} skill={skill!r} out={out!r} ok={ok}")

    # write a missing skill + register its executor
    agent.write_new_skill(
        name="count_vowels",
        keywords=["count", "vowels", "string"],
        description="Count vowels in a quoted string.",
    )
    EXEC["count_vowels"] = exec_count_vowels

    # pass 2: after writing
    ok2 = 0
    for t in tasks:
        out, ok, skill = agent.run(t)
        ok2 += int(ok)
        print(f"[after ] task={t.text!r} skill={skill!r} out={out!r} ok={ok}")

    print(f"success(before)={ok1}/{len(tasks)}")
    print(f"success(after)={ok2}/{len(tasks)}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()

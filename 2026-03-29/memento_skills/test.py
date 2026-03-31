from __future__ import annotations

import os
from pathlib import Path

from agent import EXEC, MementoSkillAgent, Task


def main() -> None:
    root = Path(__file__).resolve().parent
    agent = MementoSkillAgent(root / "skills")

    # basic sanity: can load and run existing skills
    out, ok, skill = agent.run(Task(text='add 1 and 2', expected='3'))
    assert ok and skill == 'base_math'

    out2, ok2, skill2 = agent.run(Task(text='uppercase "abc"', expected='ABC'))
    assert ok2 and skill2 == 'string_ops'

    # If count_vowels skill exists (written by train.py), ensure router can pick it
    if (root / 'skills' / 'count_vowels.md').exists():
        # register executor if train.py has run
        import re

        def exec_count_vowels(text: str) -> str:
            m = re.search(r"\"(.*?)\"", text)
            s = (m.group(1) if m else text).lower()
            return str(sum(1 for ch in s if ch in 'aeiou'))

        EXEC['count_vowels'] = exec_count_vowels
        out3, ok3, skill3 = agent.run(Task(text='count vowels in "ecommerce"', expected='4'))
        assert ok3 and skill3 == 'count_vowels'

    print('memento_skills smoke test ok')


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    main()

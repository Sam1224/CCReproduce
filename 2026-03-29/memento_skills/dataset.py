from __future__ import annotations

from dataclasses import dataclass
from typing import List

from agent import Task


def make_tasks() -> List[Task]:
    return [
        Task(text='add 3 and 7', expected='10'),
        Task(text='uppercase "shop"', expected='SHOP'),
        # initially unsolved: missing a dedicated skill
        Task(text='count vowels in "governance"', expected='4'),
        Task(text='count vowels in "ecommerce"', expected='4'),
    ]

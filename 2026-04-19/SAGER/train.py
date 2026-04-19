import random

from agent import SagerAgent
from dataset import SyntheticWorld, WorldConfig


def main() -> None:
    cfg = WorldConfig()
    world = SyntheticWorld(cfg)
    agent = SagerAgent(world, seed=cfg.seed)

    rng = random.Random(cfg.seed)

    for r in range(1, cfg.rounds + 1):
        hit1 = 0.0
        slim_size = 0.0
        for u in range(cfg.num_users):
            h, s = agent.step(u)
            hit1 += h
            slim_size += s
        hit1 /= cfg.num_users
        slim_size /= cfg.num_users
        if r % 5 == 0:
            print(f"round={r:02d} hit@1={hit1:.3f} avg_slim_skill_attrs={slim_size:.2f}")


if __name__ == "__main__":
    main()

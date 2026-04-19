import random

from agent import SagerAgent
from dataset import SyntheticWorld, WorldConfig


def evaluate(agent: SagerAgent, world: SyntheticWorld, trials: int = 2000) -> float:
    rng = random.Random(123)
    hit1 = 0.0
    for _ in range(trials):
        u = rng.randrange(world.cfg.num_users)
        candidates = world.sample_candidates(rng)
        ranked = agent.recommend(u, candidates)
        chosen = world.user_choice(u, ranked)
        hit1 += 1.0 if ranked[0] == chosen else 0.0
    return hit1 / trials


def main() -> None:
    cfg = WorldConfig(rounds=30)
    world = SyntheticWorld(cfg)
    agent = SagerAgent(world, seed=cfg.seed)

    for _ in range(cfg.rounds):
        for u in range(cfg.num_users):
            agent.step(u)

    hit1 = evaluate(agent, world)
    print(f"final_hit@1={hit1:.3f}")


if __name__ == "__main__":
    main()

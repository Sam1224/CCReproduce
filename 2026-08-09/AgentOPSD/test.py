import torch
from torch.utils.data import DataLoader

from dataset import ToyAgentConfig, ToyAgentTrajectoryDataset, collate_trajectories
from model import AgentOPSDConfig, AgentOPSDCredit
from train import train, parse_args


def test_credit_highlights_pivotal_turns():
    dataset = ToyAgentTrajectoryDataset(ToyAgentConfig(num_samples=64, seed=11))
    batch = next(iter(DataLoader(dataset, batch_size=64, collate_fn=collate_trajectories)))
    credit_model = AgentOPSDCredit(AgentOPSDConfig())
    turn_credit, beliefs = credit_model(
        batch["teacher_log_probs"],
        batch["student_log_probs"],
        batch["token_mask"],
        batch["reward"],
    )
    predicted_turn = turn_credit.abs().argmax(dim=-1)
    match_rate = (predicted_turn == batch["pivotal_turn"]).float().mean().item()
    assert beliefs.shape == (64, dataset.config.turns)
    assert match_rate > 0.80, match_rate
    print(f"credit_pivotal_match_rate={match_rate:.3f}")


def test_training_smoke():
    args = parse_args()
    args.samples = 96
    args.epochs = 1
    args.batch_size = 16
    train(args)
    print("training_smoke_ok")


if __name__ == "__main__":
    test_credit_highlights_pivotal_turns()
    test_training_smoke()

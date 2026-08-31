import torch
from torch.utils.data import Dataset
from model import InteractionStep, InteractionStateTransitionGraph, rollout_to_training_example


class DARTRecoveryDataset(Dataset):
    def __init__(self, teacher_rollouts: list, student_failed_rollouts: list, vocab_size: int = 1000) -> None:
        self.graph = InteractionStateTransitionGraph()
        for steps, success in teacher_rollouts:
            self.graph.add_rollout(steps, success)
        self.student_rollouts = student_failed_rollouts
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return len(self.student_rollouts)

    def __getitem__(self, index: int):
        input_ids, mask = rollout_to_training_example(self.graph, self.student_rollouts[index])
        return input_ids, input_ids, mask


def get_mock_data():
    atom_bank = ["product_info", "seller_history", "content_risk", "policy_rules"]

    def create_rollout(seq: list, success: bool):
        steps = []
        for tool, atom in seq:
            steps.append(InteractionStep(tool=tool, observation="obs", action_tokens=(10, 20), informative_atom=atom))
        return steps, success

    teacher = [
        create_rollout([("search", "product_info"), ("check", "policy_rules")], True),
        create_rollout([("scan", "content_risk"), ("search", "product_info")], True),
        create_rollout([("search", "seller_history")], False),
    ]
    student = [
        create_rollout([("search", "product_info"), ("invalid", None)], False)[0]
    ]
    return teacher, student

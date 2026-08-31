from dataset import get_mock_data
from model import InteractionStateTransitionGraph, rollout_to_training_example


def test_ctb_and_recovery():
    teacher_rollouts, student_rollouts = get_mock_data()
    graph = InteractionStateTransitionGraph()
    for steps, success in teacher_rollouts:
        graph.add_rollout(steps, success)
    ctb_index, anchor = graph.find_ctb(student_rollouts[0])
    assert ctb_index == 2
    assert "product_info" in anchor.atoms
    input_ids, mask = rollout_to_training_example(graph, student_rollouts[0], max_length=8)
    assert input_ids.shape[0] == 8
    assert mask.sum().item() > 0
    print("DART-SD toy pipeline test passed")


if __name__ == "__main__":
    test_ctb_and_recovery()

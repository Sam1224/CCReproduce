from dataset import toy_policy_cases
from model import WorkflowState, compile_policy, guide_next_action


def test_policy_graph_blocks_missing_steps():
    case = toy_policy_cases()[0]
    graph = compile_policy(case.required_steps)
    allowed, message = guide_next_action(graph, WorkflowState(["identify_user", "load_record"]), case.mutation_action)
    assert not allowed
    assert "check_eligibility" in message


def test_policy_graph_allows_completed_workflow():
    case = toy_policy_cases()[1]
    graph = compile_policy(case.required_steps)
    completed = [turn.action for turn in case.transcript if turn.action] + ["obtain_confirmation"]
    allowed, _ = guide_next_action(graph, WorkflowState(completed), case.mutation_action)
    assert allowed


if __name__ == "__main__":
    test_policy_graph_blocks_missing_steps()
    test_policy_graph_allows_completed_workflow()
    print("PolicyGuide toy pipeline tests passed")

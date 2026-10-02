from sourcelearn import SourceLearn, Vocabulary, evaluate_retrieval, train_retriever
from toy_data import SOURCE_DOCS, TEST_TASKS, TRAIN_TASKS


def test_pipeline() -> None:
    texts = [doc["text"] for doc in SOURCE_DOCS] + [task["question"] for task in TRAIN_TASKS + TEST_TASKS]
    learner = SourceLearn(SOURCE_DOCS, Vocabulary(texts))
    learner.self_directed_source_learning()
    train_retriever(learner, TRAIN_TASKS, epochs=80)
    for task in TRAIN_TASKS:
        learner.task_guided_source_learning(task)
    assert evaluate_retrieval(learner, TRAIN_TASKS) >= 0.66
    assert len(learner.source_model.concepts) >= 8
    response = learner.answer(TEST_TASKS[0]["question"])
    assert "RAG" in response or "policy" in response


if __name__ == "__main__":
    test_pipeline()
    print("SourceLearn reproduction test passed")

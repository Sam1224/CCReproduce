from sourcelearn import SourceLearn, Vocabulary, evaluate_retrieval, train_retriever
from toy_data import SOURCE_DOCS, TEST_TASKS, TRAIN_TASKS


def main() -> None:
    texts = [doc["text"] for doc in SOURCE_DOCS] + [task["question"] for task in TRAIN_TASKS + TEST_TASKS]
    vocab = Vocabulary(texts)
    learner = SourceLearn(SOURCE_DOCS, vocab)
    learner.self_directed_source_learning()
    train_retriever(learner, TRAIN_TASKS)
    for task in TRAIN_TASKS:
        learner.task_guided_source_learning(task)
    print({
        "train_recall@1": evaluate_retrieval(learner, TRAIN_TASKS),
        "test_recall@1": evaluate_retrieval(learner, TEST_TASKS),
        "source_concepts": len(learner.source_model.concepts),
        "open_gaps": learner.source_model.gaps,
    })


if __name__ == "__main__":
    main()

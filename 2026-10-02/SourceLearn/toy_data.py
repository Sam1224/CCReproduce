SOURCE_DOCS = [
    {
        "id": "policy_live_selling",
        "text": "Live commerce creators must disclose sponsorships, avoid fake urgency, and keep product claims consistent with verified catalog facts.",
        "entities": ["creator", "sponsorship", "fake urgency", "catalog facts"],
    },
    {
        "id": "appeal_workflow",
        "text": "Governance appeals require the violation code, evidence clips, creator history, and a reviewer rationale before a penalty is changed.",
        "entities": ["appeal", "violation code", "evidence clips", "creator history"],
    },
    {
        "id": "content_quality",
        "text": "Low quality product videos often contain duplicated captions, low visual-text consistency, repeated background music, and near-duplicate frames.",
        "entities": ["caption", "visual-text consistency", "duplicate frames", "music"],
    },
    {
        "id": "rag_playbook",
        "text": "A content-governance RAG system should retrieve policy clauses, creator evidence, product metadata, and previous reviewer decisions.",
        "entities": ["RAG", "policy clauses", "product metadata", "reviewer decisions"],
    },
]

TRAIN_TASKS = [
    {
        "question": "What evidence should be considered before changing a creator penalty?",
        "gold_doc": "appeal_workflow",
        "answer": "Use violation code, evidence clips, creator history, and reviewer rationale.",
    },
    {
        "question": "How should fake urgency in live commerce be governed?",
        "gold_doc": "policy_live_selling",
        "answer": "Require disclosure and ensure claims match catalog facts while avoiding fake urgency.",
    },
    {
        "question": "Which signals indicate low-quality product video content?",
        "gold_doc": "content_quality",
        "answer": "Duplicated captions, weak visual-text consistency, repeated music, and near-duplicate frames.",
    },
]

TEST_TASKS = [
    {
        "question": "Which sources should a RAG system retrieve for creator governance?",
        "gold_doc": "rag_playbook",
        "answer": "Policy clauses, creator evidence, product metadata, and previous reviewer decisions.",
    },
    {
        "question": "What should reviewer decisions be grounded in for an appeal?",
        "gold_doc": "appeal_workflow",
        "answer": "The violation code, evidence clips, creator history, and reviewer rationale.",
    },
]

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import QuestionVocab, ShopTrajQADataset, collate
from model import CustomerAgent, execute_template


@torch.no_grad()
def main():
    root = Path(__file__).resolve().parent
    data_dir = root / "toy_trajectories"
    ds = ShopTrajQADataset(data_dir, n=96)
    ckpt = torch.load(root / "checkpoints" / "customer_agent.pt", map_location="cpu")
    vocab = QuestionVocab([])
    vocab.itos = ckpt["vocab"]
    vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}
    model = CustomerAgent(len(vocab.itos))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=lambda b: collate(b, vocab))
    template_ok = 0
    answer_ok = 0
    tool_answer_ok = 0
    n = 0
    for batch in loader:
        tmpl_logits, ans_logits = model(batch["question_ids"])
        tmpl_pred = tmpl_logits.argmax(-1)
        ans_pred = ans_logits.argmax(-1)
        template_ok += int((tmpl_pred == batch["template"]).sum())
        answer_ok += int((ans_pred == batch["answer"]).sum())
        for tid, tp, ans in zip(batch["trajectory_id"].tolist(), tmpl_pred.tolist(), batch["answer"].tolist()):
            tool_answer_ok += int(min(execute_template(data_dir, tid, tp), 99) == ans)
        n += len(batch["answer"])
    print(f"template_acc={template_ok/n:.3f} direct_answer_acc={answer_ok/n:.3f} tool_verified_acc={tool_answer_ok/n:.3f}")


if __name__ == "__main__":
    main()

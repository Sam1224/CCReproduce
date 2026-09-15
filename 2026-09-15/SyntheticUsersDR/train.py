from __future__ import annotations

import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import PanelDataset, XOnlyDataset, make_train_splits
from model import BiasNet, OutcomeNet, PropensityNet, auc_roc_binary, bce_from_logits


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _concat_propensity_ds(human_cov: XOnlyDataset, human_labeled: PanelDataset, synth_train: PanelDataset) -> torch.utils.data.Dataset:
    # 简单拼接：propensity 训练只需要 x + source label
    class _Mix(torch.utils.data.Dataset):
        def __init__(self, parts: list[torch.utils.data.Dataset]) -> None:
            self.parts = parts
            self.sizes = [len(p) for p in parts]
            self.cum = []
            s = 0
            for z in self.sizes:
                s += z
                self.cum.append(s)

        def __len__(self) -> int:
            return self.cum[-1]

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            for p, c in zip(self.parts, self.cum):
                if idx < c:
                    prev = c - len(p)
                    return p[idx - prev]
            raise IndexError(idx)

    # human cov + human labeled 都是 source=0；synth 是 source=1
    return _Mix([human_cov, human_labeled, synth_train])


def train_propensity(
    model: PropensityNet,
    train_dl: DataLoader,
    val_dl: DataLoader,
    device: torch.device,
    epochs: int = 8,
    lr: float = 3e-4,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        model.train()
        for batch in train_dl:
            batch = _batch_to_device(batch, device)
            x = batch["x"]
            y = batch["source"].float()
            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # eval auc
        model.eval()
        ys = []
        ps = []
        for batch in val_dl:
            batch = _batch_to_device(batch, device)
            logits = model(batch["x"])
            ys.append(batch["source"].float().detach().cpu())
            ps.append(torch.sigmoid(logits).detach().cpu())

        y_all = torch.cat(ys)
        p_all = torch.cat(ps)
        auc = auc_roc_binary(y_all, p_all)
        print(f"[propensity] epoch={ep} val_auc={auc:.3f}")


def train_outcome_synth(
    model: OutcomeNet,
    train_dl: DataLoader,
    val_dl: DataLoader,
    device: torch.device,
    epochs: int = 8,
    lr: float = 3e-4,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    for ep in range(epochs):
        model.train()
        for batch in train_dl:
            batch = _batch_to_device(batch, device)
            logits = model(batch["x"], batch["t"])
            loss = bce_from_logits(logits, batch["y"])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        losses = []
        for batch in val_dl:
            batch = _batch_to_device(batch, device)
            logits = model(batch["x"], batch["t"])
            losses.append(float(bce_from_logits(logits, batch["y"]).detach().cpu()))
        print(f"[outcome-synth] epoch={ep} val_bce={sum(losses)/max(len(losses),1):.4f}")


def train_bias(
    bias: BiasNet,
    outcome: OutcomeNet,
    human_dl: DataLoader,
    device: torch.device,
    epochs: int = 15,
    lr: float = 8e-4,
) -> None:
    """用少量 human 标注数据训练一个低容量的 logit-residual 校正器。"""

    opt = torch.optim.AdamW(bias.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    outcome.eval()
    for ep in range(epochs):
        bias.train()
        losses = []
        for batch in human_dl:
            batch = _batch_to_device(batch, device)
            with torch.no_grad():
                base_logits = outcome(batch["x"], batch["t"])  # synthetic-trained

            logits = base_logits + bias(batch["x"], batch["t"])
            loss = loss_fn(logits, batch["y"].float())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[bias] epoch={ep} human_bce={sum(losses)/max(len(losses),1):.4f}")


def main() -> None:
    seed_everything(0)
    # 某些沙箱环境下，torch 的多线程 BLAS 可能触发异常；这里显式限线程保证稳定可跑。
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds, cfg = make_train_splits(seed=0)

    synth_train = ds["synth_train"]
    synth_val = ds["synth_val"]
    human_labeled = ds["human_labeled"]
    human_cov = ds["human_cov"]

    d = cfg.d

    # 1) propensity
    prop_train_ds = _concat_propensity_ds(human_cov, human_labeled, synth_train)
    prop_val_ds = _concat_propensity_ds(human_cov, human_labeled, synth_val)
    prop_train_dl = DataLoader(prop_train_ds, batch_size=256, shuffle=True)
    prop_val_dl = DataLoader(prop_val_ds, batch_size=512, shuffle=False)

    propensity = PropensityNet(d).to(device)
    train_propensity(propensity, prop_train_dl, prop_val_dl, device)

    # 2) outcome (synthetic)
    out_train_dl = DataLoader(synth_train, batch_size=256, shuffle=True)
    out_val_dl = DataLoader(synth_val, batch_size=512, shuffle=False)

    outcome = OutcomeNet(d).to(device)
    train_outcome_synth(outcome, out_train_dl, out_val_dl, device)

    # 3) bias (human residual correction)
    human_dl = DataLoader(human_labeled, batch_size=128, shuffle=True)
    bias = BiasNet(d).to(device)
    train_bias(bias, outcome, human_dl, device)

    artifacts = Path("artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    ckpt_path = artifacts / "ckpt.pt"

    torch.save(
        {
            # 避免直接 pickle 自定义 class（新版本 torch.load 默认更严格）
            "cfg": cfg.__dict__,
            "d": d,
            "propensity": propensity.state_dict(),
            "outcome": outcome.state_dict(),
            "bias": bias.state_dict(),
        },
        ckpt_path,
    )

    print(f"\nSaved: {ckpt_path.resolve()}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()

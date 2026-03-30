from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from lora import LoRAConfig
from model import BaseMultimodal, LoRAProbe, ModelConfig, accuracy, kl_probe_to_base, route_by_nll_improvement
from toy_data import iter_batches, make_dataset, split_train_val


def to_torch(batch, device: str):
    x_text = torch.from_numpy(batch.x_text).to(device)
    x_img = torch.from_numpy(batch.x_img).to(device)
    y = torch.from_numpy(batch.y).long().to(device)
    dom = torch.from_numpy(batch.domain).long().to(device)
    return x_text, x_img, y, dom


def train_base(device: str = 'cpu') -> BaseMultimodal:
    x_text, x_img, y, domain = make_dataset(n=12000, d=16, n_domains=3, seed=0)
    (xtr_t, xtr_i, ytr, dtr), (xva_t, xva_i, yva, dva) = split_train_val(x_text, x_img, y, domain, seed=0)

    model = BaseMultimodal(ModelConfig(d_in=32, d_hidden=64)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    stream = iter_batches(xtr_t, xtr_i, ytr, dtr, batch_size=512, seed=1)

    for step in range(200):
        b = next(stream)
        xt, xi, yy, _ = to_torch(b, device)
        lg = model(xt, xi)
        loss = F.cross_entropy(lg, yy)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 50 == 0:
            with torch.no_grad():
                lg_va = model(torch.from_numpy(xva_t).to(device), torch.from_numpy(xva_i).to(device))
                acc = accuracy(lg_va, torch.from_numpy(yva).to(device).long())
            print(f'[base] step={step+1} val_acc={acc:.3f}')

    model.eval()
    return model


def train_probes(base: BaseMultimodal, use_kl: bool, device: str = 'cpu'):
    x_text, x_img, y, domain = make_dataset(n=12000, d=16, n_domains=3, seed=0)
    (xtr_t, xtr_i, ytr, dtr), (xva_t, xva_i, yva, dva) = split_train_val(x_text, x_img, y, domain, seed=0)

    probes = {}
    for dom in range(3):
        probes[dom] = LoRAProbe(base, LoRAConfig(r=8, alpha=16.0, dropout=0.05)).to(device)

    opts = {dom: torch.optim.AdamW(probes[dom].parameters(), lr=3e-3) for dom in probes.keys()}

    # Domain-specific batches
    rng = np.random.default_rng(0)
    dom_idx = {d: np.where(dtr == d)[0] for d in range(3)}

    kl_lambda = 0.5

    for step in range(200):
        dom = int(rng.integers(0, 3))
        idx = rng.choice(dom_idx[dom], size=256, replace=False)

        xt = torch.from_numpy(xtr_t[idx]).to(device)
        xi = torch.from_numpy(xtr_i[idx]).to(device)
        yy = torch.from_numpy(ytr[idx]).long().to(device)

        with torch.no_grad():
            base_logits = base(xt, xi)

        probe = probes[dom]
        logits = probe(xt, xi)
        loss = F.cross_entropy(logits, yy)
        if use_kl:
            loss = loss + kl_lambda * kl_probe_to_base(logits, base_logits)

        opts[dom].zero_grad()
        loss.backward()
        opts[dom].step()

        if (step + 1) % 50 == 0:
            with torch.no_grad():
                xt_va = torch.from_numpy(xva_t).to(device)
                xi_va = torch.from_numpy(xva_i).to(device)
                yy_va = torch.from_numpy(yva).long().to(device)
                base_logits = base(xt_va, xi_va)
                probe_logits = {d: probes[d](xt_va, xi_va) for d in probes.keys()}
                chosen, chosen_dom = route_by_nll_improvement(base_logits, probe_logits, yy_va)
                acc = accuracy(chosen, yy_va)

                counts = defaultdict(int)
                for dd in chosen_dom.cpu().numpy().tolist():
                    counts[int(dd)] += 1
                print(
                    f"[probes {'KL' if use_kl else 'noKL'}] step={step+1} routed_acc={acc:.3f} "
                    f"(base={counts[-1]}, d0={counts[0]}, d1={counts[1]}, d2={counts[2]})"
                )

    for p in probes.values():
        p.eval()
    return probes


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    base = train_base(device)

    probes_kl = train_probes(base, use_kl=True, device=device)
    probes_nokl = train_probes(base, use_kl=False, device=device)

    # Final comparison on a fresh validation set
    x_text, x_img, y, domain = make_dataset(n=12000, d=16, n_domains=3, seed=42)
    (_, _, _, _), (xva_t, xva_i, yva, dva) = split_train_val(x_text, x_img, y, domain, seed=42)

    xt = torch.from_numpy(xva_t).to(device)
    xi = torch.from_numpy(xva_i).to(device)
    yy = torch.from_numpy(yva).long().to(device)

    with torch.no_grad():
        base_logits = base(xt, xi)
        base_acc = accuracy(base_logits, yy)

        chosen_kl, _ = route_by_nll_improvement(base_logits, {d: probes_kl[d](xt, xi) for d in probes_kl}, yy)
        chosen_nokl, _ = route_by_nll_improvement(base_logits, {d: probes_nokl[d](xt, xi) for d in probes_nokl}, yy)

        acc_kl = accuracy(chosen_kl, yy)
        acc_nokl = accuracy(chosen_nokl, yy)

    print('\n=== Summary ===')
    print(f'Base acc:          {base_acc:.3f}')
    print(f'Routing + KL acc:  {acc_kl:.3f}')
    print(f'Routing + noKL acc:{acc_nokl:.3f}')


if __name__ == '__main__':
    main()

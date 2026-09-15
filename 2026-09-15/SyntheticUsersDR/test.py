from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import ToyDGPConfig, human_ate_from_x, make_test_splits
from model import BiasNet, OutcomeNet, PropensityNet, auc_roc_binary, corrected_prob, effective_sample_size


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def _naive_ate(dl: DataLoader) -> float:
    ys1, ys0 = [], []
    for batch in dl:
        t = batch["t"]
        y = batch["y"]
        ys1.append(y[t == 1])
        ys0.append(y[t == 0])
    m1 = torch.cat(ys1).mean().item()
    m0 = torch.cat(ys0).mean().item()
    return float(m1 - m0)


@torch.no_grad()
def _ipw_ate(dl: DataLoader, propensity: PropensityNet, device: torch.device, clip: float = 0.05) -> tuple[float, dict]:
    # w(x) = (1-e)/e, e=P(S=1|x)
    num1 = 0.0
    den1 = 0.0
    num0 = 0.0
    den0 = 0.0
    all_w = []

    for batch in dl:
        batch = _batch_to_device(batch, device)
        x = batch["x"]
        t = batch["t"]
        y = batch["y"]

        e = torch.sigmoid(propensity(x)).clamp(clip, 1.0 - clip)
        w = ((1.0 - e) / e).detach()
        all_w.append(w.cpu())

        w1 = w[t == 1]
        y1 = y[t == 1]
        w0 = w[t == 0]
        y0 = y[t == 0]

        num1 += float((w1 * y1).sum().cpu())
        den1 += float(w1.sum().cpu())
        num0 += float((w0 * y0).sum().cpu())
        den0 += float(w0.sum().cpu())

    mu1 = num1 / max(den1, 1e-12)
    mu0 = num0 / max(den0, 1e-12)
    w_all = torch.cat(all_w)
    logs = {
        "w_mean": float(w_all.mean()),
        "w_std": float(w_all.std()),
        "w_max": float(w_all.max()),
        "ess": effective_sample_size(w_all),
    }
    return float(mu1 - mu0), logs


@torch.no_grad()
def _dr_aipw_ate(
    synth_dl: DataLoader,
    human_calib_dl: DataLoader,
    propensity: PropensityNet,
    outcome: OutcomeNet,
    bias: BiasNet,
    device: torch.device,
    p_treat: float = 0.5,
    clip: float = 0.05,
) -> float:
    # 目标：估计 human ATE = E_h[Y(1)-Y(0)]
    # 这里的 AIPW/DR：
    #   mu(t) = E_s[w(x) * m_corr(x,t)]/E_s[w(x)]  +  E_h[ I(T=t)/p_t * (Y - m_corr(x,t)) ]
    # - 第一项：用 propensity weights 修 covariate shift（synthetic -> human）
    # - 第二项：用少量 human 标注样本做 augmentation，纠 concept shift / 模型误差

    def mu_t(t_int: int) -> float:
        # weighted model term on synthetic
        num = 0.0
        den = 0.0
        for batch in synth_dl:
            batch = _batch_to_device(batch, device)
            x = batch["x"]
            e = torch.sigmoid(propensity(x)).clamp(clip, 1.0 - clip)
            w = (1.0 - e) / e

            t = torch.full((x.shape[0],), t_int, device=device, dtype=torch.long)
            m = corrected_prob(outcome, bias, x, t)

            num += float((w * m).sum().cpu())
            den += float(w.sum().cpu())
        model_term = num / max(den, 1e-12)

        # augmentation term on human calib
        aug_terms = []
        for batch in human_calib_dl:
            batch = _batch_to_device(batch, device)
            xh = batch["x"]
            th = batch["t"]
            yh = batch["y"]

            m_h = corrected_prob(outcome, bias, xh, torch.full_like(th, t_int))
            aug = ((th == t_int).float() / p_treat) * (yh - m_h)
            aug_terms.append(aug)

        aug_term = torch.cat(aug_terms).mean().item()
        return float(model_term + aug_term)

    return float(mu_t(1) - mu_t(0))


@torch.no_grad()
def main() -> None:
    # 同 train.py：限制线程，提升沙箱环境稳定性
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load("artifacts/ckpt.pt", map_location="cpu", weights_only=False)
    cfg = ToyDGPConfig(**ckpt["cfg"])
    d = int(ckpt["d"])

    propensity = PropensityNet(d).to(device)
    outcome = OutcomeNet(d).to(device)
    bias = BiasNet(d).to(device)

    propensity.load_state_dict(ckpt["propensity"])
    outcome.load_state_dict(ckpt["outcome"])
    bias.load_state_dict(ckpt["bias"])

    propensity.eval()
    outcome.eval()
    bias.eval()

    ds, cfg = make_test_splits(seed=123, cfg=cfg)

    synth = ds["synth_test"]
    human_calib = ds["human_calib"]
    human_test = ds["human_test"]

    synth_dl = DataLoader(synth, batch_size=512, shuffle=False)
    human_calib_dl = DataLoader(human_calib, batch_size=256, shuffle=False)
    human_test_dl = DataLoader(human_test, batch_size=512, shuffle=False)

    # ----------------
    # 1) Diagnostics
    # ----------------
    # covariate shift: 用 propensity 在 held-out 的 synthetic/human 上做 AUC
    xs = []
    ys = []
    ps = []
    for batch in synth_dl:
        batch = _batch_to_device(batch, device)
        p = torch.sigmoid(propensity(batch["x"]))
        xs.append(batch["x"].detach().cpu())
        ys.append(torch.ones_like(p).detach().cpu())
        ps.append(p.detach().cpu())
    for batch in human_test_dl:
        batch = _batch_to_device(batch, device)
        p = torch.sigmoid(propensity(batch["x"]))
        xs.append(batch["x"].detach().cpu())
        ys.append(torch.zeros_like(p).detach().cpu())
        ps.append(p.detach().cpu())

    y_dom = torch.cat(ys)
    p_dom = torch.cat(ps)
    cov_auc = auc_roc_binary(y_dom, p_dom)

    # concept shift: synthetic outcome 模型在 synthetic vs human 的 BCE gap
    def bce_on(dl: DataLoader) -> float:
        losses = []
        for batch in dl:
            batch = _batch_to_device(batch, device)
            logits = outcome(batch["x"], batch["t"])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["y"].float())
            losses.append(float(loss.detach().cpu()))
        return sum(losses) / max(len(losses), 1)

    synth_bce = bce_on(synth_dl)
    human_bce = bce_on(human_calib_dl)
    concept_gap = human_bce - synth_bce

    print("=== Diagnostics ===")
    print(f"covariate_shift_auc(domain clf): {cov_auc:.3f} (越高越偏移)")
    print(f"concept_shift_gap(BCE human - BCE synth): {concept_gap:+.4f} (越大越 concept shift)")

    # ----------------
    # 2) Estimation
    # ----------------
    naive = _naive_ate(synth_dl)
    ipw, w_logs = _ipw_ate(synth_dl, propensity, device)
    dr = _dr_aipw_ate(synth_dl, human_calib_dl, propensity, outcome, bias, device)

    # ground-truth human ATE（用 DGP 的潜在结果概率，在 human_test 的 X 上 Monte-Carlo 积分）
    # 仅用于评估 ground truth（这里直接用 dataset 内部缓存的 covariates）
    xh = human_test.x
    true_ate = float(human_ate_from_x(xh, cfg).mean().cpu())

    def err(x: float) -> float:
        return abs(x - true_ate)

    print("\n=== ATE on held-out toy data ===")
    print(f"true_ate:  {true_ate:+.4f}")
    print(f"naive:     {naive:+.4f}  |abs_err|={err(naive):.4f}")
    print(f"ipw:       {ipw:+.4f}  |abs_err|={err(ipw):.4f}  (ESS={w_logs['ess']:.1f}, w_max={w_logs['w_max']:.2f})")
    print(f"dr(AIPW):  {dr:+.4f}  |abs_err|={err(dr):.4f}")

    # ----------------
    # 3) Trust decision
    # ----------------
    # 一个极简的“是否信任 synthetic 面板”决策逻辑：
    # - covariate shift 不大（AUC 接近 0.5）
    # - concept shift 不大（gap 接近 0）
    trust_naive = (cov_auc < 0.62) and (abs(concept_gap) < 0.02)
    choice = "naive" if trust_naive else "dr(AIPW)"

    print("\n=== Decision ===")
    print(f"trust_naive={trust_naive} -> choose: {choice}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()

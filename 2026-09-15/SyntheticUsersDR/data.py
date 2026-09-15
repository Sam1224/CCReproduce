from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


@dataclass(frozen=True)
class ToyDGPConfig:
    d: int = 6

    # covariate shift: synthetic 的 X 均值/尺度不同
    synth_mean_scale: float = 0.8

    # concept shift: synthetic 的响应机制不同（logit 上做系统性偏移）
    concept_bias: float = 0.4
    concept_scale: float = 0.9


def _human_x_mean(cfg: ToyDGPConfig) -> torch.Tensor:
    return torch.zeros(cfg.d)


def _synth_x_mean(cfg: ToyDGPConfig) -> torch.Tensor:
    base = torch.tensor([1.0, -1.0, 0.6, 0.2, 0.0, -0.7])
    return cfg.synth_mean_scale * base[: cfg.d]


def sample_x(domain: str, n: int, cfg: ToyDGPConfig, g: torch.Generator) -> torch.Tensor:
    if domain not in {"human", "synthetic"}:
        raise ValueError(f"unknown domain={domain}")

    x = torch.randn(n, cfg.d, generator=g)
    if domain == "human":
        mean = _human_x_mean(cfg)
        scale = torch.ones(cfg.d)
    else:
        mean = _synth_x_mean(cfg)
        # 让 synthetic 的协方差也略不同，强化 covariate shift
        scale = torch.tensor([1.3, 0.7, 1.1, 0.9, 1.0, 1.2])[: cfg.d]

    return x * scale + mean


def _human_logit(x: torch.Tensor, t: torch.Tensor, cfg: ToyDGPConfig) -> torch.Tensor:
    # 一个固定的“真实人类”响应机制（用于构造 ground truth）
    beta = torch.tensor([0.9, -0.6, 0.3, 0.0, 0.4, -0.2], device=x.device)[: cfg.d]
    tau = torch.tensor([0.6, 0.2, -0.1, 0.0, 0.1, 0.2], device=x.device)[: cfg.d]
    tau0 = torch.tensor(0.4, device=x.device)
    b0 = torch.tensor(-0.2, device=x.device)

    # logit = b0 + x@beta + t*(tau0 + x@tau)
    return b0 + x @ beta + t * (tau0 + x @ tau)


def _synthetic_logit(x: torch.Tensor, t: torch.Tensor, cfg: ToyDGPConfig) -> torch.Tensor:
    # synthetic concept shift：
    # 1) 系统性偏移（bias）
    # 2) 参数缩放（scale）
    # 3) 非线性偏置项，模拟 LLM 面板的“错位偏好”
    base = _human_logit(x, t, cfg)

    delta_beta = torch.tensor([-0.3, 0.25, 0.15, 0.0, -0.2, 0.1], device=x.device)[: cfg.d]
    delta_tau = torch.tensor([0.2, -0.15, 0.0, 0.0, 0.15, -0.1], device=x.device)[: cfg.d]

    nonlinear = 0.35 * torch.tanh(x @ torch.ones(cfg.d, device=x.device) / cfg.d) * (2.0 * t - 1.0)

    shifted = base + cfg.concept_bias + x @ delta_beta + t * (x @ delta_tau) + nonlinear
    return cfg.concept_scale * shifted


@torch.no_grad()
def human_potential_outcome_prob(x: torch.Tensor, t: int, cfg: ToyDGPConfig) -> torch.Tensor:
    tt = torch.full((x.shape[0],), float(t), device=x.device)
    return sigmoid(_human_logit(x, tt, cfg))


@torch.no_grad()
def human_ate_from_x(x: torch.Tensor, cfg: ToyDGPConfig) -> torch.Tensor:
    return human_potential_outcome_prob(x, 1, cfg) - human_potential_outcome_prob(x, 0, cfg)


class PanelDataset(Dataset):
    def __init__(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, source: int) -> None:
        super().__init__()
        self.x = x.float()
        self.t = t.long()
        self.y = y.float()
        self.source = int(source)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.x[idx],
            "t": self.t[idx],
            "y": self.y[idx],
            "source": torch.tensor(self.source, dtype=torch.long),
        }


class XOnlyDataset(Dataset):
    """只有 covariates 的数据集。

    为了复用默认 collate，这里也返回占位的 t/y 字段。
    """

    def __init__(self, x: torch.Tensor, source: int) -> None:
        super().__init__()
        self.x = x.float()
        self.source = int(source)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.x[idx],
            "t": torch.tensor(0, dtype=torch.long),
            "y": torch.tensor(0.0, dtype=torch.float),
            "source": torch.tensor(self.source, dtype=torch.long),
        }


def _sample_t(n: int, g: torch.Generator, p: float = 0.5) -> torch.Tensor:
    return (torch.rand(n, generator=g) < p).long()


def _sample_y(domain: str, x: torch.Tensor, t: torch.Tensor, cfg: ToyDGPConfig, g: torch.Generator) -> torch.Tensor:
    tt = t.float()
    if domain == "human":
        p = sigmoid(_human_logit(x, tt, cfg))
    elif domain == "synthetic":
        p = sigmoid(_synthetic_logit(x, tt, cfg))
    else:
        raise ValueError(domain)

    u = torch.rand(p.shape, generator=g, device=p.device)
    return (u < p).float()


def make_train_splits(
    seed: int = 0,
    cfg: ToyDGPConfig | None = None,
    n_synth_train: int = 8000,
    n_synth_val: int = 2000,
    n_human_labeled: int = 800,
    n_human_cov: int = 4000,
) -> Tuple[Dict[str, Dataset], ToyDGPConfig]:
    cfg = cfg or ToyDGPConfig()
    g = torch.Generator().manual_seed(seed)

    # synthetic labeled
    xs_tr = sample_x("synthetic", n_synth_train, cfg, g)
    ts_tr = _sample_t(n_synth_train, g)
    ys_tr = _sample_y("synthetic", xs_tr, ts_tr, cfg, g)

    xs_va = sample_x("synthetic", n_synth_val, cfg, g)
    ts_va = _sample_t(n_synth_val, g)
    ys_va = _sample_y("synthetic", xs_va, ts_va, cfg, g)

    # human labeled（少量）
    xh = sample_x("human", n_human_labeled, cfg, g)
    th = _sample_t(n_human_labeled, g)
    yh = _sample_y("human", xh, th, cfg, g)

    # human covariates only（可理解为：能拿到用户画像，但拿不到真实反馈）
    xh_cov = sample_x("human", n_human_cov, cfg, g)

    ds: Dict[str, Dataset] = {
        "synth_train": PanelDataset(xs_tr, ts_tr, ys_tr, source=1),
        "synth_val": PanelDataset(xs_va, ts_va, ys_va, source=1),
        "human_labeled": PanelDataset(xh, th, yh, source=0),
        "human_cov": XOnlyDataset(xh_cov, source=0),
    }
    return ds, cfg


def make_test_splits(
    seed: int = 123,
    cfg: ToyDGPConfig | None = None,
    n_synth_test: int = 4000,
    n_human_calib: int = 800,
    n_human_test: int = 5000,
) -> Tuple[Dict[str, Dataset], ToyDGPConfig]:
    cfg = cfg or ToyDGPConfig()
    g = torch.Generator().manual_seed(seed)

    xs = sample_x("synthetic", n_synth_test, cfg, g)
    ts = _sample_t(n_synth_test, g)
    ys = _sample_y("synthetic", xs, ts, cfg, g)

    xh_cal = sample_x("human", n_human_calib, cfg, g)
    th_cal = _sample_t(n_human_calib, g)
    yh_cal = _sample_y("human", xh_cal, th_cal, cfg, g)

    xh_te = sample_x("human", n_human_test, cfg, g)
    th_te = _sample_t(n_human_test, g)
    yh_te = _sample_y("human", xh_te, th_te, cfg, g)

    ds: Dict[str, Dataset] = {
        "synth_test": PanelDataset(xs, ts, ys, source=1),
        "human_calib": PanelDataset(xh_cal, th_cal, yh_cal, source=0),
        "human_test": PanelDataset(xh_te, th_te, yh_te, source=0),
    }
    return ds, cfg

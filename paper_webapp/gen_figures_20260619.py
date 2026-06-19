#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate light, research-style matplotlib figures for the 2026-06-19 paper-radar inspection.

For each of today's 4 papers we produce two PNGs in ``assets/figures``:
  * ``{id}.png``      -- a clean schematic of the method pipeline (methodology figure)
  * ``{id}_exp.png``  -- a chart of the paper's standout experimental results (experiment figure)

The methodology figure is drawn as labelled rounded boxes + arrows (input -> modules ->
output) with small input/output samples, matching the light research style of the
hand-built SVG figures from gen_figures_20260617/18.py. The experiment figure uses
matplotlib bar/line charts of the headline numbers.

Run:  python gen_figures_20260619.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

WEBAPP_DIR = Path(__file__).resolve().parent
FIG_DIR = WEBAPP_DIR / "assets" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- soft light research palette --------------------------------------------
INK = "#0f172a"
SUB = "#475569"
MUTED = "#64748b"
EDGE = "#94a3b8"
BG = "#ffffff"

# box (fill, stroke, title-color) by accent
ACC = {
    "blue":   ("#eff6ff", "#bfdbfe", "#2563eb"),
    "cyan":   ("#ecfeff", "#a5f3fc", "#0891b2"),
    "teal":   ("#f0fdfa", "#99f6e4", "#0d9488"),
    "green":  ("#ecfdf5", "#bbf7d0", "#10b981"),
    "amber":  ("#fffbeb", "#fde68a", "#d97706"),
    "violet": ("#f5f3ff", "#ddd6fe", "#7c3aed"),
    "rose":   ("#fff1f2", "#fecdd3", "#e11d48"),
    "slate":  ("#f8fafc", "#e2e8f0", "#334155"),
}
BAR_COLORS = ["#2563eb", "#0891b2", "#10b981", "#f59e0b", "#7c3aed", "#e11d48"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": EDGE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SUB,
    "ytick.color": SUB,
    "axes.titlecolor": INK,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
})


# ============================ schematic helpers ==============================
def _box(ax, x, y, w, h, accent, title, lines=None, title_size=11, body_size=9.0):
    fill, stroke, tc = ACC[accent]
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.4, edgecolor=stroke, facecolor=fill, zorder=2,
    )
    ax.add_patch(p)
    cx = x + w / 2
    if lines:
        ax.text(cx, y + h - 0.052, title, ha="center", va="top",
                fontsize=title_size, fontweight="bold", color=tc, zorder=3)
        ty = y + h - 0.052 - 0.060
        for ln in lines:
            ax.text(cx, ty, ln, ha="center", va="top", fontsize=body_size,
                    color=SUB, zorder=3)
            ty -= 0.050
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=tc, zorder=3)


def _arrow(ax, x1, y1, x2, y2, color=None, dashed=False, rad=0.0):
    color = color or EDGE
    style = "->,head_width=3.4,head_length=6"
    ls = (0, (5, 4)) if dashed else "solid"
    a = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=1.0,
        linewidth=1.9, color=color, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", zorder=1,
    )
    ax.add_patch(a)


def _canvas(title, subtitle, figsize=(11.0, 4.5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # accent header bar
    ax.add_patch(plt.Rectangle((0.012, 0.93), 0.10, 0.045, color="#2563eb",
                               transform=ax.transAxes, zorder=4))
    ax.text(0.012, 0.905, title, ha="left", va="top", fontsize=15.5,
            fontweight="bold", color=INK)
    ax.text(0.012, 0.852, subtitle, ha="left", va="top", fontsize=9.6, color=MUTED)
    return fig, ax


def _save(fig, name):
    out = FIG_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size} bytes)")


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _bar_labels(ax, bars, fmt="{:.1f}", dy=0.0, suffix=""):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h) + suffix,
                ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=INK)


# =============================================================================
# 1) PerceptionDLM (2606.19534)
# =============================================================================
def perceptiondlm_method():
    fig, ax = _canvas(
        "PerceptionDLM — Parallel Region Perception with a Diffusion LM",
        "Image + region masks  ->  region prompting + RoI replay + block-wise attention  ->  single diffusion denoise  ->  parallel captions",
    )
    y = 0.42
    h = 0.30
    w = 0.175
    xs = [0.012, 0.213, 0.414, 0.615, 0.816]
    _box(ax, xs[0], y, w, h, "cyan", "Input",
         ["image + N region", "masks (R1..RN)", "shared text prompt"], title_size=10.5)
    _box(ax, xs[1], y, w, h, "blue", "Region Prompting",
         ["learnable embed", "per region binds", "region identity"], title_size=10.5)
    _box(ax, xs[2], y, w, h, "violet", "Block Attn Mask",
         ["each region sees", "only global +", "own RoI / span"], title_size=10.5)
    _box(ax, xs[3], y, w, h, "amber", "Diffusion Denoise",
         ["LLaDA-8B backbone", "all regions decoded", "in ONE pass"], title_size=10.5)
    _box(ax, xs[4], y, w, h, "green", "Parallel Captions",
         ["cap(R1) || cap(R2)", "|| ... || cap(RN)", "no serial latency"], title_size=10.5)
    for i in range(4):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)
    # RoI feature replay note box (below, feeding the attention/denoise stage)
    _box(ax, 0.213, 0.13, 0.577, 0.15, "slate", "RoI-aligned feature replay",
         ["inject each region's pooled RoI features back into the denoising stream"],
         title_size=10, body_size=8.6)
    _arrow(ax, 0.50, 0.28, 0.50, y - 0.005, color="#0d9488", dashed=True)
    ax.text(0.5, 0.045, "Autoregressive baselines decode regions serially (latency grows with N); "
                        "PerceptionDLM exploits diffusion-LM token parallelism to decode all regions at once.",
            ha="center", va="bottom", fontsize=9, color=MUTED, style="italic")
    _save(fig, "2606.19534.png")


def perceptiondlm_exp():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    fig.suptitle("PerceptionDLM — standout results (ParaDLC-Bench & efficiency)",
                 fontsize=14, fontweight="bold", x=0.012, ha="left", y=1.005)

    ax = axes[0]
    models = ["LLaDA-V", "SDAR-VL", "Perception\nDLM", "GAR-8B\n(AR)"]
    acc = [35.2, 31.3, 62.4, 69.5]
    colors = ["#94a3b8", "#94a3b8", "#2563eb", "#10b981"]
    bars = ax.bar(models, acc, color=colors, width=0.62, zorder=3)
    _bar_labels(ax, bars, fmt="{:.1f}")
    ax.set_ylabel("ParaDLC-Bench avg accuracy (%)")
    ax.set_ylim(0, 80)
    ax.set_title("Near-doubles diffusion-VLM accuracy", fontsize=11, color=SUB)
    _style_axes(ax)
    ax.annotate("diffusion VLMs", xy=(0.5, 38), ha="center", fontsize=8.5, color=MUTED)

    ax = axes[1]
    methods = ["Perception\nDLM", "GAR", "PixelRefer"]
    times = [276, 479, 718]
    colors = ["#2563eb", "#f59e0b", "#e11d48"]
    bars = ax.bar(methods, times, color=colors, width=0.6, zorder=3)
    _bar_labels(ax, bars, fmt="{:.0f}", suffix="s")
    ax.set_ylabel("Total benchmark inference time (s)")
    ax.set_ylim(0, 820)
    ax.set_title("Up to 3.44x–3.5x throughput  ·  TPF 2.9 vs 1.0", fontsize=11, color=SUB)
    _style_axes(ax)
    ax.annotate("1.7x faster\nthan GAR", xy=(0, 276), xytext=(0.45, 470),
                fontsize=8.6, color="#2563eb", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.3))

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.19534_exp.png")


# =============================================================================
# 2) NNN (2606.17910)
# =============================================================================
def nnn_method():
    fig, ax = _canvas(
        "NNN — Non-negative Elastic Net Decoding for Retrieval",
        "Query embedding  ->  sparse non-negative reconstruction from doc embeddings (elastic net, FISTA)  ->  corpus-aware, low-redundancy set",
    )
    y = 0.36
    h = 0.30
    _box(ax, 0.015, y, 0.17, h, "cyan", "Query embedding",
         ["q  (dense vector)"])
    _box(ax, 0.215, y, 0.20, h, "slate", "Document embeddings",
         ["corpus matrix D", "= [d1, d2, ... dM]"])
    _box(ax, 0.445, y + 0.025, 0.235, h - 0.05, "violet", "Elastic-Net Reconstruction",
         ["min ||q - D w||^2", "+ a||w||1 + b||w||2", "w >= 0  (non-negative)"])
    _box(ax, 0.71, y + 0.025, 0.13, h - 0.05, "amber", "FISTA Solver",
         ["unrolled,", "end-to-end", "trainable"])
    _box(ax, 0.86, y, 0.13, h, "green", "Selected set",
         ["support(w):", "corpus-aware,", "low redundancy"])
    _box(ax, 0.215, 0.07, 0.465, 0.14, "rose", "Why it differs from dense retrieval",
         ["dense scores each doc independently (corpus-blind); NNN makes docs compete & complement"],
         title_size=10, body_size=8.6)
    _arrow(ax, 0.185, y + h / 2, 0.214, y + h / 2)
    _arrow(ax, 0.415, y + h / 2, 0.444, y + h / 2)
    _arrow(ax, 0.68, y + h / 2, 0.709, y + h / 2)
    _arrow(ax, 0.84, y + h / 2, 0.859, y + h / 2)
    _save(fig, "2606.17910.png")


def nnn_exp():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    fig.suptitle("NNN — standout results (completeness & end-to-end gains)",
                 fontsize=14, fontweight="bold", x=0.012, ha="left", y=1.005)

    ax = axes[0]
    labels = ["Dense\nretrieval", "NNN\n(frozen emb.)"]
    vals = [100, 136]
    bars = ax.bar(labels, vals, color=["#94a3b8", "#2563eb"], width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="{:.0f}")
    ax.set_ylabel("Completeness (relative, dense = 100)")
    ax.set_ylim(0, 160)
    ax.set_title("Up to +36% completeness on frozen embeddings", fontsize=11, color=SUB)
    _style_axes(ax)
    ax.annotate("+36%", xy=(1, 136), xytext=(1, 150), ha="center",
                fontsize=11, fontweight="bold", color="#2563eb")

    ax = axes[1]
    k = [1, 2, 3, 5, 8, 12]
    dense = [100, 100, 100, 100, 100, 100]
    nnn = [108, 116, 123, 130, 134, 136]
    ax.plot(k, dense, "--", color="#94a3b8", lw=2, marker="o", label="Dense retrieval")
    ax.plot(k, nnn, "-", color="#10b981", lw=2.4, marker="o", label="NNN decoding")
    ax.fill_between(k, dense, nnn, color="#10b981", alpha=0.12)
    ax.set_xlabel("# relevant documents per query")
    ax.set_ylabel("Completeness (relative)")
    ax.set_ylim(90, 150)
    ax.set_title("Gap widens with more relevant docs", fontsize=11, color=SUB)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.17910_exp.png")


# =============================================================================
# 3) VRP (2606.17389)
# =============================================================================
def vrp_method():
    fig, ax = _canvas(
        "VRP — Disentangling Spatial Attention from Reliability in VLMs",
        "VLM  ->  (A) structural attention metrics  +  (B) self-consistency over sampled paths  ->  finding: attention != reliability; consistency does",
    )
    _box(ax, 0.015, 0.36, 0.16, 0.30, "cyan", "VLM",
         ["image + question", "LLaVA / Qwen2-VL", "/ PaliGemma"])
    _box(ax, 0.215, 0.55, 0.30, 0.20, "violet", "(A) Structural attention",
         ["cluster count Ck · spatial entropy Hs", "cross-layer evolution dHs"],
         title_size=10.5, body_size=8.8)
    _box(ax, 0.215, 0.26, 0.30, 0.20, "teal", "(B) Self-consistency",
         ["sample multiple reasoning paths", "agreement rate across answers"],
         title_size=10.5, body_size=8.8)
    _box(ax, 0.565, 0.55, 0.20, 0.20, "rose", "R ~ 0.001",
         ["attention is", "uncorrelated", "with accuracy"], title_size=11, body_size=8.8)
    _box(ax, 0.565, 0.26, 0.20, 0.20, "green", "R = 0.429",
         ["consistency is the", "dominant truth", "predictor"], title_size=11, body_size=8.8)
    _box(ax, 0.80, 0.36, 0.19, 0.30, "amber", "Takeaway",
         ["judge reliability by", "generation consistency", "+ hidden-state probes,", "NOT attention maps"],
         title_size=10.5, body_size=8.6)
    _arrow(ax, 0.175, 0.55, 0.214, 0.62)
    _arrow(ax, 0.175, 0.47, 0.214, 0.38)
    _arrow(ax, 0.515, 0.65, 0.564, 0.65, color="#e11d48")
    _arrow(ax, 0.515, 0.36, 0.564, 0.36, color="#10b981")
    _arrow(ax, 0.765, 0.65, 0.80, 0.55)
    _arrow(ax, 0.765, 0.36, 0.80, 0.47)
    _save(fig, "2606.17389.png")


def vrp_exp():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    fig.suptitle("VRP — standout results (correlation & layer-ablation robustness)",
                 fontsize=14, fontweight="bold", x=0.012, ha="left", y=1.005)

    ax = axes[0]
    labels = ["Spatial attention\nvs accuracy", "Self-consistency\nvs truth"]
    vals = [0.001, 0.429]
    bars = ax.bar(labels, vals, color=["#e11d48", "#10b981"], width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="{:.3f}")
    ax.set_ylabel("Correlation with correctness (R)")
    ax.set_ylim(0, 0.5)
    ax.set_title("Attention ~0 ; consistency is the signal", fontsize=11, color=SUB)
    _style_axes(ax)

    ax = axes[1]
    x = [0, 20, 35, 50, 65, 80]
    qwen = [62, 61, 60, 58, 52, 44]
    paligemma = [60, 59, 58, 56, 50, 41]
    llava = [61, 55, 46, 33, 20, 12]
    ax.plot(x, qwen, "-o", color="#2563eb", lw=2.2, label="Qwen2-VL")
    ax.plot(x, paligemma, "-o", color="#0d9488", lw=2.2, label="PaliGemma")
    ax.plot(x, llava, "-o", color="#e11d48", lw=2.2, label="LLaVA")
    ax.axvline(50, color="#94a3b8", ls="--", lw=1.2)
    ax.text(50, 64, "~50% of most\npredictive layer", ha="center", fontsize=8.2, color=MUTED)
    ax.set_xlabel("% of most predictive layer destroyed")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 70)
    ax.set_title("Qwen2-VL / PaliGemma robust ; LLaVA collapses", fontsize=11, color=SUB)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.17389_exp.png")


# =============================================================================
# 4) LoopWM (2606.18208)
# =============================================================================
def loopwm_method():
    fig, ax = _canvas(
        "LoopWM — Looped World Models",
        "Shared transformer block applied iteratively (looped) to refine latent state, with an adaptive number of loops per step",
    )
    _box(ax, 0.015, 0.36, 0.16, 0.30, "cyan", "Input",
         ["observation +", "action a_t", "-> latent s0"])
    # shared looped block
    _box(ax, 0.30, 0.34, 0.28, 0.34, "violet", "Shared Transformer Block",
         ["one parameter-shared block", "applied K times (looped)", "refines latent state s_k"],
         title_size=11, body_size=9.0)
    # loop arrow
    _arrow(ax, 0.58, 0.62, 0.58, 0.40, color="#7c3aed", rad=-0.9)
    ax.text(0.66, 0.51, "loop x K", ha="center", va="center", fontsize=9.5,
            color="#7c3aed", fontweight="bold", rotation=90)
    _box(ax, 0.30, 0.095, 0.28, 0.19, "amber", "Adaptive computation",
         ["K = effective depth,", "scaled by step difficulty"],
         title_size=10, body_size=8.8)
    _arrow(ax, 0.44, 0.34, 0.44, 0.29, color="#d97706", dashed=True)
    _box(ax, 0.71, 0.36, 0.275, 0.30, "green", "Predicted next state",
         ["faithful long-horizon", "rollout at a fraction", "of the parameters"])
    _arrow(ax, 0.175, 0.51, 0.299, 0.51)
    _arrow(ax, 0.58, 0.51, 0.709, 0.51)
    ax.text(0.5, 0.03,
            "Iterative latent depth becomes a NEW scaling axis, orthogonal to model size and training data.",
            ha="center", va="bottom", fontsize=9, color=MUTED, style="italic")
    _save(fig, "2606.18208.png")


def loopwm_exp():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    fig.suptitle("LoopWM — standout results (parameter efficiency & extreme categories)",
                 fontsize=14, fontweight="bold", x=0.012, ha="left", y=1.005)

    ax = axes[0]
    labels = ["Conventional\nworld model", "LoopWM"]
    params = [100, 1]
    bars = ax.bar(labels, params, color=["#94a3b8", "#7c3aed"], width=0.5, zorder=3)
    ax.set_yscale("log")
    for b, v in zip(bars, params):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.25,
                ("1x" if v == 100 else "0.01x"), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=INK)
    ax.set_ylabel("Relative parameters (log scale)")
    ax.set_ylim(0.5, 300)
    ax.set_title("Up to 100x parameter efficiency", fontsize=11, color=SUB)
    ax.annotate("", xy=(1, 1.6), xytext=(1, 90),
                arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1.6))
    ax.text(1.18, 12, "100x\nfewer", color="#7c3aed", fontsize=9.5, fontweight="bold")
    _style_axes(ax)
    ax.grid(axis="y", which="both", color="#e5e7eb", linewidth=0.7)

    ax = axes[1]
    cats = ["AlfWorld\n(overall)", "Lifespan\n(extreme)"]
    base = [71, 0]
    loop = [83, 100]
    xpos = range(len(cats))
    w = 0.36
    b1 = ax.bar([x - w / 2 for x in xpos], base, width=w, color="#94a3b8",
                label="Best baseline", zorder=3)
    b2 = ax.bar([x + w / 2 for x in xpos], loop, width=w, color="#10b981",
                label="LoopWM", zorder=3)
    _bar_labels(ax, b1, fmt="{:.0f}", suffix="%")
    _bar_labels(ax, b2, fmt="{:.0f}", suffix="%")
    ax.set_xticks(list(xpos))
    ax.set_xticklabels(cats)
    ax.set_ylabel("World-modelling accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Beats baselines ; Lifespan 0% -> 100%", fontsize=11, color=SUB)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.18208_exp.png")


FIGS = [
    perceptiondlm_method, perceptiondlm_exp,
    nnn_method, nnn_exp,
    vrp_method, vrp_exp,
    loopwm_method, loopwm_exp,
]


def main() -> None:
    for fn in FIGS:
        fn()
    print(f"\nDone: {len(FIGS)} figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()

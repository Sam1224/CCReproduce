#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate light, research-style SVG figures for the 2026-06-25 paper inspection.

For each paper we emit two SVGs in ``assets/figures``:
  * ``{id}.svg``      -- a clean schematic of the method/pipeline
  * ``{id}_exp.svg``  -- a compact chart of standout experimental numbers

Run:
  python gen_figures_20260625.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

WEBAPP_DIR = Path(__file__).resolve().parent
FIG_DIR = WEBAPP_DIR / "assets" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- soft light research palette --------------------------------------------
INK, SUB, MUTED, EDGE, BG = "#0f172a", "#475569", "#64748b", "#94a3b8", "#ffffff"
ACC = {
    "blue": ("#eff6ff", "#bfdbfe", "#2563eb"),
    "cyan": ("#ecfeff", "#a5f3fc", "#0891b2"),
    "teal": ("#f0fdfa", "#99f6e4", "#0d9488"),
    "green": ("#ecfdf5", "#bbf7d0", "#10b981"),
    "amber": ("#fffbeb", "#fde68a", "#d97706"),
    "violet": ("#f5f3ff", "#ddd6fe", "#7c3aed"),
    "rose": ("#fff1f2", "#fecdd3", "#e11d48"),
    "slate": ("#f8fafc", "#e2e8f0", "#334155"),
}

plt.rcParams.update(
    {
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
    }
)


# ============================ schematic helpers ==============================

def _box(ax, x, y, w, h, accent, title, lines=None, title_size=11, body_size=9.0):
    fill, stroke, tc = ACC[accent]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.4,
            edgecolor=stroke,
            facecolor=fill,
            zorder=2,
        )
    )
    cx = x + w / 2
    if lines:
        ax.text(
            cx,
            y + h - 0.052,
            title,
            ha="center",
            va="top",
            fontsize=title_size,
            fontweight="bold",
            color=tc,
            zorder=3,
        )
        ty = y + h - 0.052 - 0.060
        for ln in lines:
            ax.text(cx, ty, ln, ha="center", va="top", fontsize=body_size, color=SUB, zorder=3)
            ty -= 0.050
    else:
        ax.text(
            cx,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=tc,
            zorder=3,
        )


def _arrow(ax, x1, y1, x2, y2, color=None, dashed=False, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->,head_width=3.4,head_length=6",
            mutation_scale=1.0,
            linewidth=1.9,
            color=color or EDGE,
            linestyle=(0, (5, 4)) if dashed else "solid",
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
    )


def _canvas(title, subtitle, figsize=(11.0, 4.6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.012, 0.93), 0.10, 0.045, color="#2563eb", transform=ax.transAxes, zorder=4))
    ax.text(0.012, 0.905, title, ha="left", va="top", fontsize=15.5, fontweight="bold", color=INK)
    ax.text(0.012, 0.852, subtitle, ha="left", va="top", fontsize=9.6, color=MUTED)
    return fig, ax


def _save(fig, name):
    out = FIG_DIR / name
    fig.savefig(out, format="svg", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _bar_labels(ax, bars, fmt="{:.1f}", dy=0.0, suffix=""):
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + dy,
            fmt.format(h) + suffix,
            ha="center",
            va="bottom",
            fontsize=9.3,
            fontweight="bold",
            color=INK,
        )


# =============================================================================
# 1) UNIVID (2606.05748)
# =============================================================================

def univid_method():
    fig, ax = _canvas(
        "UNIVID — Unified Vision-Language Model for Video Moderation",
        "Video -> (Risk Filter) -> (Caption-as-evidence) -> RAG policy decision -> Trend governance",
    )
    y, h, w = 0.40, 0.30, 0.19
    xs = [0.012, 0.235, 0.458, 0.681, 0.814]
    _box(ax, xs[0], y, w, h, "cyan", "Input",
         ["short-form video", "(frames + audio)", "policy taxonomy"], title_size=10.5)
    _box(ax, xs[1], y, w, h, "blue", "Risk Filter",
         ["UNIVID embedding", "high-throughput", "pre-screen"], title_size=10.5)
    _box(ax, xs[2], y, w, h, "violet", "UNIVID-Lite",
         ["policy-aware", "captions as", "evidence"], title_size=10.5)
    _box(ax, xs[3], y, w, h, "amber", "UNIVID-RAG",
         ["retrieve policy", "+ examples", "decision"], title_size=10.5)
    _box(ax, xs[4], y, 0.175, h, "green", "Trend",
         ["few-shot", "emerging risks"], title_size=10.5)
    for i in range(4):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)
    ax.text(
        0.5,
        0.06,
        "Key idea: turn moderation into interpretable caption generation; captions become verifiable evidence.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
        style="italic",
    )
    _save(fig, "2606.05748.svg")


def univid_exp():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    fig.suptitle(
        "UNIVID — production-style gains (relative)",
        fontsize=13.5,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=1.005,
    )

    ax = axes[0]
    labels = ["Leakage↓", "Overkill↓"]
    vals = [42.7, 37.0]
    bars = ax.bar(labels, vals, color=["#2563eb", "#10b981"], width=0.6, zorder=3)
    _bar_labels(ax, bars, fmt="{:.1f}", suffix="%")
    ax.set_ylabel("Relative reduction")
    ax.set_ylim(0, 60)
    ax.set_title("Lower violation leakage & overkill", fontsize=11, color=SUB)
    _style_axes(ax)

    ax = axes[1]
    labels = ["Recall\n(before RAG)", "Recall\n(with RAG)"]
    vals = [39.3, 53.6]
    bars = ax.bar(labels, vals, color=["#94a3b8", "#d97706"], width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="{:.1f}")
    ax.set_ylabel("Leakage recall (%)")
    ax.set_ylim(0, 70)
    ax.set_title("RAG boosts recall (39.3 → 53.6)", fontsize=11, color=SUB)
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.05748_exp.svg")


# =============================================================================
# 2) VCG (2606.19627v1)
# =============================================================================

def vcg_method():
    fig, ax = _canvas(
        "VCG — Multimodal Retrieval for E-commerce Video Feeds",
        "Extreme cold-start: semantic-first CLIP embeddings + kNN vector search + LVLM-as-a-judge",
    )
    _box(ax, 0.012, 0.42, 0.19, 0.30, "cyan", "User history",
         ["clicked products", "(images/text)", "time decay"], title_size=10.5)
    _box(ax, 0.235, 0.42, 0.19, 0.30, "blue", "CLIP embed",
         ["domain-adapted", "product/video", "shared space"], title_size=10.5)
    _box(ax, 0.458, 0.42, 0.19, 0.30, "violet", "Vector DB",
         ["kNN search", "P50 ~17.5ms", "online serving"], title_size=10.5)
    _box(ax, 0.681, 0.42, 0.19, 0.30, "amber", "Video feed",
         ["Product→Video", "Video→Product", "Zero-shot"], title_size=10.5)
    _box(ax, 0.815, 0.12, 0.17, 0.18, "green", "LVLM Judge",
         ["mitigate", "exposure bias"], title_size=10.2, body_size=8.8)

    _arrow(ax, 0.202, 0.57, 0.235, 0.57)
    _arrow(ax, 0.425, 0.57, 0.458, 0.57)
    _arrow(ax, 0.648, 0.57, 0.681, 0.57)
    _arrow(ax, 0.83, 0.42, 0.90, 0.30, dashed=True, color="#10b981")

    ax.text(
        0.5,
        0.06,
        "Key idea: replace CF with semantic retrieval for cold-start video feeds.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
        style="italic",
    )
    _save(fig, "2606.19627v1.svg")


def vcg_exp():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    fig.suptitle(
        "VCG — online uplift & latency",
        fontsize=13.5,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=1.005,
    )

    ax = axes[0]
    labels = ["VideoProgress@50%", "Start→25% conv"]
    vals = [50.10, 30.32]
    bars = ax.bar(labels, vals, color=["#2563eb", "#10b981"], width=0.58, zorder=3)
    _bar_labels(ax, bars, fmt="{:.2f}", suffix="%")
    ax.set_ylabel("Relative lift")
    ax.set_ylim(0, 60)
    ax.set_title("Large online gains in cold-start", fontsize=11, color=SUB)
    _style_axes(ax)

    ax = axes[1]
    labels = ["P50 latency"]
    vals = [17.5]
    bars = ax.bar(labels, vals, color=["#0891b2"], width=0.5, zorder=3)
    _bar_labels(ax, bars, fmt="{:.1f}", suffix="ms")
    ax.set_ylabel("Median latency")
    ax.set_ylim(0, 25)
    ax.set_title("Vector retrieval is real-time", fontsize=11, color=SUB)
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.19627v1_exp.svg")


# =============================================================================
# 3) SAERec (2606.18897v1)
# =============================================================================

def saerec_method():
    fig, ax = _canvas(
        "SAERec — Interpretable Intent Priors via Sparse Autoencoders",
        "Reviews/text -> LLM embeddings -> SAE disentangle intents -> retrieve priors -> sequential recommender",
    )
    y, h, w = 0.40, 0.30, 0.18
    xs = [0.012, 0.215, 0.418, 0.621, 0.824]
    _box(ax, xs[0], y, w, h, "cyan", "Text data",
         ["reviews", "product text", "user feedback"], title_size=10.5)
    _box(ax, xs[1], y, w, h, "blue", "LLM embed",
         ["dense vectors", "polysemantic", "noisy"], title_size=10.5)
    _box(ax, xs[2], y, w, h, "violet", "SAE",
         ["sparse atoms", "intent directions", "interpretable"], title_size=10.5)
    _box(ax, xs[3], y, w, h, "amber", "Retrieve",
         ["personal intents", "+ public intents", "priors"], title_size=10.5)
    _box(ax, xs[4], y, w, h, "green", "SeqRec",
         ["attention fusion", "better ranking", "explainable"], title_size=10.5)
    for i in range(4):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)
    ax.text(0.5, 0.06, "Key idea: SAE turns LLM embeddings into interpretable intent atoms for Rec.",
            ha="center", va="bottom", fontsize=9, color=MUTED, style="italic")
    _save(fig, "2606.18897v1.svg")


def saerec_exp():
    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    fig.suptitle(
        "SAERec — HR@10 gains on multiple datasets",
        fontsize=13.5,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=1.005,
    )
    ds = ["Beauty", "Toys", "Sports", "Yelp"]
    hr = [0.1055, 0.1147, 0.0611, 0.0452]
    bars = ax.bar(ds, hr, color=["#2563eb", "#10b981", "#0891b2", "#7c3aed"], width=0.6, zorder=3)
    _bar_labels(ax, bars, fmt="{:.4f}")
    ax.set_ylabel("HR@10")
    ax.set_ylim(0, 0.14)
    ax.set_title("Higher hit-rate with intent priors", fontsize=11, color=SUB)
    _style_axes(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.18897v1_exp.svg")


# =============================================================================
# 4) Beyond Item IDs (2606.07546)
# =============================================================================

def semanticid_method():
    fig, ax = _canvas(
        "Beyond Item IDs — Semantic-Native Long Sequence Rec",
        "Semantic IDs (RQ-VAE) + Temporal Folding -> efficient 2000+ sequence modeling",
    )
    _box(ax, 0.012, 0.42, 0.20, 0.30, "cyan", "Content",
         ["video frames", "text/audio", "multimodal embed"], title_size=10.5)
    _box(ax, 0.245, 0.42, 0.20, 0.30, "violet", "Semantic IDs",
         ["RQ-VAE", "bounded vocab", "cold-start"], title_size=10.5)
    _box(ax, 0.478, 0.55, 0.22, 0.18, "blue", "Temporal Folding",
         ["super-tokens", "-83.9% step time"], title_size=10.2, body_size=8.8)
    _box(ax, 0.478, 0.30, 0.22, 0.18, "amber", "Global Query",
         ["capture long intent"], title_size=10.2, body_size=8.8)
    _box(ax, 0.735, 0.42, 0.25, 0.30, "green", "Ranking",
         ["2000+ sequence", "online serving", "better freshness"], title_size=10.5)
    _arrow(ax, 0.212, 0.57, 0.245, 0.57)
    _arrow(ax, 0.445, 0.57, 0.478, 0.64)
    _arrow(ax, 0.445, 0.57, 0.478, 0.39)
    _arrow(ax, 0.698, 0.57, 0.735, 0.57)
    ax.text(0.5, 0.06, "Key idea: replace ID tables with Semantic IDs and scale sequence length efficiently.",
            ha="center", va="bottom", fontsize=9, color=MUTED, style="italic")
    _save(fig, "2606.07546.svg")


def semanticid_exp():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    fig.suptitle(
        "Beyond Item IDs — online gains & efficiency",
        fontsize=13.5,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=1.005,
    )

    ax = axes[0]
    labels = ["Sat. watch", "Sat. views", "Active users"]
    vals = [1.42, 1.08, 0.52]
    bars = ax.bar(labels, vals, color=["#2563eb", "#10b981", "#0891b2"], width=0.58, zorder=3)
    _bar_labels(ax, bars, fmt="{:.2f}", suffix="%")
    ax.set_ylabel("Relative lift")
    ax.set_ylim(0, 2.0)
    ax.set_title("Stable online improvements", fontsize=11, color=SUB)
    _style_axes(ax)

    ax = axes[1]
    labels = ["Step time↓", "HBM↓"]
    vals = [83.9, 92.2]
    bars = ax.bar(labels, vals, color=["#d97706", "#7c3aed"], width=0.58, zorder=3)
    _bar_labels(ax, bars, fmt="{:.1f}", suffix="%")
    ax.set_ylabel("Reduction")
    ax.set_ylim(0, 110)
    ax.set_title("Temporal Folding cuts cost", fontsize=11, color=SUB)
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.07546_exp.svg")


# =============================================================================
# 5) Streaming probe (2606.10487v1)
# =============================================================================

def probe_method():
    fig, ax = _canvas(
        "Streaming Moderation via Hidden-State Probes",
        "Decode loop -> hidden state -> tiny probe -> early stop / rewrite",
    )
    y, h, w = 0.40, 0.30, 0.19
    xs = [0.012, 0.235, 0.458, 0.681]
    _box(ax, xs[0], y, w, h, "cyan", "LLM decode",
         ["token-by-token", "generation"], title_size=10.5)
    _box(ax, xs[1], y, w, h, "blue", "Hidden state",
         ["mid-layer activations", "already contain", "safety signals"], title_size=10.2)
    _box(ax, xs[2], y, w, h, "violet", "Tiny probe",
         ["linear / MLP", "~MB params"], title_size=10.5)
    _box(ax, xs[3], y, w, h, "amber", "Action",
         ["stop early", "mask/rewrite", "log & review"], title_size=10.5)
    for i in range(3):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)
    ax.text(0.5, 0.06, "Key idea: moderation moves inside generation; cost is tiny.",
            ha="center", va="bottom", fontsize=9, color=MUTED, style="italic")
    _save(fig, "2606.10487v1.svg")


def probe_exp():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    fig.suptitle(
        "Streaming probe — accuracy & cost",
        fontsize=13.5,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=1.005,
    )

    ax = axes[0]
    labels = ["F1 (ZHUYI)"]
    vals = [0.9746]
    bars = ax.bar(labels, vals, color=["#10b981"], width=0.5, zorder=3)
    _bar_labels(ax, bars, fmt="{:.4f}")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Strong detection", fontsize=11, color=SUB)
    _style_axes(ax)

    ax = axes[1]
    labels = ["Latency", "Compute"]
    vals = [0.1, 0.6]
    bars = ax.bar(labels, vals, color=["#0891b2", "#2563eb"], width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="{:.1f}")
    ax.set_ylabel("ms (latency) / GFLOPs (compute)")
    ax.set_ylim(0, 1.0)
    ax.set_title("~0.1ms probe, ~0.6 GFLOPs", fontsize=11, color=SUB)
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.10487v1_exp.svg")


# =============================================================================
# 6) MemeReason (2606.15307)
# =============================================================================

def meme_method():
    fig, ax = _canvas(
        "MemeReason — GRPO + CoT for Explainable Meme Moderation",
        "Image+text meme -> MLLM -> SFT warmup -> GRPO post-training -> label + rationale",
    )
    y, h, w = 0.40, 0.30, 0.17
    xs = [0.012, 0.205, 0.398, 0.591, 0.784]
    _box(ax, xs[0], y, w, h, "cyan", "Meme",
         ["image + text", "implicit harm"], title_size=10.5)
    _box(ax, xs[1], y, w, h, "blue", "MLLM",
         ["reasoning", "multimodal"], title_size=10.5)
    _box(ax, xs[2], y, w, h, "violet", "SFT",
         ["CoT distill", "fine-grained labels"], title_size=10.2)
    _box(ax, xs[3], y, w, h, "amber", "GRPO",
         ["reward: correct", "+ rationale", "+ format"], title_size=10.2)
    _box(ax, xs[4], y, 0.19, h, "green", "Output",
         ["hate/propaganda", "+ explanation"], title_size=10.5)
    for i in range(4):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)
    ax.text(0.5, 0.06, "Key idea: optimize both classification and explanation with RL.",
            ha="center", va="bottom", fontsize=9, color=MUTED, style="italic")
    _save(fig, "2606.15307.svg")


def meme_exp():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    fig.suptitle(
        "MemeReason — improvements on hate/propaganda meme benchmarks",
        fontsize=13.2,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=1.005,
    )

    ax = axes[0]
    labels = ["FHM\n(acc)", "ArMeme\n(Macro-F1)"]
    base = [79.9, 0.536]
    ours = [82.0, 0.612]
    x = range(len(labels))
    w = 0.35
    b1 = ax.bar([i - w / 2 for i in x], base, width=w, color="#94a3b8", zorder=3, label="baseline")
    b2 = ax.bar([i + w / 2 for i in x], ours, width=w, color="#10b981", zorder=3, label="GRPO+CoT")
    _bar_labels(ax, b1, fmt="{:.1f}")
    _bar_labels(ax, b2, fmt="{:.1f}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_title("Better detection", fontsize=11, color=SUB)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style_axes(ax)

    ax = axes[1]
    labels = ["METEOR (rationale)"]
    vals = [0.52]
    bars = ax.bar(labels, vals, color=["#2563eb"], width=0.5, zorder=3)
    _bar_labels(ax, bars, fmt="{:.2f}")
    ax.set_ylabel("METEOR")
    ax.set_ylim(0, 1.0)
    ax.set_title("Explanation quality", fontsize=11, color=SUB)
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.15307_exp.svg")


FIGS = [
    univid_method,
    univid_exp,
    vcg_method,
    vcg_exp,
    saerec_method,
    saerec_exp,
    semanticid_method,
    semanticid_exp,
    probe_method,
    probe_exp,
    meme_method,
    meme_exp,
]


def main() -> None:
    for fn in FIGS:
        fn()
    print(f"\nDone: {len(FIGS)} figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()

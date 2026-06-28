#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate light, research-style SVG figures for the 2026-06-28 inspection.

For each paper we emit two SVGs in ``assets/figures``:
  * ``{id}.svg``      -- method schematic
  * ``{id}_exp.svg``  -- compact highlight chart / callouts

Run:
  python gen_figures_20260628.py
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
# 2606.23911 — Scaling Dense Retrieval (EmbeddingModelV3)
# =============================================================================


def dr_curriculum_method():
    fig, ax = _canvas(
        "Scaling Dense Retrieval — structured mining + curriculum",
        "Multi-channel disagreement -> LLM cascade labels -> BCE/MNR/Triplet -> deployable dual-encoder",
    )

    y, h, w = 0.40, 0.30, 0.18
    xs = [0.012, 0.215, 0.418, 0.621, 0.824]

    _box(ax, xs[0], y, w, h, "cyan", "Retrievers", ["lexical", "BM25/ANN", "prod logs"])
    _box(ax, xs[1], y, w, h, "blue", "Mining", ["disagreement", "easy/hard", "negatives"])
    _box(ax, xs[2], y, w, h, "violet", "LLM Cascade", ["3-stage", "calibration", "5-grade"])
    _box(ax, xs[3], y, w, h, "amber", "Curriculum", ["BCE", "MNR", "Triplet"])
    _box(ax, xs[4], y, w, h, "green", "Deploy", ["dual-encoder", "ANN index", "online"])

    for i in range(4):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    _box(ax, 0.34, 0.15, 0.32, 0.16, "slate", "Scale", ["240M+ pairs", "4M queries", "tail gains"])
    _arrow(ax, 0.50, 0.40, 0.50, 0.31, dashed=True, color="#334155")

    ax.text(
        0.5,
        0.06,
        "Key idea: turn production retrieval disagreements into structured supervision, then train progressively.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
        style="italic",
    )

    _save(fig, "2606.23911v1.svg")


def dr_curriculum_exp():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    fig.suptitle(
        "Scaling Dense Retrieval — highlights",
        fontsize=13.5,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=1.005,
    )

    ax = axes[0]
    labels = ["NDCG@10\n(base)", "NDCG@10\n(ours)"]
    vals = [0.878, 0.923]
    bars = ax.bar(labels, vals, color=["#94a3b8", "#2563eb"], width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="{:.3f}")
    ax.set_ylim(0.80, 1.0)
    ax.set_title("Offline relevance", fontsize=11, color=SUB)
    _style_axes(ax)

    ax = axes[1]
    labels = ["Embarrassing\nretrievals", "After"]
    vals = [8.7, 3.5]
    bars = ax.bar(labels, vals, color=["#94a3b8", "#d97706"], width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="{:.1f}", suffix="%")
    ax.set_ylim(0, 10)
    ax.set_title("Bad-case reduction", fontsize=11, color=SUB)
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.23911v1_exp.svg")


# =============================================================================
# 2606.15330 — OneBar
# =============================================================================


def onebar_method():
    fig, ax = _canvas(
        "OneBar — end-to-end generative query recommendation",
        "Video+user context -> evidence schema -> prompt compression -> BART generator -> preference internalization",
    )

    y, h, w = 0.40, 0.30, 0.18
    xs = [0.012, 0.215, 0.418, 0.621, 0.824]

    _box(ax, xs[0], y, w, h, "cyan", "Inputs", ["video", "product meta", "user history"])
    _box(ax, xs[1], y, w, h, "blue", "Grounding", ["multi-source", "intent schema", "clean evidence"])
    _box(ax, xs[2], y, w, h, "violet", "Compression", ["prompt shrink", "latency", "stable"])
    _box(ax, xs[3], y, w, h, "amber", "Generator", ["BART-like", "query list", "end-to-end"])
    _box(ax, xs[4], y, w, h, "green", "PIOPD", ["on-policy", "preferences", "no reward model"])

    for i in range(4):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    ax.text(
        0.5,
        0.06,
        "Key idea: replace multi-stage cascades with a single generator under strict latency budgets.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
        style="italic",
    )

    _save(fig, "2606.15330v1.svg")


def onebar_exp():
    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    fig.suptitle(
        "OneBar — online A/B uplift",
        fontsize=13.5,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=1.005,
    )

    labels = ["Query\nExposure", "Query\nClick", "Orders", "GMV"]
    vals = [16.91, 18.68, 20.36, 21.67]
    bars = ax.bar(labels, vals, color=["#2563eb", "#0891b2", "#10b981", "#d97706"], width=0.60, zorder=3)
    _bar_labels(ax, bars, fmt="{:.2f}", suffix="%")
    ax.set_ylabel("Uplift")
    ax.set_ylim(0, 28)
    ax.set_title("All key business metrics move together", fontsize=11, color=SUB)
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.15330v1_exp.svg")


# =============================================================================
# 2606.19627 — VCG
# =============================================================================


def vcg_method():
    fig, ax = _canvas(
        "VCG — multimodal retrieval under extreme cold-start",
        "Domain-adapted CLIP embeds video frames; two-tower retrieval serves new videos without interactions",
    )

    y, h, w = 0.40, 0.30, 0.19
    xs = [0.012, 0.235, 0.458, 0.681]

    _box(ax, xs[0], y, w, h, "cyan", "Cold-start", ["new videos", "no clicks", "biased signals"])
    _box(ax, xs[1], y, w, h, "blue", "Video CLIP", ["uniform frames", "mean pooling", "domain-tuned"])
    _box(ax, xs[2], y, w, h, "violet", "Two-tower", ["user tower", "video tower", "ANN"])
    _box(ax, xs[3], y, w, h, "green", "Serve", ["zero-shot", "retrieve", "rank"])

    for i in range(3):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    _box(ax, 0.70, 0.15, 0.28, 0.16, "amber", "Eval", ["LVLM-as-judge", "offline proxy", "diagnose"])
    _arrow(ax, 0.78, 0.40, 0.84, 0.31, dashed=True, color="#d97706")

    ax.text(
        0.5,
        0.06,
        "Key idea: semantic-first multimodal retrieval to bootstrap a new video feed.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
        style="italic",
    )

    _save(fig, "2606.19627v1.svg")


def vcg_exp():
    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    fig.suptitle(
        "VCG — online cold-start gains",
        fontsize=13.5,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=1.005,
    )

    labels = ["Deep video\ncompletion"]
    vals = [50.0]
    bars = ax.bar(labels, vals, color=["#2563eb"], width=0.45, zorder=3)
    _bar_labels(ax, bars, fmt="{:.0f}", suffix="%")
    ax.set_ylim(0, 70)
    ax.set_ylabel("Uplift")
    ax.set_title("Extreme cold-start: strong watch-time proxy improvement", fontsize=11, color=SUB)
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.19627v1_exp.svg")


# =============================================================================
# 2606.23889 — INSPIRE
# =============================================================================


def inspire_method():
    fig, ax = _canvas(
        "INSPIRE — intent-aware sponsored product retrieval",
        "LLM teachers label structured intents; distilled student predicts intents for dense retrieval",
    )

    y, h, w = 0.40, 0.30, 0.18
    xs = [0.012, 0.215, 0.418, 0.621, 0.824]

    _box(ax, xs[0], y, w, h, "cyan", "Inputs", ["query", "product", "catalog"])
    _box(ax, xs[1], y, w, h, "blue", "Teacher LLMs", ["intent attrs", "weak labels", "multi-teacher"])
    _box(ax, xs[2], y, w, h, "violet", "Consensus", ["filter noise", "calibrate", "audit"])
    _box(ax, xs[3], y, w, h, "amber", "Student", ["LoRA SFT", "vLLM serve", "fast"])
    _box(ax, xs[4], y, w, h, "green", "Bi-encoder", ["intent-aug", "dense retrieval", "ads"])

    for i in range(4):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    ax.text(
        0.5,
        0.06,
        "Key idea: make implicit constraints (diet/allergen) explicit and controllable for retrieval.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
        style="italic",
    )

    _save(fig, "2606.23889v1.svg")


def inspire_exp():
    fig, ax = _canvas(
        "INSPIRE — experiment notes",
        "The abstract does not disclose headline gains; the paper reports offline retrieval improvements and scalable serving.",
        figsize=(11.0, 4.1),
    )

    _box(
        ax,
        0.06,
        0.33,
        0.40,
        0.36,
        "blue",
        "What is measured",
        ["NDCG@K / Precision@K", "intent coverage", "constraint satisfaction"],
    )
    _box(
        ax,
        0.54,
        0.33,
        0.40,
        0.36,
        "amber",
        "What is deployable",
        ["LoRA student", "vLLM throughput", "intent auditability"],
    )
    _arrow(ax, 0.46, 0.51, 0.54, 0.51)

    ax.text(
        0.5,
        0.14,
        "Tip: for governance, track constraint violations (e.g., allergen/diet mismatches) as an explicit guardrail metric.",
        ha="center",
        va="center",
        fontsize=9.5,
        color=MUTED,
        style="italic",
    )

    _save(fig, "2606.23889v1_exp.svg")


# =============================================================================
# 2606.25871 — AutoRelAnnotator
# =============================================================================


def autorel_method():
    fig, ax = _canvas(
        "AutoRelAnnotator — calibrated model cascade for relevance labels",
        "Fine-tune models -> per-class calibration -> defer low-confidence to larger models",
    )

    y, h, w = 0.40, 0.30, 0.18
    xs = [0.012, 0.215, 0.418, 0.621, 0.824]

    _box(ax, xs[0], y, w, h, "cyan", "Pairs", ["query", "ad/product", "context"])
    _box(ax, xs[1], y, w, h, "blue", "Cross-Enc", ["fine-tune", "accurate", "fast"])
    _box(ax, xs[2], y, w, h, "violet", "Calibrate", ["classwise", "deferral", "reliability"])
    _box(ax, xs[3], y, w, h, "amber", "LLM(s)", ["2B -> 8B", "only if needed", "vote"])
    _box(ax, xs[4], y, w, h, "green", "Labels", ["150M+", "hours", "offline eval"])

    for i in range(4):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    ax.text(
        0.5,
        0.06,
        "Key idea: decouple accuracy (domain fine-tuning) from cost (calibrated cascading).",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
        style="italic",
    )

    _save(fig, "2606.25871v1.svg")


def autorel_exp():
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.3))
    fig.suptitle(
        "AutoRelAnnotator — production impact",
        fontsize=13.5,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=1.005,
    )

    ax = axes[0]
    labels = ["Before", "After"]
    vals = [120.0, 2.0]  # 5 days vs ~2 hours
    bars = ax.bar(labels, vals, color=["#94a3b8", "#2563eb"], width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="{:.0f}")
    ax.set_ylabel("Turnaround (hours)")
    ax.set_ylim(0, 130)
    ax.set_title("Time-to-label", fontsize=11, color=SUB)
    _style_axes(ax)

    ax = axes[1]
    labels = ["Compute", "Cascade"]
    vals = [1.0, 0.5]
    bars = ax.bar(labels, vals, color=["#94a3b8", "#d97706"], width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="{:.2f}")
    ax.set_ylabel("Relative cost")
    ax.set_ylim(0, 1.2)
    ax.set_title("Cost reduction", fontsize=11, color=SUB)
    _style_axes(ax)

    ax = axes[2]
    labels = ["+FT", "+Calib"]
    vals = [20.0, 0.6]
    bars = ax.bar(labels, vals, color=["#10b981", "#0891b2"], width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="{:.1f}")
    ax.set_ylabel("Δ score (points)")
    ax.set_ylim(0, 22)
    ax.set_title("Quality gains", fontsize=11, color=SUB)
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.25871v1_exp.svg")


# =============================================================================
# 2606.26787 — AIGP
# =============================================================================


def aigp_method():
    fig, ax = _canvas(
        "AIGP — LLM pricing with long-term value alignment",
        "LLM proposes actions; offline-RL value estimator builds DPO preferences; distilled model deploys",
    )

    y, h, w = 0.40, 0.30, 0.18
    xs = [0.012, 0.215, 0.418, 0.621, 0.824]

    _box(ax, xs[0], y, w, h, "cyan", "Context", ["demand", "inventory", "price rules"])
    _box(ax, xs[1], y, w, h, "blue", "LLM Policy", ["interpret", "propose price", "rationale"])
    _box(ax, xs[2], y, w, h, "violet", "LTVE", ["offline RL", "long-term", "reward"])
    _box(ax, xs[3], y, w, h, "amber", "DPO Align", ["preference pairs", "optimize", "stable"])
    _box(ax, xs[4], y, w, h, "green", "Deploy", ["SFT distill", "fast", "monitor"])

    for i in range(4):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    ax.text(
        0.5,
        0.06,
        "Key idea: turn long-term business goals into preferences, not just short-term rewards.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
        style="italic",
    )

    _save(fig, "2606.26787v1.svg")


def aigp_exp():
    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    fig.suptitle(
        "AIGP — online 14-day A/B uplift",
        fontsize=13.5,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=1.005,
    )

    labels = ["GMV", "ROI", "Milestone"]
    vals = [13.21, 7.59, 8.20]
    bars = ax.bar(labels, vals, color=["#2563eb", "#10b981", "#d97706"], width=0.58, zorder=3)
    _bar_labels(ax, bars, fmt="{:.2f}", suffix="%")
    ax.set_ylabel("Uplift")
    ax.set_ylim(0, 16)
    ax.set_title("Long-term alignment improves multiple KPIs", fontsize=11, color=SUB)
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.26787v1_exp.svg")


# =============================================================================
# 2606.27314 — ILE taxonomy
# =============================================================================


def ile_method():
    fig, ax = _canvas(
        "ILE Taxonomy — coded language detection via mechanisms",
        "Mechanism-oriented taxonomy -> prompt scaffolding -> doc/span detection + mechanism labels",
    )

    y, h, w = 0.40, 0.30, 0.19
    xs = [0.012, 0.235, 0.458, 0.681]

    _box(ax, xs[0], y, w, h, "cyan", "Posts", ["TikTok", "Bluesky", "concept drift"])
    _box(ax, xs[1], y, w, h, "blue", "Taxonomy", ["11 classes", "33 mechanisms", "auditable"])
    _box(ax, xs[2], y, w, h, "violet", "LLM Prompt", ["scaffold", "span locate", "classify"])
    _box(ax, xs[3], y, w, h, "green", "Govern", ["detect", "explain", "update"])

    for i in range(3):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    ax.text(
        0.5,
        0.06,
        "Key idea: model *mechanisms* (not keywords) to handle coded-language drift.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
        style="italic",
    )

    _save(fig, "2606.27314v1.svg")


def ile_exp():
    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    fig.suptitle(
        "ILE Taxonomy — improvements over baselines",
        fontsize=13.5,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=1.005,
    )

    labels = ["Accuracy\n(+)", "Span F1\n(+)"]
    vals = [4.7, 5.4]
    bars = ax.bar(labels, vals, color=["#2563eb", "#10b981"], width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="{:.1f}", suffix="%")
    ax.set_ylabel("Absolute gain")
    ax.set_ylim(0, 7.0)
    ax.set_title("Taxonomy scaffolding helps LLM moderation prompts", fontsize=11, color=SUB)
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.27314v1_exp.svg")


# =============================================================================
# 2606.27187 — HarmVideoBench
# =============================================================================


def harmvideo_method():
    fig, ax = _canvas(
        "HarmVideoBench — benchmarking harmful video understanding",
        "3-level reasoning (evidence/meaning/beyond) + BCR boundary prediction with retrieval",
    )

    y, h, w = 0.40, 0.30, 0.18
    xs = [0.012, 0.215, 0.418, 0.621, 0.824]

    _box(ax, xs[0], y, w, h, "cyan", "Videos", ["harmful", "contextual", "multimodal"])
    _box(ax, xs[1], y, w, h, "blue", "3 Layers", ["evidence", "internal", "beyond"])
    _box(ax, xs[2], y, w, h, "violet", "MCQs", ["4,137", "diagnostic", "rationales"])
    _box(ax, xs[3], y, w, h, "amber", "BCR", ["boundary", "retrieve", "reason"])
    _box(ax, xs[4], y, w, h, "green", "Audit", ["model gaps", "policy", "iter"])

    for i in range(4):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    ax.text(
        0.5,
        0.06,
        "Key idea: moderation needs layered reasoning + evidence-based explanations.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
        style="italic",
    )

    _save(fig, "2606.27187v1.svg")


def harmvideo_exp():
    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    fig.suptitle(
        "HarmVideoBench — BCR gain",
        fontsize=13.5,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=1.005,
    )

    labels = ["Macro avg\n(base)", "Macro avg\n(BCR)"]
    vals = [61.7, 84.4]
    bars = ax.bar(labels, vals, color=["#94a3b8", "#2563eb"], width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="{:.1f}", suffix="%")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 100)
    ax.set_title("Better deep reasoning on harmful videos", fontsize=11, color=SUB)
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.27187v1_exp.svg")


# =============================================================================
# 2606.27274 — BetXplain
# =============================================================================


def betxplain_method():
    fig, ax = _canvas(
        "BetXplain — explanation-annotated manipulative ad detection",
        "Collect ads -> label + human explanations -> train explainable classifier -> audit strategies",
    )

    y, h, w = 0.40, 0.30, 0.19
    xs = [0.012, 0.235, 0.458, 0.681]

    _box(ax, xs[0], y, w, h, "cyan", "Ads", ["Instagram", "Reddit", "betting apps"])
    _box(ax, xs[1], y, w, h, "blue", "Labels+Explan.", ["manipulative", "deceptive", "responsible"])
    _box(ax, xs[2], y, w, h, "violet", "Models", ["detect", "generate expl.", "debug"])
    _box(ax, xs[3], y, w, h, "green", "Govern", ["audit", "policy", "mitigation"])

    for i in range(3):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    ax.text(
        0.5,
        0.06,
        "Key idea: explanations are supervision for auditability, not just classification.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
        style="italic",
    )

    _save(fig, "2606.27274v1.svg")


def betxplain_exp():
    fig, ax = _canvas(
        "BetXplain — dataset structure",
        "Counts are not disclosed in the anonymized version; below shows the supervision schema.",
        figsize=(11.0, 4.1),
    )

    _box(ax, 0.08, 0.33, 0.26, 0.36, "rose", "Deceptive", ["false claims", "misleading odds", "bait"])
    _box(ax, 0.37, 0.33, 0.26, 0.36, "amber", "Manipulative", ["pressure", "FOMO", "dark patterns"])
    _box(ax, 0.66, 0.33, 0.26, 0.36, "green", "Responsible", ["balanced", "warning", "no coercion"])

    ax.text(0.5, 0.18, "Each sample includes a human-written explanation for why the label applies.", ha="center", va="center", fontsize=9.8, color=SUB)

    _save(fig, "2606.27274v1_exp.svg")


def main():
    dr_curriculum_method()
    dr_curriculum_exp()

    onebar_method()
    onebar_exp()

    vcg_method()
    vcg_exp()

    inspire_method()
    inspire_exp()

    autorel_method()
    autorel_exp()

    aigp_method()
    aigp_exp()

    ile_method()
    ile_exp()

    harmvideo_method()
    harmvideo_exp()

    betxplain_method()
    betxplain_exp()

    print(f"wrote figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()

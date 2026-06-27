#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate light, research-style SVG figures for the 2026-06-27 inspection.

For each paper we emit two SVGs in ``assets/figures``:
  * ``{id}.svg``      -- method schematic
  * ``{id}_exp.svg``  -- compact highlight chart

Run:
  python gen_figures_20260627.py
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
# 2605.05991 — Case-driven multi-agent relevance
# =============================================================================


def case_agent_method():
    fig, ax = _canvas(
        "Case-Driven Multi-Agent — E-commerce Search Relevance",
        "Bad cases -> standards+annotation -> diagnosis/repair -> eval/deploy (with Memory + DeepSearch)",
    )
    y, h, w = 0.40, 0.30, 0.18
    xs = [0.012, 0.215, 0.418, 0.621, 0.824]

    _box(ax, xs[0], y, w, h, "cyan", "User Agent", ["mine bad cases", "conversation", "feedback"])
    _box(ax, xs[1], y, w, h, "blue", "Annotator", ["multi-turn", "standards", "labeling"])
    _box(ax, xs[2], y, w, h, "violet", "Optimizer", ["root cause", "data/model", "repair"])
    _box(ax, xs[3], y, w, h, "amber", "Harness", ["All-in-One", "instruction", "ops bot"])
    _box(ax, xs[4], y, w, h, "green", "Deploy", ["human eval", "online", "iterate"])

    for i in range(4):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    _box(ax, 0.35, 0.15, 0.30, 0.16, "slate", "Global Memory", ["resolved cases", "standard updates"])
    _arrow(ax, 0.50, 0.40, 0.50, 0.31, dashed=True, color="#334155")
    ax.text(
        0.5,
        0.06,
        "Key idea: treat relevance iteration as a closed-loop multi-role system, automated by agents.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
        style="italic",
    )

    _save(fig, "2605.05991.svg")


def case_agent_exp():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    fig.suptitle(
        "Case-Driven Multi-Agent — highlight numbers",
        fontsize=13.5,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=1.005,
    )

    ax = axes[0]
    labels = ["LLM\nAnnotator", "GRM\nAnnotator", "Optimizer", "Auto\nPipeline"]
    vals = [12.38, 9.78, 7.28, 10.17]
    bars = ax.bar(labels, vals, color=["#2563eb", "#0891b2", "#7c3aed", "#10b981"], width=0.58, zorder=3)
    _bar_labels(ax, bars, fmt="{:.2f}", suffix="%")
    ax.set_ylabel("SBS win-rate gain")
    ax.set_ylim(0, 15)
    ax.set_title("Human SBS improvements", fontsize=11, color=SUB)
    _style_axes(ax)

    ax = axes[1]
    labels = ["Instr. ACC\n(before)", "Instr. ACC\n(after)"]
    vals = [0.5033, 0.8402]
    bars = ax.bar(labels, vals, color=["#94a3b8", "#d97706"], width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="{:.4f}")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Instruction-following relevance", fontsize=11, color=SUB)
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2605.05991_exp.svg")


# =============================================================================
# 2606.17449 — MODE-RAG
# =============================================================================


def mode_rag_method():
    fig, ax = _canvas(
        "MODE-RAG — Risk-aware multi-agent M-RAG",
        "VFE+attention router gates interventions; stage agents + MCTS for auditable reasoning",
    )

    y, h, w = 0.40, 0.30, 0.18
    xs = [0.012, 0.215, 0.418, 0.621, 0.824]

    _box(ax, xs[0], y, w, h, "cyan", "Query", ["image+text", "retrieval need", "high-risk?"])
    _box(ax, xs[1], y, w, h, "blue", "FE-Router", ["VFE energy", "attention", "gate"])
    _box(ax, xs[2], y, w, h, "violet", "Stage Agents", ["Per/Ret/Rea", "+Gen", "intervene"])
    _box(ax, xs[3], y, w, h, "amber", "MCTS", ["causal DAG", "search", "verify"])
    _box(ax, xs[4], y, w, h, "green", "Output", ["fewer halluc.", "stable format", "overseer"])

    for i in range(4):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    ax.text(
        0.5,
        0.06,
        "Key idea: dynamic intervention avoids the 'intervention paradox' of static RAG pipelines.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
        style="italic",
    )

    _save(fig, "2606.17449.svg")


def mode_rag_exp():
    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    fig.suptitle(
        "MODE-RAG — ModeVent improvements (Total)",
        fontsize=13.5,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=1.005,
    )

    labels = ["Majority\nText Bias", "Info\nSparsity", "Attribute\nHijacking"]
    before = [5.41, 3.93, 2.74]
    after = [6.83, 5.22, 3.82]

    x = range(len(labels))
    w = 0.34
    ax.bar([i - w / 2 for i in x], before, width=w, color="#94a3b8", label="baseline", zorder=3)
    bars = ax.bar([i + w / 2 for i in x], after, width=w, color="#2563eb", label="MODE-RAG", zorder=3)
    _bar_labels(ax, bars, fmt="{:.2f}")

    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Total score")
    ax.set_ylim(0, 8)
    ax.legend(frameon=False)
    ax.set_title("Higher robustness on challenging hallucination subsets", fontsize=11, color=SUB)
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.17449_exp.svg")


# =============================================================================
# 2606.27330 — PEEU
# =============================================================================


def peeu_method():
    fig, ax = _canvas(
        "PEEU — Autonomous exploration + hindsight for GUI planning",
        "Explore -> collect experience -> hindsight rewrite -> high-level planning supervision",
    )

    y, h, w = 0.40, 0.30, 0.19
    xs = [0.012, 0.235, 0.458, 0.681]

    _box(ax, xs[0], y, w, h, "cyan", "Web env", ["multiple sites", "GUI states", "tools"])
    _box(ax, xs[1], y, w, h, "blue", "Explore", ["build tree", "collect traj", "experience"])
    _box(ax, xs[2], y, w, h, "violet", "Hindsight", ["rewrite tasks", "align steps", "high-level"])
    _box(ax, xs[3], y, w, h, "green", "Train", ["SFT/GRPO", "better planning", "OOD"])

    for i in range(3):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    _box(ax, 0.73, 0.15, 0.25, 0.16, "amber", "TDHAF", ["low/mid/high", "generalization"])
    _arrow(ax, 0.80, 0.40, 0.84, 0.31, dashed=True, color="#d97706")

    ax.text(
        0.5,
        0.06,
        "Key idea: scale planning data automatically via exploration + hindsight distillation.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
        style="italic",
    )

    _save(fig, "2606.27330.svg")


def peeu_exp():
    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    fig.suptitle(
        "PEEU — accuracy on real-world GUI benchmarks",
        fontsize=13.5,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=1.005,
    )

    labels = ["Atomic-SFT", "Qwen2.5-VL\n32B", "PEEU-SFT\n(7B)"]
    vals = [21.7, 22.7, 30.6]

    bars = ax.bar(labels, vals, color=["#94a3b8", "#7c3aed", "#2563eb"], width=0.58, zorder=3)
    _bar_labels(ax, bars, fmt="{:.1f}", suffix="%")

    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 40)
    ax.set_title("Small model beats a much larger baseline", fontsize=11, color=SUB)
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.27330_exp.svg")


# =============================================================================
# 2606.23543 — VeriEvol
# =============================================================================


def verievol_method():
    fig, ax = _canvas(
        "VeriEvol — Verifiable Evol-Instruct data scaling",
        "Difficulty evolution + answer verification -> audited data -> GRPO/RL",
    )

    y, h, w = 0.40, 0.30, 0.18
    xs = [0.012, 0.215, 0.418, 0.621, 0.824]

    _box(ax, xs[0], y, w, h, "cyan", "Seeds", ["easy VQA", "math", "images"])
    _box(ax, xs[1], y, w, h, "blue", "Evolution", ["route ops", "harder", "grounded"])
    _box(ax, xs[2], y, w, h, "violet", "HTV-Agent", ["counter-evidence", "hypothesis test", "trace"])
    _box(ax, xs[3], y, w, h, "amber", "Verified", ["clean labels", "auditable", "scalable"])
    _box(ax, xs[4], y, w, h, "green", "GRPO", ["SFT init", "RL", "improve"])

    for i in range(4):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    ax.text(
        0.5,
        0.06,
        "Key idea: scaling needs reliable labels; verification is first-class.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
        style="italic",
    )

    _save(fig, "2606.23543.svg")


def verievol_exp():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    fig.suptitle(
        "VeriEvol — scaling & incremental gains",
        fontsize=13.5,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=1.005,
    )

    ax = axes[0]
    labels = ["SFT 10K", "SFT 250K"]
    vals = [35.42, 54.73]
    bars = ax.bar(labels, vals, color=["#94a3b8", "#2563eb"], width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="{:.2f}")
    ax.set_ylabel("Mean accuracy")
    ax.set_ylim(0, 70)
    ax.set_title("More evolved SFT data helps", fontsize=11, color=SUB)
    _style_axes(ax)

    ax = axes[1]
    labels = ["RL baseline", "VeriEvol"]
    vals = [0.0, 3.88]
    bars = ax.bar(labels, vals, color=["#94a3b8", "#d97706"], width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="{:.2f}")
    ax.set_ylabel("Δ accuracy")
    ax.set_ylim(0, 5)
    ax.set_title("+3.88 over un-evolved RL", fontsize=11, color=SUB)
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.23543_exp.svg")


# =============================================================================
# 2606.27377 — DanceOPD
# =============================================================================


def danceopd_method():
    fig, ax = _canvas(
        "DanceOPD — On-policy generative field distillation",
        "Capabilities = velocity fields; hard routing + student rollout query; compose T2I/edit/CFG",
    )

    y, h, w = 0.40, 0.30, 0.18
    xs = [0.012, 0.215, 0.418, 0.621, 0.824]

    _box(ax, xs[0], y, w, h, "cyan", "Tasks", ["T2I", "local edit", "global edit"])
    _box(ax, xs[1], y, w, h, "blue", "Fields", ["velocity", "per-capability", "teachers"])
    _box(ax, xs[2], y, w, h, "violet", "Hard route", ["per-sample", "choose field", "simple loss"])
    _box(ax, xs[3], y, w, h, "amber", "On-policy", ["student rollout", "low-noise query", "v-MSE"])
    _box(ax, xs[4], y, w, h, "green", "Student", ["composed", "capabilities", "preserve T2I"])

    for i in range(4):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    ax.text(
        0.5,
        0.06,
        "Key idea: distill composable fields on-policy to reduce student–teacher gap.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
        style="italic",
    )

    _save(fig, "2606.27377.svg")


def danceopd_exp():
    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    fig.suptitle(
        "DanceOPD — GEditBench (Avg)",
        fontsize=13.5,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=1.005,
    )

    labels = ["T2I + Edit", "Local + Global"]
    vals = [5.347, 5.498]
    bars = ax.bar(labels, vals, color=["#2563eb", "#10b981"], width=0.58, zorder=3)
    _bar_labels(ax, bars, fmt="{:.3f}")

    ax.set_ylabel("GEditBench Avg")
    ax.set_ylim(0, 6.2)
    ax.set_title("Strong multi-capability composition", fontsize=11, color=SUB)
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.27377_exp.svg")


# =============================================================================
# 2606.26907 — Qwen-Image-Agent
# =============================================================================


def qwen_image_agent_method():
    fig, ax = _canvas(
        "Qwen-Image-Agent — Context-centric agentic image generation",
        "Context-aware planning -> grounding (reason/search/memory/feedback) -> generation",
    )

    y, h, w = 0.40, 0.30, 0.18
    xs = [0.012, 0.215, 0.418, 0.621, 0.824]

    _box(ax, xs[0], y, w, h, "cyan", "User context", ["underspecified", "implicit", "needs info"])
    _box(ax, xs[1], y, w, h, "blue", "Planning", ["find gaps", "route tools", "spec"])
    _box(ax, xs[2], y, w, h, "violet", "Grounding", ["search", "memory", "feedback"])
    _box(ax, xs[3], y, w, h, "amber", "Gen context", ["assembled", "detailed", "alloc"])
    _box(ax, xs[4], y, w, h, "green", "Generator", ["T2I/edit", "images", "iterate"])

    for i in range(4):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    ax.text(
        0.5,
        0.06,
        "Key idea: close the context gap by acquiring missing context before generation.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=MUTED,
        style="italic",
    )

    _save(fig, "2606.26907.svg")


def qwen_image_agent_exp():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    fig.suptitle(
        "Qwen-Image-Agent — IA-Bench & MindBench",
        fontsize=13.5,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=1.005,
    )

    ax = axes[0]
    labels = ["Direct\n(Qwen-Image-2.0)", "Agent"]
    vals = [17.4, 45.4]
    bars = ax.bar(labels, vals, color=["#94a3b8", "#2563eb"], width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="{:.1f}")
    ax.set_ylabel("IA-score")
    ax.set_ylim(0, 55)
    ax.set_title("IA-Bench overall", fontsize=11, color=SUB)
    _style_axes(ax)

    ax = axes[1]
    labels = ["MindBench\nboost"]
    vals = [82.6]
    bars = ax.bar(labels, vals, color=["#d97706"], width=0.45, zorder=3)
    _bar_labels(ax, bars, fmt="{:.1f}", suffix="%")
    ax.set_ylabel("Relative improvement")
    ax.set_ylim(0, 100)
    ax.set_title("Over direct generation baseline", fontsize=11, color=SUB)
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.26907_exp.svg")


def main() -> None:
    case_agent_method()
    case_agent_exp()

    mode_rag_method()
    mode_rag_exp()

    peeu_method()
    peeu_exp()

    verievol_method()
    verievol_exp()

    danceopd_method()
    danceopd_exp()

    qwen_image_agent_method()
    qwen_image_agent_exp()


if __name__ == "__main__":
    main()

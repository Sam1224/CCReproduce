#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate light, research-style SVG figures for the 2026-06-21 paper inspection.

For each of today's 6 papers we emit two SVGs in ``assets/figures``:
  * ``{id}.svg``      -- a clean schematic of the method/pipeline (methodology figure)
  * ``{id}_exp.svg``  -- a chart of the paper's standout experimental numbers

Style matches the light research palette used by gen_figures_20260619.py (labelled
rounded boxes + arrows for the schematic, matplotlib bar/line charts for results),
but everything is written as vector SVG.

Run:  python gen_figures_20260621.py
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
    "blue":   ("#eff6ff", "#bfdbfe", "#2563eb"),
    "cyan":   ("#ecfeff", "#a5f3fc", "#0891b2"),
    "teal":   ("#f0fdfa", "#99f6e4", "#0d9488"),
    "green":  ("#ecfdf5", "#bbf7d0", "#10b981"),
    "amber":  ("#fffbeb", "#fde68a", "#d97706"),
    "violet": ("#f5f3ff", "#ddd6fe", "#7c3aed"),
    "rose":   ("#fff1f2", "#fecdd3", "#e11d48"),
    "slate":  ("#f8fafc", "#e2e8f0", "#334155"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": EDGE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": SUB, "ytick.color": SUB, "axes.titlecolor": INK,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
})


# ============================ schematic helpers ==============================
def _box(ax, x, y, w, h, accent, title, lines=None, title_size=11, body_size=9.0):
    fill, stroke, tc = ACC[accent]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.4, edgecolor=stroke, facecolor=fill, zorder=2))
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
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="->,head_width=3.4,head_length=6",
        mutation_scale=1.0, linewidth=1.9, color=color or EDGE,
        linestyle=(0, (5, 4)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}", zorder=1))


def _canvas(title, subtitle, figsize=(11.0, 4.6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.012, 0.93), 0.10, 0.045, color="#2563eb",
                               transform=ax.transAxes, zorder=4))
    ax.text(0.012, 0.905, title, ha="left", va="top", fontsize=15.5,
            fontweight="bold", color=INK)
    ax.text(0.012, 0.852, subtitle, ha="left", va="top", fontsize=9.6, color=MUTED)
    return fig, ax


def _save(fig, name):
    out = FIG_DIR / name
    fig.savefig(out, format="svg", bbox_inches="tight", pad_inches=0.12)
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
                ha="center", va="bottom", fontsize=9.3, fontweight="bold", color=INK)


# =============================================================================
# 1) TimeProVe (2606.20561)
# =============================================================================
def timeprove_method():
    fig, ax = _canvas(
        "TimeProVe — Propose, then Verify for Long-Video LVQA",
        "Long ADL video + query  ->  ACE (cheap, full video once)  ->  rerank hypotheses  ->  VLM verifies only a few short clips",
    )
    y, h, w = 0.40, 0.30, 0.175
    xs = [0.012, 0.213, 0.414, 0.615, 0.816]
    _box(ax, xs[0], y, w, h, "cyan", "Input",
         ["hour-long ADL video", "+ free-form query", "(answer in a few secs)"], title_size=10.5)
    _box(ax, xs[1], y, w, h, "blue", "Action Detector",
         ["one pass over video", "P = f_act(v)", "-> action timeline A"], title_size=10.5)
    _box(ax, xs[2], y, w, h, "violet", "Proposal Generator",
         ["atomic + merged windows", "(answer, evidence) pairs", "edge-LLM, cheap"], title_size=10.5)
    _box(ax, xs[3], y, w, h, "amber", "Rerank R(w|q)",
         ["R_tmp + R_sem", "+ R_cov - R_len", "top-K hypotheses"], title_size=10.5)
    _box(ax, xs[4], y, w, h, "green", "Temporal Verifier",
         ["cloud VLM on few clips", "verify -> answer", "+ visual evidence"], title_size=10.5)
    for i in range(4):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)
    _box(ax, 0.213, 0.12, 0.577, 0.15, "slate", "ACE = Action-based Candidate Evidence (lightweight, local / edge)",
         ["reasoning cost is decoupled from raw video length L — VLM is invoked only on selected short RGB windows"],
         title_size=10, body_size=8.6)
    _arrow(ax, 0.50, 0.27, 0.50, y - 0.005, color="#0d9488", dashed=True)
    ax.text(0.5, 0.045, "Dense 'watch-it-all' VLM cost grows with duration; TimeProVe proposes cheap candidates, "
                        "verifies only the promising clips.", ha="center", va="bottom",
            fontsize=9, color=MUTED, style="italic")
    _save(fig, "2606.20561.svg")


def timeprove_exp():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    fig.suptitle("TimeProVe — standout results (OpenTSUBench accuracy & efficiency)",
                 fontsize=14, fontweight="bold", x=0.012, ha="left", y=1.005)
    ax = axes[0]
    models = ["Strongest\nbaseline", "TimeProVe\n(Ours)"]
    acc = [37.8, 45.1]
    bars = ax.bar(models, acc, color=["#94a3b8", "#2563eb"], width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="{:.1f}")
    ax.set_ylabel("OpenTSUBench overall accuracy (%)")
    ax.set_ylim(0, 60)
    ax.set_title("+7.3% over the strongest baseline", fontsize=11, color=SUB)
    ax.annotate("+7.3%", xy=(1, 45.1), xytext=(1, 53), ha="center",
                fontsize=11, fontweight="bold", color="#2563eb")
    _style_axes(ax)

    ax = axes[1]
    labels = ["VLM calls", "Inference cost"]
    red = [75, 93]
    bars = ax.bar(labels, red, color=["#0891b2", "#10b981"], width=0.5, zorder=3)
    _bar_labels(ax, bars, fmt="{:.0f}", suffix="%")
    ax.set_ylabel("Reduction vs dense baseline (%)")
    ax.set_ylim(0, 110)
    ax.set_title("-75% VLM calls  ·  -93% cost", fontsize=11, color=SUB)
    _style_axes(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.20561_exp.svg")


# =============================================================================
# 2) NEST (2606.19706)
# =============================================================================
def nest_method():
    fig, ax = _canvas(
        "NEST — Narrative Event Structures in Time",
        "1005 full movies  ->  ~102 multimodal events/film (visual + dialogue + audio)  ->  linked by relations  ->  4 tasks",
    )
    _box(ax, 0.015, 0.40, 0.20, 0.30, "cyan", "Full-length movies",
         ["1005 films (avg ~98 min)", "1639.3 total hours", "untrimmed, complete"], title_size=10.5)
    _box(ax, 0.245, 0.40, 0.235, 0.30, "blue", "Multimodal events",
         ["~102 events / film", "grounded in visual,", "dialogue & audio"], title_size=10.5)
    _box(ax, 0.51, 0.40, 0.235, 0.30, "violet", "Narrative relations",
         ["temporal ordering", "hierarchical composition", "long-range dependency"], title_size=10.5)
    _arrow(ax, 0.215, 0.55, 0.245, 0.55)
    _arrow(ax, 0.48, 0.55, 0.51, 0.55)
    tasks = [("ETD", "trigger detect"), ("EL", "event localize"),
             ("EAE", "argument extract"), ("ERE", "relation extract")]
    bx, bw = 0.015, 0.175
    for i, (t, d) in enumerate(tasks):
        x = bx + i * 0.193
        _box(ax, x, 0.10, bw, 0.18, "amber", t, [d], title_size=11, body_size=8.6)
        _arrow(ax, 0.62, 0.40, x + bw / 2, 0.285, color="#d97706", dashed=True)
    _box(ax, 0.77, 0.40, 0.215, 0.30, "green", "Beyond multiple-choice",
         ["link early setups to", "later payoffs across", "long time gaps"], title_size=10.2)
    ax.text(0.5, 0.035, "Moves long-video understanding from needle-in-a-haystack retrieval to narrative event-structure reasoning.",
            ha="center", va="bottom", fontsize=9, color=MUTED, style="italic")
    _save(fig, "2606.19706.svg")


def nest_exp():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    fig.suptitle("NEST — benchmark difficulty (grounded discovery vs relation extraction)",
                 fontsize=14, fontweight="bold", x=0.012, ha="left", y=1.005)
    ax = axes[0]
    labels = ["ETD", "EL", "EAE"]
    vals = [8, 6, 11]
    bars = ax.bar(labels, vals, color=["#e11d48", "#e11d48", "#e11d48"], width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="<{:.0f}", suffix="%")
    ax.set_ylabel("Best score (%)")
    ax.set_ylim(0, 50)
    ax.set_title("Grounded event discovery is very hard", fontsize=11, color=SUB)
    _style_axes(ax)
    ax = axes[1]
    labels = ["ERE\nzero-shot", "ERE\nfine-tuned"]
    vals = [35.45, 44.42]
    bars = ax.bar(labels, vals, color=["#94a3b8", "#10b981"], width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="{:.2f}")
    ax.set_ylabel("Event Relation Extraction F1 (%)")
    ax.set_ylim(0, 60)
    ax.set_title("Relations more tractable: 35.45 -> 44.42 F1", fontsize=11, color=SUB)
    ax.annotate("+8.97", xy=(1, 44.42), xytext=(1, 52), ha="center",
                fontsize=10.5, fontweight="bold", color="#10b981")
    _style_axes(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.19706_exp.svg")


# =============================================================================
# 3) Connect the Dots / CoD (2606.20002)
# =============================================================================
def cod_method():
    fig, ax = _canvas(
        "Connect the Dots (CoD) — RL for long-lifecycle agents",
        "Deployed agent solves a long task sequence  ->  interleave solve-task & update-context  ->  end-to-end RL on the meta-capability",
    )
    _box(ax, 0.015, 0.42, 0.18, 0.30, "cyan", "Long task stream",
         ["agent deployed, solves", "many tasks in sequence", "explores environment"], title_size=10.2)
    _box(ax, 0.225, 0.55, 0.24, 0.18, "blue", "Solve-task episode",
         ["act in environment", "-> task reward"], title_size=10.2, body_size=8.8)
    _box(ax, 0.225, 0.30, 0.24, 0.18, "violet", "Update-context episode",
         ["learn from experience", "self-update env context"], title_size=10.2, body_size=8.8)
    _arrow(ax, 0.195, 0.57, 0.225, 0.64)
    _arrow(ax, 0.195, 0.50, 0.225, 0.40)
    _arrow(ax, 0.465, 0.39, 0.49, 0.55, color="#7c3aed", rad=0.3)
    ax.text(0.50, 0.50, "context\nimproves\nfuture tasks", ha="center", va="center",
            fontsize=8.2, color="#7c3aed", fontweight="bold")
    _box(ax, 0.55, 0.42, 0.20, 0.30, "amber", "End-to-end RL",
         ["long rollout over both", "episode types", "GRPO + fine-grained", "credit assignment"], title_size=10.2, body_size=8.6)
    _arrow(ax, 0.465, 0.57, 0.55, 0.57)
    _box(ax, 0.78, 0.42, 0.205, 0.30, "green", "OOD generalization",
         ["within domain", "cross-domain", "-> Ralph-loop setting"], title_size=10.2)
    _arrow(ax, 0.75, 0.57, 0.78, 0.57)
    ax.text(0.5, 0.06, "Experience-driven self-updating of context becomes the RL-optimizable meta-capability (not per-task skills).",
            ha="center", va="bottom", fontsize=9, color=MUTED, style="italic")
    _save(fig, "2606.20002.svg")


def cod_exp():
    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    fig.suptitle("CoD — proof-of-concept: experience-updated context drives OOD generalization",
                 fontsize=13.5, fontweight="bold", x=0.012, ha="left", y=1.005)
    cats = ["In-domain", "Cross-domain", "Ralph-loop\n(OOD)"]
    fixed = [54, 41, 33]
    cod = [78, 66, 58]
    x = range(len(cats))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], fixed, width=w, color="#94a3b8",
                label="Fixed context (no CoD)", zorder=3)
    b2 = ax.bar([i + w / 2 for i in x], cod, width=w, color="#7c3aed",
                label="CoD (self-updating)", zorder=3)
    _bar_labels(ax, b1, fmt="{:.0f}")
    _bar_labels(ax, b2, fmt="{:.0f}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats)
    ax.set_ylabel("Task success (%, illustrative)")
    ax.set_ylim(0, 100)
    ax.set_title("Gains hold within, across domains, and transferring to Ralph-loop", fontsize=11, color=SUB)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    _style_axes(ax)
    ax.text(0.5, -0.16, "Proof-of-concept numbers illustrate the reported trend (open-source: Trinity-RFT); paper reports no standard SOTA table.",
            transform=ax.transAxes, ha="center", fontsize=8.3, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    _save(fig, "2606.20002_exp.svg")


# =============================================================================
# 4) MATM (2606.19911)
# =============================================================================
def matm_method():
    fig, ax = _canvas(
        "MATM — Multi-Agent Transactive Memory",
        "Producer agents contribute trajectories  ->  shared repository (+ LTR rerank)  ->  consumer agents retrieve & reuse",
    )
    _box(ax, 0.015, 0.42, 0.20, 0.30, "blue", "Producer agents",
         ["solve tasks in", "ALFWorld / WebArena", "-> action-obs trajectories"], title_size=10.2)
    _box(ax, 0.255, 0.42, 0.22, 0.30, "slate", "Transactive memory",
         ["shared repository of", "agent-generated", "trajectories"], title_size=10.2)
    _box(ax, 0.255, 0.12, 0.22, 0.18, "amber", "LTR reranker",
         ["learning-to-rank", "improves retrieval"], title_size=10.2, body_size=8.8)
    _box(ax, 0.515, 0.42, 0.22, 0.30, "violet", "Retrieve",
         ["query repository for", "relevant procedural", "knowledge"], title_size=10.2)
    _box(ax, 0.775, 0.42, 0.21, 0.30, "green", "Consumer agents",
         ["reuse others' experience", "+ task success", "- interaction steps"], title_size=10.2)
    _arrow(ax, 0.215, 0.57, 0.255, 0.57)
    _arrow(ax, 0.475, 0.57, 0.515, 0.57)
    _arrow(ax, 0.735, 0.57, 0.775, 0.57)
    _arrow(ax, 0.365, 0.30, 0.365, 0.41, color="#d97706", dashed=True)
    ax.text(0.5, 0.045, "Extends RAG from human documents to agent trajectories — heterogeneous agents reuse each other's experience, no joint training.",
            ha="center", va="bottom", fontsize=9, color=MUTED, style="italic")
    _save(fig, "2606.19911.svg")


def matm_exp():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    fig.suptitle("MATM — retrieving shared trajectories improves success & cuts steps",
                 fontsize=13.5, fontweight="bold", x=0.012, ha="left", y=1.005)
    ax = axes[0]
    labels = ["No reuse", "Retrieval", "Retrieval\n+ LTR"]
    vals = [52, 64, 71]
    bars = ax.bar(labels, vals, color=["#94a3b8", "#0891b2", "#10b981"], width=0.6, zorder=3)
    _bar_labels(ax, bars, fmt="{:.0f}", suffix="%")
    ax.set_ylabel("Task success rate (%, illustrative)")
    ax.set_ylim(0, 90)
    ax.set_title("Success rises with reuse + reranking", fontsize=11, color=SUB)
    _style_axes(ax)
    ax = axes[1]
    labels = ["No reuse", "Retrieval", "Retrieval\n+ LTR"]
    steps = [18.0, 13.5, 11.2]
    bars = ax.bar(labels, steps, color=["#94a3b8", "#0891b2", "#10b981"], width=0.6, zorder=3)
    _bar_labels(ax, bars, fmt="{:.1f}")
    ax.set_ylabel("Avg interaction steps (lower = better)")
    ax.set_ylim(0, 24)
    ax.set_title("Fewer steps to complete a task", fontsize=11, color=SUB)
    _style_axes(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.19911_exp.svg")


# =============================================================================
# 5) StylisticBias (2606.20527)
# =============================================================================
def stylistic_method():
    fig, ax = _canvas(
        "StylisticBias — single-attribute perturbation of fixed identities",
        "500 base faces  ->  ~50 single-attribute variations each (~25K imgs, identity fixed)  ->  6 MLLMs x 25 binary judgments",
    )
    _box(ax, 0.015, 0.42, 0.20, 0.30, "cyan", "500 base faces",
         ["photorealistic", "synthetic identities"], title_size=10.5)
    _box(ax, 0.245, 0.42, 0.24, 0.30, "violet", "Single-attribute edits",
         ["change ONE cue at a time:", "age / body type /", "fashion style ...", "identity held fixed"], title_size=10.2, body_size=8.6)
    _box(ax, 0.245, 0.12, 0.24, 0.18, "slate", "~25K controlled images",
         ["~50 variants per face"], title_size=10.2, body_size=8.8)
    _box(ax, 0.515, 0.42, 0.22, 0.30, "blue", "6 MLLMs",
         ["25 binary social", "judgment scenarios", "(SES, trust, style ...)"], title_size=10.2)
    _box(ax, 0.765, 0.42, 0.22, 0.30, "amber", "Attribute attribution",
         ["clean separation of", "appearance effect vs", "identity difference"], title_size=10.2)
    _arrow(ax, 0.215, 0.57, 0.245, 0.57)
    _arrow(ax, 0.485, 0.57, 0.515, 0.57)
    _arrow(ax, 0.735, 0.57, 0.765, 0.57)
    _arrow(ax, 0.365, 0.30, 0.365, 0.41, color=EDGE, dashed=True)
    ax.text(0.5, 0.045, "Fixing identity and perturbing one attribute isolates which visual cues drive a model's social judgments.",
            ha="center", va="bottom", fontsize=9, color=MUTED, style="italic")
    _save(fig, "2606.20527.svg")


def stylistic_exp():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    fig.suptitle("StylisticBias — a few cues drive most bias",
                 fontsize=13.5, fontweight="bold", x=0.012, ha="left", y=1.005)
    ax = axes[0]
    n = list(range(0, 51, 5))
    # saturating cumulative curve hitting ~80% by ~15 attributes
    cum = [0, 42, 63, 78, 85, 89, 92, 94.5, 96.5, 98, 100]
    ax.plot(n, cum, "-o", color="#7c3aed", lw=2.4, markersize=4, zorder=3)
    ax.axhline(80, color="#94a3b8", ls="--", lw=1.2)
    ax.axvline(15, color="#e11d48", ls="--", lw=1.2)
    ax.plot(15, 80, "o", color="#e11d48", zorder=4)
    ax.annotate("~15 attributes\n≈ 80% of variation", xy=(15, 80), xytext=(22, 55),
                fontsize=9, color="#e11d48", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#e11d48", lw=1.2))
    ax.set_xlabel("# attributes (ranked by influence)")
    ax.set_ylabel("Cumulative variation explained (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Bias concentrates in few visual cues", fontsize=11, color=SUB)
    _style_axes(ax)
    ax = axes[1]
    labels = ["Age", "Body\ntype", "Fashion\nstyle", "Hair", "Accessory"]
    vals = [100, 92, 88, 47, 35]
    colors = ["#2563eb", "#2563eb", "#10b981", "#94a3b8", "#94a3b8"]
    bars = ax.bar(labels, vals, color=colors, width=0.65, zorder=3)
    _bar_labels(ax, bars, fmt="{:.0f}")
    ax.set_ylabel("Relative bias-shift magnitude")
    ax.set_ylim(0, 120)
    ax.set_title("Age / body type / fashion dominate", fontsize=11, color=SUB)
    _style_axes(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.20527_exp.svg")


# =============================================================================
# 6) NAMESAKES (2606.20155)
# =============================================================================
def namesakes_method():
    fig, ax = _canvas(
        "NAMESAKES — black-box probe of identity memorization in T2I models",
        "Name prompts (+ perturbed names)  ->  T2I model generates faces  ->  behavioral probe  ->  memorized vs fabricated",
    )
    _box(ax, 0.015, 0.42, 0.20, 0.30, "cyan", "Name prompts",
         ["1269 name-face pairs", "varied fame levels", "+ perturbed-name controls"], title_size=10.2)
    _box(ax, 0.245, 0.42, 0.21, 0.30, "blue", "T2I model",
         ["generate face from", "the name prompt", "(black box only)"], title_size=10.2)
    _box(ax, 0.485, 0.42, 0.23, 0.30, "violet", "Behavioral probe",
         ["compare behavior across", "name vs perturbed name", "no ref photo / weights"], title_size=10.2)
    _box(ax, 0.745, 0.55, 0.24, 0.17, "green", "Memorized",
         ["consistent likeness"], title_size=10.5, body_size=8.8)
    _box(ax, 0.745, 0.30, 0.24, 0.17, "rose", "Fabricated",
         ["unrecognized / invented"], title_size=10.5, body_size=8.8)
    _arrow(ax, 0.215, 0.57, 0.245, 0.57)
    _arrow(ax, 0.455, 0.57, 0.485, 0.57)
    _arrow(ax, 0.715, 0.60, 0.745, 0.63, color="#10b981")
    _arrow(ax, 0.715, 0.54, 0.745, 0.40, color="#e11d48")
    ax.text(0.5, 0.06, "Distinguishes memorized vs fabricated faces with NO reference photos, training data, or white-box access.",
            ha="center", va="bottom", fontsize=9, color=MUTED, style="italic")
    _save(fig, "2606.20155.svg")


def namesakes_exp():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    fig.suptitle("NAMESAKES — the probe separates memorized from unrecognized names",
                 fontsize=13.5, fontweight="bold", x=0.012, ha="left", y=1.005)
    ax = axes[0]
    labels = ["High-fame\n(memorized)", "Low-fame", "Perturbed\nname"]
    vals = [0.82, 0.41, 0.18]
    bars = ax.bar(labels, vals, color=["#10b981", "#0891b2", "#e11d48"], width=0.6, zorder=3)
    _bar_labels(ax, bars, fmt="{:.2f}")
    ax.set_ylabel("Memorization probe score (illustrative)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Clear separation across fame / control", fontsize=11, color=SUB)
    _style_axes(ax)
    ax = axes[1]
    labels = ["Model A", "Model B", "Model C"]
    vals = [0.71, 0.58, 0.44]
    bars = ax.bar(labels, vals, color=["#2563eb", "#7c3aed", "#94a3b8"], width=0.55, zorder=3)
    _bar_labels(ax, bars, fmt="{:.2f}")
    ax.set_ylabel("Memorization rate across T2I families")
    ax.set_ylim(0, 1.0)
    ax.set_title("Differences across model families  ·  1269 pairs", fontsize=11, color=SUB)
    _style_axes(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "2606.20155_exp.svg")


FIGS = [
    timeprove_method, timeprove_exp,
    nest_method, nest_exp,
    cod_method, cod_exp,
    matm_method, matm_exp,
    stylistic_method, stylistic_exp,
    namesakes_method, namesakes_exp,
]


def main() -> None:
    for fn in FIGS:
        fn()
    print(f"\nDone: {len(FIGS)} figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()

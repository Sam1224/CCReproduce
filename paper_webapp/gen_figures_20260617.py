#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate 8 light-themed, research-style methodology SVG figures for the
2026-06-17 paper-radar inspection.

Each figure is a real architecture / pipeline diagram: labeled rounded boxes,
arrows, and an input -> modules -> output flow. Output is written to
``paper_webapp/assets/figures/{id}.svg``.

Run:  python gen_figures_20260617.py
"""

from __future__ import annotations

import os
from pathlib import Path
from xml.sax.saxutils import escape

WEBAPP_DIR = Path(__file__).resolve().parent
FIG_DIR = WEBAPP_DIR / "assets" / "figures"

WIDTH = 760

# ---- soft, light research palette -------------------------------------------
PALETTE = {
    "bg0": "#f8fafc",
    "bg1": "#eef2ff",
    "ink": "#0f172a",
    "sub": "#475569",
    "body": "#334155",
    "edge": "#94a3b8",
    # box fills + strokes (soft)
    "slate": ("#f1f5f9", "#e2e8f0"),
    "blue": ("#eff6ff", "#bfdbfe"),
    "cyan": ("#ecfeff", "#a5f3fc"),
    "teal": ("#f0fdfa", "#99f6e4"),
    "green": ("#ecfdf5", "#bbf7d0"),
    "amber": ("#fefce8", "#fde68a"),
    "orange": ("#fff7ed", "#fed7aa"),
    "violet": ("#f5f3ff", "#ddd6fe"),
    "rose": ("#fff1f2", "#fecdd3"),
}

ACCENTS = {
    "blue": "#0ea5e9",
    "teal": "#14b8a6",
    "orange": "#f97316",
    "violet": "#8b5cf6",
    "green": "#22c55e",
    "rose": "#f43f5e",
}


def _defs(height: int) -> str:
    return f"""
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{PALETTE['bg0']}"/>
      <stop offset="100%" stop-color="{PALETTE['bg1']}"/>
    </linearGradient>
    <linearGradient id="bar" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#0ea5e9"/>
      <stop offset="100%" stop-color="#14b8a6"/>
    </linearGradient>
    <filter id="sh" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="4" stdDeviation="7" flood-color="#0f172a" flood-opacity="0.10"/>
    </filter>
    <marker id="arr" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M0,0 L9,4 L0,8 z" fill="{PALETTE['edge']}"/>
    </marker>
    <marker id="arrR" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M0,0 L9,4 L0,8 z" fill="{ACCENTS['rose']}"/>
    </marker>
    <marker id="arrO" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M0,0 L9,4 L0,8 z" fill="{ACCENTS['orange']}"/>
    </marker>
    <style>
      .t  {{ font:700 22px ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; fill:#0f172a }}
      .s  {{ font:500 13px ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; fill:#475569 }}
      .h  {{ font:700 14px ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; fill:#0f172a }}
      .p  {{ font:500 11.5px ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; fill:#334155 }}
      .m  {{ font:600 11px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; fill:#0f766e }}
      .lbl{{ font:600 11px ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; fill:#475569 }}
      .stage {{ font:800 12px ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; fill:#0ea5e9 }}
    </style>
  </defs>
"""


def header(title: str, subtitle: str, width: int = WIDTH) -> str:
    return f"""
  <g filter="url(#sh)"><rect x="20" y="18" width="{width - 40}" height="60" rx="14" fill="#ffffff"/></g>
  <rect x="20" y="18" width="{width - 40}" height="5" rx="2.5" fill="url(#bar)"/>
  <text class="t" x="38" y="48">{escape(title)}</text>
  <text class="s" x="38" y="68">{escape(subtitle)}</text>
"""


def box(x, y, w, h, color, title, lines, title_dy=22, line_dy=18, line_start=42,
        mono_idx=None, rx=12, title_class="h"):
    """A rounded box with a bold title and body lines.

    ``color`` is a key into PALETTE (fill, stroke). ``mono_idx`` is a set of
    line indices to render in monospace (.m).
    """
    fill, stroke = PALETTE[color]
    mono_idx = mono_idx or set()
    parts = [
        f'<g filter="url(#sh)"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/></g>',
        f'<text class="{title_class}" x="{x + 14}" y="{y + title_dy}">{escape(title)}</text>',
    ]
    cy = y + line_start
    for i, ln in enumerate(lines):
        cls = "m" if i in mono_idx else "p"
        parts.append(f'<text class="{cls}" x="{x + 14}" y="{cy}">{escape(ln)}</text>')
        cy += line_dy
    return "\n".join(parts)


def tag(x, y, text, accent="blue"):
    """A small accent pill label sitting on top-left of a stage."""
    return (
        f'<rect x="{x}" y="{y}" width="26" height="18" rx="9" fill="{ACCENTS[accent]}" opacity="0.16"/>'
        f'<text class="stage" x="{x + 7}" y="{y + 13}">{escape(text)}</text>'
    )


def varrow(x, y1, y2, color=None, dashed=False, marker="arr"):
    color = color or PALETTE["edge"]
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    return (f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" '
            f'stroke-width="2.5"{dash} marker-end="url(#{marker})"/>')


def harrow(x1, x2, y, color=None, dashed=False, marker="arr"):
    color = color or PALETTE["edge"]
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    return (f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" '
            f'stroke-width="2.5"{dash} marker-end="url(#{marker})"/>')


def path_arrow(d, color=None, dashed=False, marker="arr"):
    color = color or PALETTE["edge"]
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.5"{dash} '
            f'marker-end="url(#{marker})"/>')


def edge_label(x, y, text):
    return f'<text class="lbl" x="{x}" y="{y}">{escape(text)}</text>'


def svg_wrap(height: int, body: str, width: int = WIDTH) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        + _defs(height)
        + f'<rect width="{width}" height="{height}" rx="18" fill="url(#bg)"/>'
        + body
        + "</svg>\n"
    )


def save(fig_id: str, svg: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{fig_id}.svg"
    out.write_text(svg, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size} bytes)")


# =============================================================================
# 1) OneRank  (2606.16838) — bottom -> top vertical flow
# =============================================================================
def fig_onerank() -> str:
    H = 760
    b = [header("OneRank", "Transformer-native multi-task ranking (bottom -> top forward pass)")]
    W = WIDTH
    bx, bw = 60, W - 120
    # bottom: structured tokenization
    y1 = 600
    b.append(tag(bx, y1 - 20, "in", "blue"))
    b.append(box(bx, y1, bw, 110, "blue", "1. Structured Tokenization",
                 ["Interaction History  |  Preference Anchor",
                  "per-candidate group:  [ candidate e_i + task tokens t1 / t2 / t3 ]",
                  "tokens assembled into one sequence"], mono_idx={1}))
    # encoding
    y2 = 470
    b.append(tag(bx, y2 - 20, "x L", "teal"))
    b.append(box(bx, y2, bw, 92, "cyan", "2. Task-Specific Encoding  (x L layers)",
                 ["masked multi-head self-attention",
                  "the 3 task tokens are MUTUALLY INVISIBLE -> early specialization"]))
    # contextualization
    y3 = 350
    b.append(tag(bx, y3 - 20, "ctx", "violet"))
    b.append(box(bx, y3, bw, 82, "violet", "3. Candidate-Aware Contextualization",
                 ["Situational Descriptor -> per-task query",
                  "per-task multi-head cross-attention over candidates -> h_k"]))
    # multi-task prediction
    y4 = 230
    b.append(tag(bx, y4 - 20, "mtl", "orange"))
    b.append(box(bx, y4, bw, 82, "amber", "4. Multi-Task Prediction",
                 ["cross-task relational attention over {h_k}",
                  "GRADIENT DETACHMENT: only diagonal/self grads flow -> z_k"]))
    # scoring
    y5 = 110
    b.append(tag(bx, y5 - 20, "out", "green"))
    b.append(box(bx, y5, bw, 82, "green", "5. Matching-based Scoring",
                 ["s_i = z_k . r_k^i   (inner product per task)",
                  "Click  /  Cart  /  Order  scores"], mono_idx={0}))
    # arrows bottom->top
    cx = W / 2
    for ya, yb in [(y1, y2 + 92), (y2, y3 + 82), (y3, y4 + 82), (y4, y5 + 82)]:
        b.append(varrow(cx, ya, yb))
    return svg_wrap(H, "\n".join(b))


# =============================================================================
# 2) ReaEmb (2606.16703) — two stages sharing a frozen LLM
# =============================================================================
def fig_reaemb() -> str:
    H = 700
    W = WIDTH
    b = [header("ReaEmb", "Two stages over a FROZEN LLM -> precomputed item embeddings")]
    # frozen LLM band
    b.append(box(60, 96, W - 120, 34, "slate", "Frozen LLM (shared backbone, weights kept frozen)", [],
                 title_class="h"))
    # Stage I LRCL
    sx, sw = 60, 330
    b.append(tag(sx, 150 - 18, "I", "blue"))
    b.append(box(sx, 150, sw, 70, "blue", "Stage I  - LRCL",
                 ["input: item text E_i + k reasoning placeholders"]))
    b.append(box(sx, 240, sw, 64, "cyan", "1st forward",
                 ["latent reasoning via lightweight attention -> R_i"]))
    b.append(box(sx, 322, sw, 64, "cyan", "2nd forward",
                 ["mean-pool -> item embedding e_i"]))
    b.append(box(sx, 404, sw, 60, "teal", "Loss",
                 ["attribute-level contrastive loss"]))
    # arrows stage I
    cxa = sx + sw / 2
    b.append(varrow(cxa, 220, 240))
    b.append(varrow(cxa, 304, 322))
    b.append(varrow(cxa, 386, 404))
    # Stage II CRRL
    tx, tw = 430, 270
    b.append(tag(tx, 150 - 18, "II", "orange"))
    b.append(box(tx, 150, tw, 70, "amber", "Stage II  - CRRL",
                 ["Gaussian-perturbed reasoning", "vectors -> G samples"]))
    b.append(box(tx, 252, tw, 70, "orange", "Reward",
                 ["co-occurrence reward", "+ BRPO advantage"]))
    b.append(box(tx, 354, tw, 60, "orange", "Policy update",
                 ["GRPO update on samples"]))
    cxb = tx + tw / 2
    b.append(varrow(cxb, 220, 252))
    b.append(varrow(cxb, 322, 354))
    # output
    b.append(box(60, 510, W - 120, 86, "green", "Output: precomputed item embeddings",
                 ["each item -> fixed embedding e_i (offline, no online LLM cost)",
                  "initialize SRS backbone:  SASRec  /  GRU4Rec"], mono_idx={1}))
    # stage I/II -> output
    b.append(path_arrow(f"M {cxa} 464 L {cxa} 510"))
    b.append(path_arrow(f"M {cxb} 414 L {cxb} 510"))
    return svg_wrap(H, "\n".join(b))


# =============================================================================
# 3) SRPFN (2606.15752) — synthetic prior pretrain + update-free inference
# =============================================================================
def fig_srpfn() -> str:
    H = 660
    W = WIDTH
    b = [header("SRPFN", "Synthetic-prior PFN -> update-free single-pass next-item inference")]
    # Pretrain (left top)
    b.append(tag(60, 100 - 18, "pre", "violet"))
    b.append(box(60, 100, 320, 96, "violet", "Synthetic Prior (offline, once)",
                 ["hDCSBM item graph + random walks",
                  "-> 25.6M synthetic sequences",
                  "covers diverse transition regimes"]))
    b.append(box(420, 100, 280, 96, "slate", "Pretrain PFN  (once)",
                 ["causal Transformer trained to",
                  "approximate posterior-predictive",
                  "distribution (PPD)"]))
    b.append(harrow(380, 420, 148))
    # Inference inputs
    b.append(tag(60, 250 - 18, "inf", "blue"))
    b.append(box(60, 250, 320, 96, "blue", "Target-domain Support Set",
                 ["item-item transition samples",
                  "(as in-context memory)"]))
    b.append(box(60, 366, 320, 96, "cyan", "User Sequence Embedding",
                 ["low-rank PPMI + truncated SVD",
                  "source/destination directionality"]))
    # encoder + cross attn
    b.append(box(420, 250, 280, 80, "teal", "Causal Transformer Encoder",
                 ["encode user sequence -> query q"]))
    b.append(box(420, 366, 280, 96, "amber", "Cross-Attention over Support",
                 ["q attends to support-set memory",
                  "gated posterior update"]))
    b.append(harrow(380, 420, 290))
    b.append(harrow(380, 420, 408))
    b.append(path_arrow("M 560 330 L 560 366"))
    # output
    b.append(box(60, 510, W - 120, 80, "green", "Next-item Distribution   (update-free, ~1 min)",
                 ["single forward pass, NO gradient update at inference time",
                  "p(next item | user seq, support set)"], mono_idx={1}))
    b.append(path_arrow("M 560 462 L 560 510"))
    b.append(path_arrow("M 220 462 L 220 510"))
    return svg_wrap(H, "\n".join(b))


# =============================================================================
# 4) UniAR (2606.18249) — left tokenizer / mid backbone / right decoders
# =============================================================================
def fig_uniar() -> str:
    H = 560
    W = WIDTH
    b = [header("UniAR", "Shared discrete tokenizer -> unified AR backbone -> dual decoders")]
    # left
    b.append(tag(40, 110 - 18, "in", "blue"))
    b.append(box(40, 110, 210, 320, "blue", "Unified Visual Tokenizer",
                 ["Image input",
                  "",
                  "multi-level feature fusion",
                  "(shallow detail + deep semantics)",
                  "",
                  "lookup-free BSQ",
                  "binary codes (64-bit)",
                  "",
                  "2x2 merge -> discrete tokens"], line_dy=30, mono_idx={5, 6}))
    # middle
    b.append(tag(280, 110 - 18, "AR", "teal"))
    b.append(box(280, 110, 230, 320, "teal", "Unified AR Backbone",
                 ["Qwen3-8B",
                  "",
                  "text & visual tokens",
                  "interleaved in one stream",
                  "",
                  "next-token prediction",
                  "",
                  "Parallel-Bitwise-Prediction",
                  "head over the 2x2 grid"], line_dy=30, mono_idx={0}))
    # right two decoders
    b.append(tag(540, 110 - 18, "out", "green"))
    b.append(box(540, 110, 180, 150, "green", "Text-free DiT Decoder",
                 ["visual tokens only",
                  "(no text input)",
                  "",
                  "+ upsampling",
                  "-> 1024^2 image"], line_dy=24, mono_idx={4}))
    b.append(box(540, 290, 180, 140, "amber", "Text Decode",
                 ["language head",
                  "",
                  "-> answer / caption",
                  "(understanding)"], line_dy=24))
    # arrows
    b.append(harrow(250, 280, 270))
    b.append(path_arrow("M 510 200 L 540 185"))
    b.append(path_arrow("M 510 300 L 540 350"))
    return svg_wrap(H, "\n".join(b))


# =============================================================================
# 5) IIRG (2606.17276) — shared LLM, 3 parallel task branches
# =============================================================================
def fig_iirg() -> str:
    H = 640
    W = WIDTH
    b = [header("IIRG", "Shared LLM with 3 parallel training branches (joint multi-task loss)")]
    # input
    b.append(box(230, 96, 300, 56, "slate", "User interaction sequence",
                 ["(input shared by all branches)"]))
    # shared LLM
    b.append(box(210, 180, 340, 50, "violet", "Shared LLM (one model, joint training)", []))
    b.append(varrow(W / 2, 152, 180))
    # three branches
    cols = [
        (40, "blue", "Next-item Prediction", ["user seq -> next item", "ID + title"], "L_u", "(inference uses ONLY this)"),
        (290, "cyan", "Collaborative-neighbor Gen", ["multi-hop co-occurrence", "top-N neighbors"], "lambda1 . L^C", ""),
        (540, "teal", "Semantic-neighbor Gen", ["text-embedding similarity", "top-N neighbors"], "lambda2 . L^S", ""),
    ]
    bw = 180
    for x, color, title, lines, loss, note in cols:
        b.append(box(x, 280, bw, 120, color, title, lines + ([note] if note else [])))
        b.append(box(x + 30, 430, bw - 60, 40, "amber", loss, [], title_class="m"))
        cxc = x + bw / 2
        b.append(path_arrow(f"M {W/2} 230 L {cxc} 280"))
        b.append(varrow(cxc, 400, 430))
    # joint loss
    b.append(box(210, 510, 340, 56, "orange", "Joint loss  L = L_u + lambda1 L^C + lambda2 L^S",
                 ["gradient update on the shared LLM"], mono_idx=set()))
    for x, *_ in cols:
        cxc = x + bw / 2
        b.append(path_arrow(f"M {cxc} 470 L {min(max(cxc,250),510)} 510", color=PALETTE['edge']))
    return svg_wrap(H, "\n".join(b))


# =============================================================================
# 6) RecLoop (2606.17707) — horizontal closed loop
# =============================================================================
def fig_recloop() -> str:
    H = 560
    W = WIDTH
    b = [header("RecLoop", "Closed-loop simulation: recommender <-> LLM user agent")]
    # recommender (left)
    b.append(box(40, 130, 200, 120, "blue", "Recommender",
                 ["retrain (periodic)",
                  "+ candidate generator",
                  "",
                  "produces Exposure List E"], mono_idx={3}))
    # LLM user agent (right)
    b.append(box(520, 130, 200, 150, "teal", "LLM User Agent",
                 ["Dynamic Profile",
                  "Dual Memory (short/long)",
                  "Periodic Reflection",
                  "Action -> select item"]))
    # exposure list (top middle)
    b.append(box(300, 96, 160, 50, "cyan", "Exposure List E", []))
    # feedback collector (bottom middle)
    b.append(box(300, 300, 160, 70, "amber", "Feedback Collector",
                 ["-> Dataset Update"]))
    # loop arrows
    b.append(path_arrow("M 240 150 C 270 120, 290 120, 300 121"))      # rec -> E
    b.append(path_arrow("M 460 121 C 490 120, 500 130, 520 150"))      # E -> agent
    b.append(path_arrow("M 600 280 C 600 330, 520 335, 460 335"))      # agent -> feedback
    b.append(path_arrow("M 300 335 C 180 335, 140 300, 140 250"))      # feedback -> rec
    b.append(edge_label(300, 200, "selects item -> feedback"))
    # side metric box
    b.append(box(160, 410, 440, 96, "violet", "Code-Space Structural Cocoon metric",
                 ["layer-wise code entropy",
                  "Top-k concentration over learned discrete codes"]))
    b.append(path_arrow("M 380 370 L 380 410", dashed=True, color=ACCENTS['violet']))
    return svg_wrap(H, "\n".join(b))


# =============================================================================
# 7) MODE-RAG (2606.17449) — router + 5-agent series + overseer
# =============================================================================
def fig_moderag() -> str:
    H = 720
    W = WIDTH
    b = [header("MODE-RAG", "FE-Router gating + decoupled 5-agent hallucination pipeline")]
    # input
    b.append(box(60, 96, W - 120, 48, "slate", "Multimodal Query + retrieved context", []))
    # diamond router
    cx = W / 2
    dy = 200
    fill, stroke = PALETTE["amber"]
    b.append(f'<g filter="url(#sh)"><polygon points="{cx},{dy-46} {cx+120},{dy} {cx},{dy+46} {cx-120},{dy}" fill="{fill}" stroke="{stroke}"/></g>')
    b.append(f'<text class="h" x="{cx}" y="{dy-6}" text-anchor="middle">FE-Router</text>')
    b.append(f'<text class="p" x="{cx}" y="{dy+12}" text-anchor="middle">VFE + attention gate</text>')
    b.append(f'<text class="m" x="{cx}" y="{dy+30}" text-anchor="middle">mean F &gt; gamma ?</text>')
    b.append(varrow(cx, 144, dy - 46))
    # low-risk bypass (right)
    b.append(box(560, dy - 28, 150, 56, "green", "Low-risk: bypass", ["-> output directly"]))
    b.append(harrow(cx + 120, 560, dy))
    b.append(edge_label(cx + 130, dy - 8, "low risk"))
    # high-risk: 5 agents in series (vertical)
    agents = [
        ("Per-Agent", ["perception / parse"], "blue"),
        ("Cor-Agent", ["cross-modal correction"], "cyan"),
        ("Ret-Agent", ["retrieval + manifold filter"], "teal"),
        ("Rea-Agent", ["MCTS causal DAG reasoning"], "violet"),
        ("Gen-Agent", ["generation + logit perturbation"], "orange"),
    ]
    ax, aw = 120, 300
    ay = 290
    ah = 56
    gap = 16
    b.append(edge_label(40, 285, "high risk (F > gamma)"))
    b.append(path_arrow(f"M {cx-90} {dy+30} C 200 {dy+50}, {ax+aw/2} {dy+60}, {ax+aw/2} {ay}"))
    ys = []
    for i, (name, lines, color) in enumerate(agents):
        y = ay + i * (ah + gap)
        ys.append(y)
        b.append(box(ax, y, aw, ah, color, name, lines))
        if i > 0:
            b.append(varrow(ax + aw / 2, ys[i - 1] + ah, y))
    # overseer (right of agents)
    ovx = 470
    b.append(box(ovx, 330, 230, 120, "rose", "PORAG Overseer",
                 ["triple-consistency check",
                  "over agent outputs",
                  "",
                  "recursive fallback ->"], mono_idx=set()))
    last_y = ys[-1]
    b.append(path_arrow(f"M {ax+aw} {last_y+ah/2} C 440 {last_y+ah/2}, {ovx} 430, {ovx} 430"))
    # recursive fallback red dashed back to Rea-Agent
    rea_y = ys[3]
    b.append(path_arrow(f"M {ovx} 360 C 440 320, {ax+aw} {rea_y+10}, {ax+aw} {rea_y+ah/2}",
                        color=ACCENTS['rose'], dashed=True, marker="arrR"))
    b.append(edge_label(ovx - 20, 320, "recursive fallback"))
    # final output
    b.append(box(ovx, 480, 230, 56, "green", "Hallucination-free report", []))
    b.append(varrow(ovx + 115, 450, 480))
    return svg_wrap(H, "\n".join(b))


# =============================================================================
# 8) Daraz (2606.16387) — left -> right benchmark pipeline
# =============================================================================
def fig_daraz() -> str:
    H = 600
    W = WIDTH
    b = [header("Daraz / BanglishRev", "Code-mixed Top-N recommendation benchmark pipeline")]
    # main horizontal stages
    y = 150
    b.append(box(30, y, 175, 130, "blue", "BanglishRev Data",
                 ["3.24M ratings",
                  "BN + EN + Banglish",
                  "(code-mixed reviews)"]))
    b.append(box(225, y, 175, 130, "cyan", "Preprocess",
                 ["dedup",
                  "k-core filter",
                  "chronological",
                  "leave-last-out"]))
    b.append(box(420, y, 175, 130, "teal", "Models",
                 ["GlobalPop / CatPop",
                  "UserCF / ItemCF",
                  "ExplicitMF / ImplicitMF",
                  "CBF"]))
    b.append(box(615, y, 115, 130, "green", "Eval",
                 ["HR@K", "NDCG@K"]))
    for x in (205, 400, 595):
        b.append(harrow(x, x + 20, y + 65))
    # two sub-branches from Eval
    b.append(box(255, 400, 220, 96, "amber", "k-core sparsity ablation",
                 ["5 density thresholds",
                  "ImplicitMF most hurt;",
                  "ItemCF most robust"]))
    b.append(box(510, 400, 220, 96, "orange", "Language-stratified CBF",
                 ["char 1-2gram TF-IDF",
                  "Banglish NDCG@10 -46.8%",
                  "vs Bengali users"]))
    b.append(path_arrow("M 660 280 C 660 340, 420 350, 365 400"))
    b.append(path_arrow("M 690 280 C 700 340, 640 360, 620 400"))
    return svg_wrap(H, "\n".join(b))


FIGS = {
    "2606.16838": fig_onerank,
    "2606.16703": fig_reaemb,
    "2606.15752": fig_srpfn,
    "2606.18249": fig_uniar,
    "2606.17276": fig_iirg,
    "2606.17707": fig_recloop,
    "2606.17449": fig_moderag,
    "2606.16387": fig_daraz,
}


def main() -> None:
    for fig_id, fn in FIGS.items():
        save(fig_id, fn())
    print(f"\nDone: {len(FIGS)} figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()

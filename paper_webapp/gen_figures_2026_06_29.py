"""Generate lightweight methodology + experiment SVGs for 2026-06-29 papers.

Notes
- These SVGs are intentionally self-contained (no external assets).
- They are schematic figures for the web dashboard when the original paper figure
  cannot be reused directly.
"""

from __future__ import annotations

from pathlib import Path


WEBAPP_DIR = Path(__file__).resolve().parent
FIG_DIR = WEBAPP_DIR / "assets" / "figures"


def svg_header(width: int, height: int) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>\n"
    )


def svg_footer() -> str:
    return "</svg>\n"


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def method_svg(title: str, steps: list[str]) -> str:
    width, height = 1040, 300
    pad = 28
    box_h = 74
    gap = 18

    # simple 3-4 boxes horizontally
    n = len(steps)
    box_w = int((width - pad * 2 - gap * (n - 1)) / n)
    y = 120

    out = [svg_header(width, height)]
    out.append(
        "  <defs>\n"
        "    <linearGradient id='bg' x1='0' y1='0' x2='1' y2='1'>\n"
        "      <stop offset='0%' stop-color='#f7f9ff'/>\n"
        "      <stop offset='100%' stop-color='#f4fbf7'/>\n"
        "    </linearGradient>\n"
        "    <filter id='shadow' x='-20%' y='-20%' width='140%' height='140%'>\n"
        "      <feDropShadow dx='0' dy='6' stdDeviation='8' flood-color='#0f172a' flood-opacity='0.10'/>\n"
        "    </filter>\n"
        "    <marker id='arrow' markerWidth='10' markerHeight='10' refX='9' refY='3' orient='auto'>\n"
        "      <path d='M0,0 L10,3 L0,6 Z' fill='#334155'/>\n"
        "    </marker>\n"
        "  </defs>\n"
    )
    out.append(f"  <rect x='0' y='0' width='{width}' height='{height}' fill='url(#bg)'/>\n")

    out.append(
        f"  <text x='{pad}' y='52' font-size='20' font-family='ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto' "
        f"fill='#0f172a' font-weight='700'>{esc(title)}</text>\n"
    )
    out.append(
        f"  <text x='{pad}' y='80' font-size='13' font-family='ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto' "
        f"fill='#475569'>Methodology sketch</text>\n"
    )

    x = pad
    centers = []
    for i, step in enumerate(steps):
        out.append(
            f"  <rect x='{x}' y='{y}' width='{box_w}' height='{box_h}' rx='14' ry='14' "
            f"fill='#ffffff' stroke='#cbd5e1' stroke-width='1.2' filter='url(#shadow)'/>\n"
        )
        out.append(
            f"  <text x='{x + 18}' y='{y + 30}' font-size='12' font-family='ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto' "
            f"fill='#64748b'>Step {i + 1}</text>\n"
        )
        out.append(
            f"  <text x='{x + 18}' y='{y + 55}' font-size='14' font-family='ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto' "
            f"fill='#0f172a' font-weight='600'>{esc(step)}</text>\n"
        )
        centers.append((x + box_w, y + box_h / 2))
        x += box_w + gap

    # arrows
    for i in range(n - 1):
        x1 = pad + (box_w + gap) * i + box_w
        y1 = y + box_h / 2
        x2 = pad + (box_w + gap) * (i + 1)
        y2 = y + box_h / 2
        out.append(
            f"  <line x1='{x1 + 2}' y1='{y1}' x2='{x2 - 2}' y2='{y2}' stroke='#334155' stroke-width='2' marker-end='url(#arrow)'/>\n"
        )

    out.append(svg_footer())
    return "".join(out)


def exp_svg(title: str, subtitle: str, bars: list[tuple[str, float]], unit: str = "") -> str:
    width, height = 1040, 300
    pad = 28
    chart_x = pad
    chart_y = 118
    chart_w = width - pad * 2
    chart_h = 150

    max_v = max([v for _, v in bars] + [1.0])

    out = [svg_header(width, height)]
    out.append(
        "  <defs>\n"
        "    <linearGradient id='bg' x1='0' y1='0' x2='1' y2='1'>\n"
        "      <stop offset='0%' stop-color='#f8fafc'/>\n"
        "      <stop offset='100%' stop-color='#f1f5f9'/>\n"
        "    </linearGradient>\n"
        "    <filter id='shadow' x='-20%' y='-20%' width='140%' height='140%'>\n"
        "      <feDropShadow dx='0' dy='6' stdDeviation='8' flood-color='#0f172a' flood-opacity='0.10'/>\n"
        "    </filter>\n"
        "  </defs>\n"
    )
    out.append(f"  <rect x='0' y='0' width='{width}' height='{height}' fill='url(#bg)'/>\n")

    out.append(
        f"  <text x='{pad}' y='52' font-size='20' font-family='ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto' "
        f"fill='#0f172a' font-weight='700'>{esc(title)}</text>\n"
    )
    out.append(
        f"  <text x='{pad}' y='80' font-size='13' font-family='ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto' "
        f"fill='#475569'>{esc(subtitle)}</text>\n"
    )

    out.append(
        f"  <rect x='{chart_x}' y='{chart_y}' width='{chart_w}' height='{chart_h}' rx='14' ry='14' "
        f"fill='#ffffff' stroke='#cbd5e1' stroke-width='1.2' filter='url(#shadow)'/>\n"
    )

    n = len(bars)
    inner_pad = 18
    inner_x = chart_x + inner_pad
    inner_y = chart_y + inner_pad
    inner_w = chart_w - inner_pad * 2
    inner_h = chart_h - inner_pad * 2

    gap = 14
    bar_w = (inner_w - gap * (n - 1)) / max(1, n)

    for i, (name, val) in enumerate(bars):
        x = inner_x + i * (bar_w + gap)
        h = (val / max_v) * (inner_h - 26)
        y = inner_y + (inner_h - 26) - h

        color = "#60a5fa" if i == n - 1 else "#a7f3d0"  # highlight last bar
        out.append(
            f"  <rect x='{x:.2f}' y='{y:.2f}' width='{bar_w:.2f}' height='{h:.2f}' rx='10' ry='10' fill='{color}'/>\n"
        )
        out.append(
            f"  <text x='{x + bar_w / 2:.2f}' y='{inner_y + inner_h - 6:.2f}' text-anchor='middle' "
            f"font-size='12' font-family='ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto' fill='#334155'>{esc(name)}</text>\n"
        )
        v_txt = f"{val:g}{unit}" if unit else f"{val:g}"
        out.append(
            f"  <text x='{x + bar_w / 2:.2f}' y='{y - 8:.2f}' text-anchor='middle' "
            f"font-size='12' font-family='ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto' fill='#0f172a' font-weight='700'>{esc(v_txt)}</text>\n"
        )

    out.append(svg_footer())
    return "".join(out)


SPECS = {
    "2606.28070v1": {
        "title": "Oxygen AI Item Center (JD)",
        "method_steps": [
            "SKU text + images + signals",
            "Ontology engineering (human-AI)",
            "S2D: retrieve then discriminate",
            "Structured item knowledge serving",
        ],
        "exp": {
            "subtitle": "Production metrics snapshot",
            "bars": [("Precision", 94.2), ("Recall", 82.8), ("Coverage", 80.4)],
            "unit": "%",
        },
    },
    "2606.28059v1": {
        "title": "PermR (constrained reranking)",
        "method_steps": [
            "Baseline ranking + constraints",
            "ILP objective (revenue)",
            "PermR neighbor-swap search",
            "Constraint-safe revenue uplift",
        ],
        "exp": {
            "subtitle": "Online A/B lift",
            "bars": [("Baseline", 0.0), ("PermR", 2.0)],
            "unit": "%",
        },
    },
    "2606.24655v1": {
        "title": "AI-PAVE-Br (attribute extraction)",
        "method_steps": [
            "Product title/desc",
            "Schema-aware prompting",
            "LLM outputs normalized JSON",
            "Golden Set evaluation",
        ],
        "exp": {
            "subtitle": "Mean entity F1",
            "bars": [("Baseline", 59.79), ("AI-PAVE-Br", 74.68)],
        },
    },
    "2606.26686v1": {
        "title": "LeanGuard (moderation)",
        "method_steps": [
            "User text / content",
            "Lightweight encoder",
            "Single-pass classification",
            "Allow/Block + label",
        ],
        "exp": {
            "subtitle": "Accuracy vs compute",
            "bars": [("Avg F1", 82.9), ("Compute ↓", 100.0)],
            "unit": "",
        },
    },
    "2606.27499v1": {
        "title": "DMV-Bench (visual memory)",
        "method_steps": [
            "Shopping sessions (images)",
            "Incidental cue injection",
            "Evaluate long-horizon memory",
            "DualMem improves recall",
        ],
        "exp": {
            "subtitle": "TSR @ chain length J=5",
            "bars": [("No memory", 50.0), ("DualMem", 81.1)],
        },
    },
    "2606.28062v1": {
        "title": "LLM Data Fusion (truth discovery)",
        "method_steps": [
            "Multi-source records",
            "Constraint prompts",
            "LLM selects truth value(s)",
            "Fused canonical output",
        ],
        "exp": {
            "subtitle": "Best F1 on datasets",
            "bars": [("Book", 0.7817), ("Movie", 0.8172), ("Flight", 0.9119)],
        },
    },
    "2606.25415v1": {
        "title": "S2-CAR (adaptive recommendation)",
        "method_steps": [
            "User sequence events",
            "Soft-TPP energy segmentation",
            "Multi-interest encoding",
            "Rank next items",
        ],
        "exp": {
            "subtitle": "Recall@10",
            "bars": [("ML-1M", 28.74), ("Amazon", 5.74), ("Steam", 14.41)],
        },
    },
}


def write_svg(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    missing = []
    for paper_id, spec in SPECS.items():
        method_path = FIG_DIR / f"{paper_id}.svg"
        exp_path = FIG_DIR / f"{paper_id}_exp.svg"

        if not method_path.exists():
            write_svg(method_path, method_svg(spec["title"], spec["method_steps"]))
        if not exp_path.exists():
            exp = spec["exp"]
            write_svg(
                exp_path,
                exp_svg(
                    spec["title"],
                    exp["subtitle"],
                    exp["bars"],
                    unit=exp.get("unit", ""),
                ),
            )

        if not method_path.exists() or not exp_path.exists():
            missing.append(paper_id)

    print(f"wrote/ensured figures in: {FIG_DIR}")
    if missing:
        print("missing:", missing)


if __name__ == "__main__":
    main()

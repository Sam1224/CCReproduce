"""Generate lightweight methodology + experiment SVGs for 2026-06-30 papers.

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
        x += box_w + gap

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

        color = "#60a5fa" if i == n - 1 else "#a7f3d0"
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
    "2606.27708v1": {
        "title": "ZooClaw-FashionSigLIP2",
        "method_steps": [
            "Curated fashion pairs",
            "Distilled full fine-tuning",
            "WiSE-FT interpolation",
            "Robust retrieval",
        ],
        "exp": {
            "subtitle": "R@10 (higher is better)",
            "bars": [("LoRA", 0.612), ("ZooClaw", 0.659), ("LongQ", 0.225)],
        },
    },
    "2606.28344v1": {
        "title": "PixelRAG",
        "method_steps": [
            "Render web pages",
            "Tile screenshots",
            "Visual retrieval",
            "VLM reads pixels",
        ],
        "exp": {
            "subtitle": "Structured QA accuracy",
            "bars": [("TextRAG", 42.5), ("PixelRAG", 48.8)],
            "unit": "%",
        },
    },
    "2606.26740v1": {
        "title": "LiveEdit",
        "method_steps": [
            "Bidirectional editor",
            "3-stage distillation",
            "4-step streaming",
            "AR mask cache",
        ],
        "exp": {
            "subtitle": "Speed profile",
            "bars": [("Typical steps", 25), ("LiveEdit steps", 4), ("FPS", 12.66)],
        },
    },
    "2606.29705v1": {
        "title": "GUICrafter",
        "method_steps": [
            "Unlabeled screenshots",
            "Self-supervised grounding",
            "Small labeled RL",
            "Cross-device agent",
        ],
        "exp": {
            "subtitle": "Label budget (relative)",
            "bars": [("UI-TARS", 100.0), ("GUICrafter", 0.1)],
        },
    },
    "2606.30084v1": {
        "title": "InnerZoom",
        "method_steps": [
            "MLLM decoder",
            "Mid-layer evidence",
            "Cross-layer bridge",
            "1-pass grounding",
        ],
        "exp": {
            "subtitle": "OSWorld-G score",
            "bars": [("Prev best", 60.6), ("InnerZoom", 64.7)],
        },
    },
    "2606.30616v1": {
        "title": "Agents-A1 (35B)",
        "method_steps": [
            "Long-horizon infra",
            "Full-domain SFT",
            "Multi-teacher distill",
            "Unified agent",
        ],
        "exp": {
            "subtitle": "Benchmark scores",
            "bars": [("SEAL-0", 56.4), ("IFBench", 80.6), ("HLE", 47.6)],
        },
    },
    "2606.28480v1": {
        "title": "TUA-Bench",
        "method_steps": [
            "120 real terminal tasks",
            "Deterministic env",
            "Execution scoring",
            "Reliability gaps",
        ],
        "exp": {
            "subtitle": "Overall success rate",
            "bars": [("Best agent", 65.8), ("Gap", 34.2)],
            "unit": "%",
        },
    },
    "2606.28733v1": {
        "title": "Agentic Abstention",
        "method_steps": [
            "Gap taxonomy",
            "Trajectory logging",
            "CONVOLVE playbooks",
            "Timely abstention",
        ],
        "exp": {
            "subtitle": "Timely abstention recall (WebShop)",
            "bars": [("Base", 26.7), ("CONVOLVE", 57.4)],
            "unit": "%",
        },
    },
}


def write_svg(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    for paper_id, spec in SPECS.items():
        method_path = FIG_DIR / f"{paper_id}.svg"
        exp_path = FIG_DIR / f"{paper_id}_exp.svg"

        write_svg(method_path, method_svg(spec["title"], spec["method_steps"]))

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

    print(f"wrote figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()

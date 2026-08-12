from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = REPO_ROOT / "paper_webapp" / "assets" / "figures"
PALETTE = ["#7c3aed", "#2563eb", "#0f766e", "#db2777"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date-dir", required=True, help="Directory containing papers.json")
    return parser.parse_args()


def load_papers(date_dir: Path):
    payload = json.loads((date_dir / "papers.json").read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("papers", [])


def wrap_text(text: str, width: int = 24):
    words = text.split()
    lines = []
    current = []
    current_len = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current_len + extra > width:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra
    if current:
        lines.append(" ".join(current))
    return lines


def svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def render_method_svg(paper_id: str, title: str, steps: list[str]) -> str:
    width, height = 1200, 680
    parts = [
        svg_header(width, height),
        '<rect width="100%" height="100%" rx="28" fill="#f8fafc"/>',
        '<rect x="28" y="24" width="1144" height="96" rx="24" fill="#e2e8f0"/>',
        f'<text x="56" y="72" font-size="28" font-family="Inter,Arial" font-weight="700" fill="#0f172a">{escape(title)}</text>',
        f'<text x="56" y="104" font-size="18" font-family="Inter,Arial" fill="#334155">Methodology overview · {paper_id}</text>',
    ]
    box_width = 248
    gap = 28
    origin_y = 190
    for index, step in enumerate(steps[:4]):
        origin_x = 52 + index * (box_width + gap)
        color = PALETTE[index % len(PALETTE)]
        parts.append(f'<rect x="{origin_x}" y="{origin_y}" width="{box_width}" height="260" rx="26" fill="white" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<circle cx="{origin_x + 36}" cy="{origin_y + 36}" r="18" fill="{color}"/>')
        parts.append(f'<text x="{origin_x + 31}" y="{origin_y + 43}" font-size="18" font-family="Inter,Arial" font-weight="700" fill="white">{index + 1}</text>')
        text_y = origin_y + 88
        for line in wrap_text(step, 20):
            parts.append(f'<text x="{origin_x + 24}" y="{text_y}" font-size="24" font-family="Inter,Arial" font-weight="600" fill="#0f172a">{escape(line)}</text>')
            text_y += 34
        if index < min(len(steps[:4]), 4) - 1:
            arrow_x = origin_x + box_width + 8
            parts.append(f'<path d="M {arrow_x} {origin_y + 130} L {arrow_x + 16} {origin_y + 130} L {arrow_x + 16} {origin_y + 118} L {arrow_x + 40} {origin_y + 140} L {arrow_x + 16} {origin_y + 162} L {arrow_x + 16} {origin_y + 150} L {arrow_x} {origin_y + 150} Z" fill="#94a3b8"/>')
    parts.append('<rect x="52" y="500" width="1096" height="124" rx="24" fill="#ffffff" stroke="#cbd5e1"/>')
    parts.append('<text x="82" y="548" font-size="22" font-family="Inter,Arial" font-weight="700" fill="#0f172a">Key message</text>')
    parts.append('<text x="82" y="586" font-size="20" font-family="Inter,Arial" fill="#334155">The figure highlights the minimum deployable logic so the paper card can show methodology without overflowing the layout.</text>')
    parts.append('</svg>')
    return "".join(parts)


def render_exp_svg(paper_id: str, title: str, cards: list[list[str]]) -> str:
    width, height = 1200, 520
    parts = [
        svg_header(width, height),
        '<rect width="100%" height="100%" rx="28" fill="#fffdf7"/>',
        '<rect x="28" y="24" width="1144" height="84" rx="22" fill="#fef3c7"/>',
        f'<text x="52" y="66" font-size="28" font-family="Inter,Arial" font-weight="700" fill="#0f172a">Experiment highlights · {paper_id}</text>',
        f'<text x="52" y="92" font-size="18" font-family="Inter,Arial" fill="#92400e">{escape(title)}</text>',
    ]
    card_width = 332
    gap = 34
    for index, (label, value) in enumerate(cards[:3]):
        origin_x = 58 + index * (card_width + gap)
        parts.append(f'<rect x="{origin_x}" y="156" width="{card_width}" height="220" rx="28" fill="white" stroke="#f59e0b" stroke-width="2.5"/>')
        parts.append(f'<text x="{origin_x + 26}" y="214" font-size="22" font-family="Inter,Arial" font-weight="700" fill="#0f172a">{escape(label)}</text>')
        parts.append(f'<text x="{origin_x + 26}" y="286" font-size="40" font-family="Inter,Arial" font-weight="800" fill="#b45309">{escape(value)}</text>')
        parts.append(f'<text x="{origin_x + 26}" y="338" font-size="18" font-family="Inter,Arial" fill="#475569">Brightest quantitative evidence extracted for the paper card.</text>')
    parts.append('</svg>')
    return "".join(parts)


def main() -> None:
    args = parse_args()
    date_dir = Path(args.date_dir).resolve()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for paper in load_papers(date_dir):
        paper_id = paper["id"]
        title = paper["title"]
        steps = paper.get("figure_steps") or [title, "Core model", "Training objective", "Evaluation"]
        cards = paper.get("exp_cards") or [["Metric", "See paper"], ["Gain", "N/A"], ["Note", "Manual review"]]
        (FIGURE_DIR / f"{paper_id}.svg").write_text(render_method_svg(paper_id, title, steps), encoding="utf-8")
        (FIGURE_DIR / f"{paper_id}_exp.svg").write_text(render_exp_svg(paper_id, title, cards), encoding="utf-8")
    print(f"generated figures in {FIGURE_DIR}")


if __name__ == "__main__":
    main()

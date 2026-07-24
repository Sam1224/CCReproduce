from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent
WEB_FIG_DIR = ROOT.parent / "paper_webapp" / "assets" / "figures"
PAPERS_PATH = ROOT / "papers.json"

METHOD_CONFIG = {
    "2607.21028": ["User history + semantic IDs", "ICA restores item boundaries", "HPR reranks semantic paths", "DPD fuses dual code channels"],
    "2607.21519": ["Collaborative graph signals", "CAST discrete tokenizer", "Curriculum diffusion denoising", "Voting-based recommendation output"],
    "2607.20465": ["Raw domain corpus", "LLM data construction", "DAS quality scoring", "Downstream utility evaluation"],
    "2607.20938": ["Image / audio / video content", "Multimodal verbalization", "Editable text user profile", "Controllable CF recommendation"],
    "2607.20863": ["Base recommender prediction", "Probabilistic user clustering", "Causal confounder modeling", "Residual correction output"],
    "2607.21401": ["Prompt + image + response", "Pooled VLM encoding", "Reference-vector comparison", "Real-time moderation decision"],
    "2607.20873": ["Sparse ID-list features", "Local held-out estimators", "CPU-only feature ranking", "Budget-aware retained feature set"],
}

EXP_CONFIG = {
    "2607.21028": [("Amazon Beauty R@10", "0.0935 → 0.1118"), ("Tencent CTR", "+0.60%"), ("Reading time", "+1.70%")],
    "2607.21519": [("Avg Recall", "+6.75%"), ("Avg NDCG", "+5.19%"), ("Datasets", "LastFM / ML-1M / Beauty")],
    "2607.20465": [("Finance gain", "+20 pts"), ("DAS correlation", "r > 0.70"), ("Coverage", "6 domains")],
    "2607.20938": [("ML-20M NDCG@100", "0.4836 → 0.4964"), ("Steering", "monotonic add/remove"), ("Signal", "multimodal content")],
    "2607.20863": [("CDL Recall@20", "0.0143 → 0.1091"), ("PerK Recall@20", "0.1098 → 0.1635"), ("Mode", "plug-in residual")],
    "2607.21401": [("All-avg F1", "76.56 → 77.31"), ("Latency", "10.12s → 67.6ms"), ("Speedup", "~150×")],
    "2607.20873": [("Ranking time", "~2 CPU hours"), ("Cost", "$4000 → ~$100"), ("Quality", "NE gain competitive")],
}

PALETTE = ["#7c3aed", "#2563eb", "#0f766e", "#db2777"]


def wrap_text(text: str, width: int = 28):
    words = text.split()
    lines = []
    cur = []
    cur_len = 0
    for word in words:
        extra = len(word) + (1 if cur else 0)
        if cur_len + extra > width:
            lines.append(" ".join(cur))
            cur = [word]
            cur_len = len(word)
        else:
            cur.append(word)
            cur_len += extra
    if cur:
        lines.append(" ".join(cur))
    return lines


def svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def render_method_svg(paper_id: str, title: str, steps):
    width, height = 1200, 680
    parts = [
        svg_header(width, height),
        '<rect width="100%" height="100%" rx="28" fill="#f8fafc"/>',
        '<rect x="28" y="24" width="1144" height="96" rx="24" fill="#e2e8f0"/>',
        f'<text x="56" y="72" font-size="28" font-family="Inter,Arial" font-weight="700" fill="#0f172a">{escape(title)}</text>',
        f'<text x="56" y="104" font-size="18" font-family="Inter,Arial" fill="#334155">Methodology overview · {paper_id}</text>',
    ]
    box_w = 248
    gap = 28
    y = 190
    for idx, step in enumerate(steps):
        x = 52 + idx * (box_w + gap)
        color = PALETTE[idx % len(PALETTE)]
        parts.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="260" rx="26" fill="white" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<circle cx="{x+36}" cy="{y+36}" r="18" fill="{color}"/>')
        parts.append(f'<text x="{x+31}" y="{y+43}" font-size="18" font-family="Inter,Arial" font-weight="700" fill="white">{idx+1}</text>')
        text_y = y + 88
        for line in wrap_text(step, 20):
            parts.append(f'<text x="{x+24}" y="{text_y}" font-size="24" font-family="Inter,Arial" font-weight="600" fill="#0f172a">{escape(line)}</text>')
            text_y += 34
        if idx < len(steps) - 1:
            ax = x + box_w + 8
            parts.append(f'<path d="M {ax} {y+130} L {ax+16} {y+130} L {ax+16} {y+118} L {ax+40} {y+140} L {ax+16} {y+162} L {ax+16} {y+150} L {ax} {y+150} Z" fill="#94a3b8"/>')
    parts.append('<rect x="52" y="500" width="1096" height="124" rx="24" fill="#ffffff" stroke="#cbd5e1"/>')
    parts.append('<text x="82" y="548" font-size="22" font-family="Inter,Arial" font-weight="700" fill="#0f172a">Key message</text>')
    parts.append('<text x="82" y="586" font-size="20" font-family="Inter,Arial" fill="#334155">The figure focuses on the paper\'s core decision path so the web UI can expose methodology without crowding the card body.</text>')
    parts.append('</svg>')
    return "".join(parts)


def render_exp_svg(paper_id: str, title: str, cards):
    width, height = 1200, 520
    parts = [
        svg_header(width, height),
        '<rect width="100%" height="100%" rx="28" fill="#fffdf7"/>',
        '<rect x="28" y="24" width="1144" height="84" rx="22" fill="#fef3c7"/>',
        f'<text x="52" y="66" font-size="28" font-family="Inter,Arial" font-weight="700" fill="#0f172a">Experiment highlights · {paper_id}</text>',
        f'<text x="52" y="92" font-size="18" font-family="Inter,Arial" fill="#92400e">{escape(title)}</text>',
    ]
    card_w = 332
    gap = 34
    for idx, (label, value) in enumerate(cards):
        x = 58 + idx * (card_w + gap)
        parts.append(f'<rect x="{x}" y="156" width="{card_w}" height="220" rx="28" fill="white" stroke="#f59e0b" stroke-width="2.5"/>')
        parts.append(f'<text x="{x+26}" y="214" font-size="22" font-family="Inter,Arial" font-weight="700" fill="#0f172a">{escape(label)}</text>')
        parts.append(f'<text x="{x+26}" y="286" font-size="40" font-family="Inter,Arial" font-weight="800" fill="#b45309">{escape(value)}</text>')
        parts.append(f'<text x="{x+26}" y="338" font-size="18" font-family="Inter,Arial" fill="#475569">Brightest quantitative evidence extracted for the paper card.</text>')
    parts.append('</svg>')
    return "".join(parts)


def main():
    WEB_FIG_DIR.mkdir(parents=True, exist_ok=True)
    papers = json.loads(PAPERS_PATH.read_text(encoding="utf-8"))["papers"]
    for paper in papers:
        pid = paper["id"]
        title = paper["title"]
        method_svg = render_method_svg(pid, title, METHOD_CONFIG[pid])
        exp_svg = render_exp_svg(pid, title, EXP_CONFIG[pid])
        (WEB_FIG_DIR / f"{pid}.svg").write_text(method_svg, encoding="utf-8")
        (WEB_FIG_DIR / f"{pid}_exp.svg").write_text(exp_svg, encoding="utf-8")
    print(f"generated figures in {WEB_FIG_DIR}")


if __name__ == "__main__":
    main()

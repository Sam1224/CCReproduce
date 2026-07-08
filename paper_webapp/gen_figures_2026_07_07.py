from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent / "assets" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PAPERS = {
    "2606.25034": ("YuvionVL", ["Adversarial data pipeline", "C2FT mining", "Multi-image contrastive", "SafetyGate", "YVRE eval"], "+9.9 pts vs open-source, +6.7 pts vs closed-source on YVRE"),
    "2606.14786": ("MatchLM2Lite", ["Video frames", "Audio segments", "Text (title+desc)", "TrimodalFusion", "Pairwise score"], "MatchLite: +6.55 F1 over prior model, 35x cheaper inference"),
    "2606.04448": ("RGCD-Rep", ["Short video", "Teacher MLLM reasoning", "Distill student MLLM", "Transferable repr", "Livestream ranking"], "+17.59% and +30.93% over best baseline; A/B deployed on Kuaishou"),
    "2606.06970": ("SSRLive", ["Livestream content", "Static SID encoder", "Dynamic SID encoder", "User feature fusion", "Multi-task predict"], "+3.38% watch time, +0.72% GMV on Alibaba A/B"),
    "2606.04374": ("DSIRM", ["Query text", "CBCQ item SID", "LLM query SID", "Prefix matching", "Relevance score"], "Query-bridged quantization: relevance-aware codebook partitions"),
    "2606.27632": ("YuvionLLM", ["Text input", "Adversarial data flywheel", "C2FT text version", "Safety alignment", "SOTA safety bench"], "SOTA on text safety benchmarks; complements YuvionVL for full coverage"),
}

PALETTE = ["#2563eb", "#0f766e", "#7c3aed", "#ea580c", "#0891b2"]


def method_svg(title, steps):
    boxes = []
    arrows = []
    for idx, step in enumerate(steps):
        x = 28 + idx * 138
        color = PALETTE[idx % len(PALETTE)]
        boxes.append(f'<rect x="{x}" y="88" width="116" height="72" rx="16" fill="{color}" opacity="0.92"/>')
        boxes.append(f'<text x="{x + 58}" y="119" text-anchor="middle" font-size="12" fill="white" font-weight="700">{step}</text>')
        if idx < len(steps) - 1:
            arrows.append(f'<path d="M{x + 120} 124 L{x + 136} 124" stroke="#334155" stroke-width="2" marker-end="url(#arrow)"/>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="240" viewBox="0 0 760 240">
  <defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#334155"/></marker></defs>
  <rect width="760" height="240" rx="24" fill="#f8fafc"/>
  <text x="32" y="42" font-size="22" font-weight="800" fill="#0f172a">{title} methodology</text>
  <text x="32" y="66" font-size="13" fill="#64748b">Input → model mechanism → governance/content output</text>
  {''.join(arrows)}{''.join(boxes)}
  <rect x="28" y="184" width="704" height="28" rx="14" fill="#e0f2fe"/>
  <text x="380" y="203" text-anchor="middle" font-size="13" fill="#075985">Research-style schematic generated for the daily paper radar</text>
</svg>'''


def exp_svg(title, highlight):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="240" viewBox="0 0 760 240">
  <rect width="760" height="240" rx="24" fill="#ffffff"/>
  <rect x="24" y="24" width="712" height="192" rx="20" fill="#f1f5f9"/>
  <text x="44" y="58" font-size="21" font-weight="800" fill="#0f172a">{title} experiment highlight</text>
  <rect x="50" y="92" width="160" height="82" rx="16" fill="#dbeafe"/><text x="130" y="126" text-anchor="middle" font-size="16" font-weight="700" fill="#1d4ed8">Task</text><text x="130" y="149" text-anchor="middle" font-size="12" fill="#334155">real-world signal</text>
  <rect x="300" y="78" width="160" height="110" rx="18" fill="#dcfce7"/><text x="380" y="120" text-anchor="middle" font-size="18" font-weight="800" fill="#166534">Result</text><text x="380" y="146" text-anchor="middle" font-size="12" fill="#334155">{highlight}</text>
  <rect x="550" y="92" width="160" height="82" rx="16" fill="#fef3c7"/><text x="630" y="126" text-anchor="middle" font-size="16" font-weight="700" fill="#92400e">Value</text><text x="630" y="149" text-anchor="middle" font-size="12" fill="#334155">e-commerce governance</text>
  <path d="M215 133 C250 133 260 133 292 133" stroke="#64748b" stroke-width="3" fill="none" marker-end="url(#arrow)"/><path d="M468 133 C505 133 515 133 542 133" stroke="#64748b" stroke-width="3" fill="none" marker-end="url(#arrow)"/>
  <defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#64748b"/></marker></defs>
</svg>'''

written = 0
for paper_id, (title, steps, highlight) in PAPERS.items():
    method_path = FIG_DIR / f"{paper_id}.svg"
    exp_path = FIG_DIR / f"{paper_id}_exp.svg"
    if not method_path.exists():
        method_path.write_text(method_svg(title, steps), encoding="utf-8")
        written += 1
    if not exp_path.exists():
        exp_path.write_text(exp_svg(title, highlight), encoding="utf-8")
        written += 1

print(f"wrote {written} new figure files")

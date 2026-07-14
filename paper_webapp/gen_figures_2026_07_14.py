from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent / "assets" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PAPERS = {
    "2607.11886": (
        "SpectraReward",
        [
            "Prompt",
            "Diffusion policy",
            "Generated image",
            "MLLM read-back score",
            "RL update",
        ],
        "GenEval rises from 84.0 to 89.5 with Self-SpectraReward",
    ),
    "2607.11862": (
        "EVQA",
        [
            "Video + question",
            "Spatio-temporal grounding",
            "Masklet evidence",
            "Answer + evidence",
            "Audit-ready output",
        ],
        "+27.2 t-mean and +13.8 J&F on 7B after EVQA fine-tuning",
    ),
    "2607.11844": (
        "SportMV-Agent",
        [
            "Multi-view bundle",
            "Active view selection",
            "Perception tools",
            "Evidence reasoning",
            "Decision answer",
        ],
        "14.46% relative gain over the strongest MLLM baseline",
    ),
    "2607.11465": (
        "Score-Only Distillation",
        [
            "Teacher retriever",
            "Score vectors",
            "Row-centered PairMSE",
            "Compact student",
            "Fast dense retrieval",
        ],
        "0.6B student is 4.7×/9.7× faster for query/doc encoding",
    ),
    "2607.11392": (
        "CRID",
        [
            "Item corpus",
            "Semantic clustering",
            "Business-value ranking",
            "DocID tokens",
            "Generative retrieval",
        ],
        "+1.06% GMV in full-traffic Taobao deployment",
    ),
}

PALETTE = ["#2563eb", "#0f766e", "#7c3aed", "#ea580c", "#0891b2"]


def method_svg(title, steps):
    boxes = []
    arrows = []
    for idx, step in enumerate(steps):
        x = 28 + idx * 138
        color = PALETTE[idx % len(PALETTE)]
        boxes.append(
            f'<rect x="{x}" y="88" width="116" height="72" rx="16" fill="{color}" opacity="0.92"/>'
        )
        boxes.append(
            f'<text x="{x + 58}" y="119" text-anchor="middle" font-size="12" fill="white" font-weight="700">{step}</text>'
        )
        if idx < len(steps) - 1:
            arrows.append(
                f'<path d="M{x + 120} 124 L{x + 136} 124" stroke="#334155" stroke-width="2" marker-end="url(#arrow)"/>'
            )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="240" viewBox="0 0 760 240">
  <defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#334155"/></marker></defs>
  <rect width="760" height="240" rx="24" fill="#f8fafc"/>
  <text x="32" y="42" font-size="22" font-weight="800" fill="#0f172a">{title} methodology</text>
  <text x="32" y="66" font-size="13" fill="#64748b">Input → reasoning / modeling core → output</text>
  {''.join(arrows)}{''.join(boxes)}
  <rect x="28" y="184" width="704" height="28" rx="14" fill="#e0f2fe"/>
  <text x="380" y="203" text-anchor="middle" font-size="13" fill="#075985">Research-style schematic generated for the daily paper radar</text>
</svg>'''


def exp_svg(title, highlight):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="240" viewBox="0 0 760 240">
  <rect width="760" height="240" rx="24" fill="#ffffff"/>
  <rect x="24" y="24" width="712" height="192" rx="20" fill="#f1f5f9"/>
  <text x="44" y="58" font-size="21" font-weight="800" fill="#0f172a">{title} experiment highlight</text>
  <rect x="50" y="92" width="160" height="82" rx="16" fill="#dbeafe"/><text x="130" y="126" text-anchor="middle" font-size="16" font-weight="700" fill="#1d4ed8">Task</text><text x="130" y="149" text-anchor="middle" font-size="12" fill="#334155">real-world setting</text>
  <rect x="300" y="78" width="160" height="110" rx="18" fill="#dcfce7"/><text x="380" y="120" text-anchor="middle" font-size="18" font-weight="800" fill="#166534">Result</text><text x="380" y="146" text-anchor="middle" font-size="12" fill="#334155">{highlight}</text>
  <rect x="550" y="92" width="160" height="82" rx="16" fill="#fef3c7"/><text x="630" y="126" text-anchor="middle" font-size="16" font-weight="700" fill="#92400e">Value</text><text x="630" y="149" text-anchor="middle" font-size="12" fill="#334155">content / governance use</text>
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

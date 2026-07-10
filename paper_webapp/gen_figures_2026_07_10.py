from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent / "assets" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PAPERS = {
    "2508.03296": (
        "Hi-Guard",
        ["UGC note", "Binary risk filter", "Policy-grounded reasoning", "Hierarchical risk path", "Decision + evidence"],
        "85.06% online accuracy / 79.14% risky recall / 56.38% less human moderation",
    ),
    "2512.03553": (
        "Livestream Moderation",
        ["20s clip", "Preset classifier", "Reference retrieval", "Multimodal reranker", "Moderation decision"],
        "6%–8% fewer unwanted livestream views in production",
    ),
    "2607.08768": (
        "UniClawBench",
        ["Real-world task", "Executor", "Hidden supervisor", "User simulator", "Capability score"],
        "Framework choice shifts overall pass rate from 0.338 to 0.407 for GPT-5.4",
    ),
    "2607.08763": (
        "OpenCoF",
        ["Prompt", "OpenCoF-17K", "Wan-CoF tuning", "Reasoning tokens", "Reasoned video"],
        "MME-CoF overall score improves from 1.00 to 1.35",
    ),
    "2606.30534": (
        "Orca",
        ["Multimodal world signals", "Unconscious learning", "Conscious learning", "World latent", "Readout decoders"],
        "125K video hours + 160M events for unified next-state prediction",
    ),
    "2601.04200": (
        "Controlled Product Generation",
        ["Base product", "Attribute picker", "Value provider", "Generation LLM", "Synthetic training sample"],
        "Synthetic-only MAVE accuracy 60.5%, hybrid training reaches 68.8%",
    ),
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
  <text x="32" y="66" font-size="13" fill="#64748b">Input → model mechanism → governance or capability output</text>
  {''.join(arrows)}{''.join(boxes)}
  <rect x="28" y="184" width="704" height="28" rx="14" fill="#e0f2fe"/>
  <text x="380" y="203" text-anchor="middle" font-size="13" fill="#075985">Research-style schematic generated for the daily paper radar</text>
</svg>'''


def exp_svg(title, highlight):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="240" viewBox="0 0 760 240">
  <rect width="760" height="240" rx="24" fill="#ffffff"/>
  <rect x="24" y="24" width="712" height="192" rx="20" fill="#f1f5f9"/>
  <text x="44" y="58" font-size="21" font-weight="800" fill="#0f172a">{title} experiment highlight</text>
  <rect x="50" y="92" width="160" height="82" rx="16" fill="#dbeafe"/><text x="130" y="126" text-anchor="middle" font-size="16" font-weight="700" fill="#1d4ed8">Task</text><text x="130" y="149" text-anchor="middle" font-size="12" fill="#334155">paper takeaway</text>
  <rect x="300" y="78" width="160" height="110" rx="18" fill="#dcfce7"/><text x="380" y="120" text-anchor="middle" font-size="18" font-weight="800" fill="#166534">Result</text><text x="380" y="146" text-anchor="middle" font-size="12" fill="#334155">{highlight}</text>
  <rect x="550" y="92" width="160" height="82" rx="16" fill="#fef3c7"/><text x="630" y="126" text-anchor="middle" font-size="16" font-weight="700" fill="#92400e">Value</text><text x="630" y="149" text-anchor="middle" font-size="12" fill="#334155">daily governance radar</text>
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

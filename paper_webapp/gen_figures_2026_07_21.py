from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent / "assets" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PAPERS = {
    "2607.18230": (
        "PIXAR-DG",
        [
            "Real + tampered images",
            "Balanced minibatch",
            "Base-domain convergence",
            "Late domain injection",
            "OOD pixel localization",
        ],
        "OOD gIoU +26.1% and cIoU +26.8% over PIXAR",
    ),
    "2607.18142": (
        "O-VAD",
        [
            "Industrial video",
            "Object grounding",
            "Temporal tracking",
            "State-change reasoning",
            "Anomaly report",
        ],
        "Phys-AD AUROC 0.584 > GPT-5 0.503 and Qwen3-VL 0.513",
    ),
    "2607.17017": (
        "WHALE",
        [
            "User / item / context",
            "Wukong feature crosses",
            "HSTU behavior sequence",
            "Attention fusion",
            "Unified ranking score",
        ],
        "14-day online A/B lifts main metric by 0.113% with only -5% QPS",
    ),
    "2607.16165": (
        "ActiveVision",
        [
            "Rendered visual task",
            "Repeated observation",
            "Cross-region evidence",
            "Reasoning loop",
            "Benchmark score",
        ],
        "Best GPT-5.5 xhigh only 10.6%; humans average 96.1%",
    ),
    "2607.10350": (
        "ABot-AgentOS",
        [
            "Task + observations",
            "Typed graph memory",
            "Hybrid retrieval",
            "Agent execution",
            "Self-evolution loop",
        ],
        "TSR +11.99% and GCR +10.84% over single-controller baseline",
    ),
}

PALETTE = ["#2563eb", "#0f766e", "#7c3aed", "#ea580c", "#0891b2"]


def method_svg(title, steps):
    boxes = []
    arrows = []
    for idx, step in enumerate(steps):
        x = 26 + idx * 140
        color = PALETTE[idx % len(PALETTE)]
        boxes.append(f'<rect x="{x}" y="84" width="118" height="78" rx="16" fill="{color}" opacity="0.94"/>')
        boxes.append(f'<text x="{x + 59}" y="116" text-anchor="middle" font-size="12" fill="white" font-weight="700">{step}</text>')
        if idx < len(steps) - 1:
            arrows.append(f'<path d="M{x + 122} 123 L{x + 138} 123" stroke="#334155" stroke-width="2.5" marker-end="url(#arrow)"/>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="780" height="260" viewBox="0 0 780 260">
  <defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#334155"/></marker></defs>
  <rect width="780" height="260" rx="24" fill="#f8fafc"/>
  <rect x="22" y="20" width="736" height="42" rx="16" fill="#e2e8f0"/>
  <text x="38" y="47" font-size="22" font-weight="800" fill="#0f172a">{title} methodology</text>
  <text x="38" y="70" font-size="13" fill="#64748b">Input evidence → modeling / reasoning core → output for governance or retrieval usage</text>
  {''.join(arrows)}{''.join(boxes)}
  <rect x="30" y="188" width="720" height="46" rx="16" fill="#ecfeff"/>
  <text x="390" y="216" text-anchor="middle" font-size="13" fill="#155e75">Figure generated for the bilingual daily paper radar; intended to mirror each paper's input-output story</text>
</svg>'''


def exp_svg(title, highlight):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="780" height="260" viewBox="0 0 780 260">
  <defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#64748b"/></marker></defs>
  <rect width="780" height="260" rx="24" fill="#ffffff"/>
  <rect x="24" y="22" width="732" height="216" rx="22" fill="#f1f5f9"/>
  <text x="46" y="55" font-size="21" font-weight="800" fill="#0f172a">{title} experiment highlight</text>
  <rect x="52" y="88" width="165" height="94" rx="18" fill="#dbeafe"/>
  <text x="134" y="120" text-anchor="middle" font-size="18" font-weight="800" fill="#1d4ed8">Setting</text>
  <text x="134" y="145" text-anchor="middle" font-size="12" fill="#334155">realistic benchmark</text>
  <text x="134" y="163" text-anchor="middle" font-size="12" fill="#334155">or production A/B</text>
  <rect x="304" y="76" width="172" height="118" rx="18" fill="#dcfce7"/>
  <text x="390" y="114" text-anchor="middle" font-size="19" font-weight="800" fill="#166534">Result</text>
  <text x="390" y="144" text-anchor="middle" font-size="11.5" fill="#334155">{highlight}</text>
  <rect x="560" y="88" width="165" height="94" rx="18" fill="#fef3c7"/>
  <text x="642" y="120" text-anchor="middle" font-size="18" font-weight="800" fill="#92400e">Value</text>
  <text x="642" y="145" text-anchor="middle" font-size="12" fill="#334155">content ecology /</text>
  <text x="642" y="163" text-anchor="middle" font-size="12" fill="#334155">creator governance</text>
  <path d="M219 136 C252 136 266 136 296 136" stroke="#64748b" stroke-width="3" fill="none" marker-end="url(#arrow)"/>
  <path d="M480 136 C514 136 528 136 552 136" stroke="#64748b" stroke-width="3" fill="none" marker-end="url(#arrow)"/>
</svg>'''


written = 0
for paper_id, (title, steps, highlight) in PAPERS.items():
    method_path = FIG_DIR / f"{paper_id}.svg"
    exp_path = FIG_DIR / f"{paper_id}_exp.svg"
    method_path.write_text(method_svg(title, steps), encoding="utf-8")
    exp_path.write_text(exp_svg(title, highlight), encoding="utf-8")
    written += 2

print(f"wrote {written} figure files")

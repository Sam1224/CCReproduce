from __future__ import annotations

import json
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent / "assets" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPERS_JSON = REPO_ROOT / "2026-08-28" / "papers.json"

PALETTE = ["#2563eb", "#0f766e", "#7c3aed", "#ea580c", "#0891b2"]


def _short_title(title: str, max_len: int = 18) -> str:
    base = (title or "").split(":")[0].strip() or title
    if len(base) <= max_len:
        return base
    return base[: max_len - 1] + "…"


def _shorten_text(text: str | None, max_len: int = 64) -> str:
    if not text:
        return ""
    compact = " ".join(text.replace("\n", " ").split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1] + "…"


def method_svg(title: str, steps: list[str]) -> str:
    boxes: list[str] = []
    arrows: list[str] = []

    safe_steps = steps[:5] if steps else ["Input", "Model", "Signal", "Decision", "Output"]

    for idx, step in enumerate(safe_steps):
        x = 18 + idx * 148
        color = PALETTE[idx % len(PALETTE)]
        boxes.append(
            f'<rect x="{x}" y="88" width="128" height="72" rx="16" fill="{color}" opacity="0.92"/>'
        )
        boxes.append(
            f'<text x="{x + 64}" y="117" text-anchor="middle" font-size="12" fill="white" font-weight="700">{step}</text>'
        )
        if idx < len(safe_steps) - 1:
            arrows.append(
                f'<path d="M{x + 132} 124 L{x + 146} 124" stroke="#334155" stroke-width="2" marker-end="url(#arrow)"/>'
            )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="820" height="240" viewBox="0 0 820 240">
  <defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#334155"/></marker></defs>
  <rect width="820" height="240" rx="24" fill="#f8fafc"/>
  <text x="32" y="42" font-size="22" font-weight="800" fill="#0f172a">{title} methodology</text>
  <text x="32" y="66" font-size="13" fill="#64748b">Input → core modeling / decision → output</text>
  {''.join(arrows)}{''.join(boxes)}
  <rect x="28" y="184" width="764" height="28" rx="14" fill="#e0f2fe"/>
  <text x="410" y="203" text-anchor="middle" font-size="13" fill="#075985">Daily paper radar schematic (auto-generated)</text>
</svg>'''


def exp_svg(title: str, highlight: str) -> str:
    safe_highlight = _shorten_text(highlight, 70) or "Key result / value"
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="820" height="240" viewBox="0 0 820 240">
  <rect width="820" height="240" rx="24" fill="#ffffff"/>
  <rect x="24" y="24" width="772" height="192" rx="20" fill="#f1f5f9"/>
  <text x="44" y="58" font-size="21" font-weight="800" fill="#0f172a">{title} experiment highlight</text>
  <rect x="50" y="92" width="180" height="82" rx="16" fill="#dbeafe"/>
  <text x="140" y="126" text-anchor="middle" font-size="16" font-weight="700" fill="#1d4ed8">Setup</text>
  <text x="140" y="149" text-anchor="middle" font-size="12" fill="#334155">benchmark / protocol</text>

  <rect x="320" y="78" width="180" height="110" rx="18" fill="#dcfce7"/>
  <text x="410" y="120" text-anchor="middle" font-size="18" font-weight="800" fill="#166534">Result</text>
  <text x="410" y="146" text-anchor="middle" font-size="12" fill="#334155">{safe_highlight}</text>

  <rect x="590" y="92" width="180" height="82" rx="16" fill="#fef3c7"/>
  <text x="680" y="126" text-anchor="middle" font-size="16" font-weight="700" fill="#92400e">Value</text>
  <text x="680" y="149" text-anchor="middle" font-size="12" fill="#334155">content / governance</text>

  <path d="M235 133 C285 133 295 133 312 133" stroke="#64748b" stroke-width="3" fill="none" marker-end="url(#arrow)"/>
  <path d="M505 133 C545 133 555 133 582 133" stroke="#64748b" stroke-width="3" fill="none" marker-end="url(#arrow)"/>
  <defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#64748b"/></marker></defs>
</svg>'''


def main() -> None:
    if not PAPERS_JSON.exists():
        raise SystemExit(f"missing: {PAPERS_JSON}")

    payload = json.loads(PAPERS_JSON.read_text(encoding="utf-8"))
    papers = payload.get("papers", [])

    written = 0
    for p in papers:
        pid = p.get("id")
        if not pid:
            continue

        title = _short_title(p.get("title", "Paper"))
        steps = p.get("figure_steps") or []
        highlight = (
            (((p.get("summary") or {}).get("zh") or {}).get("key_metrics"))
            or (((p.get("summary") or {}).get("en") or {}).get("key_metrics"))
            or ""
        )

        method_path = FIG_DIR / f"{pid}.svg"
        exp_path = FIG_DIR / f"{pid}_exp.svg"

        if not method_path.exists():
            method_path.write_text(method_svg(title, steps), encoding="utf-8")
            written += 1
        if not exp_path.exists():
            exp_path.write_text(exp_svg(title, highlight), encoding="utf-8")
            written += 1

    print(f"wrote {written} new figure files")


if __name__ == "__main__":
    main()

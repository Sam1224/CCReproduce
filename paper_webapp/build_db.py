import json
import os
import sqlite3
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = Path(__file__).resolve().parent


def iter_papers_json():
    for p in sorted(REPO_ROOT.glob("20??-??-??/papers.json")):
        yield p


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
          inspection_date TEXT NOT NULL,
          paper_id TEXT NOT NULL,
          title TEXT NOT NULL,
          authors TEXT,
          affiliations TEXT,
          source TEXT,
          published TEXT,
          links TEXT,
          tags TEXT,
          score_total INTEGER,
          score_breakdown TEXT,
          rationale_zh TEXT,
          method_overview_zh TEXT,
          method_overview_en TEXT,
          innovation_zh TEXT,
          innovation_en TEXT,
          key_metrics_zh TEXT,
          key_metrics_en TEXT,
          reproduce_url TEXT,
          figure_path TEXT,
          exp_figure_path TEXT,
          PRIMARY KEY (inspection_date, paper_id)
        );
        """
    )


def main() -> None:
    out_path = WEBAPP_DIR / "data" / "papers.sqlite"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        out_path.unlink()

    conn = sqlite3.connect(str(out_path))
    ensure_schema(conn)

    for pj in iter_papers_json():
        inspection_date = pj.parent.name
        with open(pj, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        if isinstance(loaded, list):
            papers = loaded
        elif isinstance(loaded, dict):
            papers = loaded.get("papers") or loaded.get("items") or []
        else:
            papers = []

        for p in papers:
            if not isinstance(p, dict):
                continue

            links = p.get("links", {})
            score = p.get("score", {})
            summary = p.get("summary", {})
            reproduce = p.get("reproduce", {})

            reproduce_url = None
            if reproduce and reproduce.get("path"):
                reproduce_url = f"https://github.com/Sam1224/CCReproduce/tree/main/{reproduce['path']}"

            fig_path_svg = WEBAPP_DIR / "assets" / "figures" / f"{p['id']}.svg"
            fig_path_png = WEBAPP_DIR / "assets" / "figures" / f"{p['id']}.png"
            if fig_path_svg.exists():
                figure_path = f"assets/figures/{p['id']}.svg"
            elif fig_path_png.exists():
                figure_path = f"assets/figures/{p['id']}.png"
            else:
                figure_path = None

            # experiment / results figure (optional, looked up as "{id}_exp.*")
            exp_path_svg = WEBAPP_DIR / "assets" / "figures" / f"{p['id']}_exp.svg"
            exp_path_png = WEBAPP_DIR / "assets" / "figures" / f"{p['id']}_exp.png"
            if exp_path_svg.exists():
                exp_figure_path = f"assets/figures/{p['id']}_exp.svg"
            elif exp_path_png.exists():
                exp_figure_path = f"assets/figures/{p['id']}_exp.png"
            else:
                exp_figure_path = None

            conn.execute(
                """
                INSERT OR REPLACE INTO papers (
                  inspection_date, paper_id, title, authors, affiliations, source, published,
                  links, tags, score_total, score_breakdown, rationale_zh,
                  method_overview_zh, method_overview_en, innovation_zh, innovation_en,
                  key_metrics_zh, key_metrics_en, reproduce_url, figure_path, exp_figure_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    inspection_date,
                    p["id"],
                    p.get("title"),
                    json.dumps(p.get("authors", []), ensure_ascii=False),
                    json.dumps(p.get("affiliations", []), ensure_ascii=False),
                    p.get("source"),
                    p.get("published"),
                    json.dumps(links, ensure_ascii=False),
                    json.dumps(p.get("tags", []), ensure_ascii=False),
                    int(score.get("total", 0)),
                    json.dumps(score, ensure_ascii=False),
                    score.get("rationale_zh"),
                    summary.get("zh", {}).get("method_overview"),
                    summary.get("en", {}).get("method_overview"),
                    summary.get("zh", {}).get("innovation"),
                    summary.get("en", {}).get("innovation"),
                    summary.get("zh", {}).get("key_metrics"),
                    summary.get("en", {}).get("key_metrics"),
                    reproduce_url,
                    figure_path,
                    exp_figure_path,
                ),
            )

    conn.commit()
    conn.close()

    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()

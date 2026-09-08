import json
import os
import sqlite3
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = Path(__file__).resolve().parent


def iter_papers_json():
    for p in sorted(REPO_ROOT.glob("20??-??-??/papers.json")):
        yield p


def summary_value(summary: dict, lang: str, key: str):
    if not isinstance(summary, dict):
        return None
    nested = summary.get(lang)
    if isinstance(nested, dict) and nested.get(key) is not None:
        return nested.get(key)
    return summary.get(f"{key}_{lang}")


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
          rationale_en TEXT,
          method_overview_zh TEXT,
          method_overview_en TEXT,
          story_zh TEXT,
          story_en TEXT,
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

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS days (
          inspection_date TEXT PRIMARY KEY,
          paper_count INTEGER NOT NULL
        );
        """
    )


def main() -> None:
    out_path = WEBAPP_DIR / "data" / "papers.sqlite"
    json_out_path = WEBAPP_DIR / "data" / "papers.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        out_path.unlink()

    conn = sqlite3.connect(str(out_path))
    ensure_schema(conn)

    web_days = []
    web_papers = []

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

        paper_count = 0
        for p in papers:
            if not isinstance(p, dict):
                continue

            paper_count += 1
            links = p.get("links", {})
            score = p.get("score", {})
            summary = p.get("summary", {})
            reproduce = p.get("reproduce", {})

            reproduce_url = None
            if reproduce and reproduce.get("url"):
                reproduce_url = reproduce["url"]
            elif reproduce and reproduce.get("path"):
                reproduce_url = f"https://github.com/Sam1224/CCReproduce/tree/main/{reproduce['path']}"

            fig_path_svg = WEBAPP_DIR / "assets" / "figures" / f"{p['id']}.svg"
            fig_path_png = WEBAPP_DIR / "assets" / "figures" / f"{p['id']}.png"
            if fig_path_svg.exists():
                figure_path = f"assets/figures/{p['id']}.svg"
            elif fig_path_png.exists():
                figure_path = f"assets/figures/{p['id']}.png"
            else:
                figure_path = None

            exp_path_svg = WEBAPP_DIR / "assets" / "figures" / f"{p['id']}_exp.svg"
            exp_path_png = WEBAPP_DIR / "assets" / "figures" / f"{p['id']}_exp.png"
            if exp_path_svg.exists():
                exp_figure_path = f"assets/figures/{p['id']}_exp.svg"
            elif exp_path_png.exists():
                exp_figure_path = f"assets/figures/{p['id']}_exp.png"
            else:
                exp_figure_path = None

            row = {
                "inspection_date": inspection_date,
                "paper_id": p["id"],
                "title": p.get("title"),
                "authors": p.get("authors", []),
                "affiliations": p.get("affiliations", []),
                "source": p.get("source"),
                "published": p.get("published"),
                "links": links,
                "tags": p.get("tags", []),
                "score_total": int(score.get("total", 0)),
                "score_breakdown": score,
                "rationale_zh": score.get("rationale_zh"),
                "rationale_en": score.get("rationale_en") or score.get("rationale_zh"),
                "method_overview_zh": summary_value(summary, "zh", "method_overview"),
                "method_overview_en": summary_value(summary, "en", "method_overview"),
                "story_zh": summary_value(summary, "zh", "story"),
                "story_en": summary_value(summary, "en", "story"),
                "innovation_zh": summary_value(summary, "zh", "innovation"),
                "innovation_en": summary_value(summary, "en", "innovation"),
                "key_metrics_zh": summary_value(summary, "zh", "key_metrics"),
                "key_metrics_en": summary_value(summary, "en", "key_metrics"),
                "reproduce_url": reproduce_url,
                "figure_path": figure_path,
                "exp_figure_path": exp_figure_path,
            }
            web_papers.append(row)

            conn.execute(
                """
                INSERT OR REPLACE INTO papers (
                  inspection_date, paper_id, title, authors, affiliations, source, published,
                  links, tags, score_total, score_breakdown, rationale_zh, rationale_en,
                  method_overview_zh, method_overview_en, story_zh, story_en,
                  innovation_zh, innovation_en, key_metrics_zh, key_metrics_en,
                  reproduce_url, figure_path, exp_figure_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["inspection_date"],
                    row["paper_id"],
                    row["title"],
                    json.dumps(row["authors"], ensure_ascii=False),
                    json.dumps(row["affiliations"], ensure_ascii=False),
                    row["source"],
                    row["published"],
                    json.dumps(row["links"], ensure_ascii=False),
                    json.dumps(row["tags"], ensure_ascii=False),
                    row["score_total"],
                    json.dumps(row["score_breakdown"], ensure_ascii=False),
                    row["rationale_zh"],
                    row["rationale_en"],
                    row["method_overview_zh"],
                    row["method_overview_en"],
                    row["story_zh"],
                    row["story_en"],
                    row["innovation_zh"],
                    row["innovation_en"],
                    row["key_metrics_zh"],
                    row["key_metrics_en"],
                    row["reproduce_url"],
                    row["figure_path"],
                    row["exp_figure_path"],
                ),
            )

        web_days.append({"inspection_date": inspection_date, "paper_count": paper_count})
        conn.execute(
            "INSERT OR REPLACE INTO days (inspection_date, paper_count) VALUES (?, ?)",
            (inspection_date, paper_count),
        )

    conn.commit()
    conn.close()

    web_payload = {
        "days": sorted(web_days, key=lambda item: item["inspection_date"], reverse=True),
        "papers": sorted(web_papers, key=lambda item: (item["inspection_date"], item["score_total"], item["title"]), reverse=True),
    }
    json_out_path.write_text(json.dumps(web_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote: {out_path}")
    print(f"wrote: {json_out_path}")


if __name__ == "__main__":
    main()

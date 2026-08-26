import json
import os
import sqlite3
import subprocess
from pathlib import Path
from urllib.parse import quote


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


def current_repo_branch() -> str:
    env_branch = os.environ.get("GITHUB_REF_NAME") or os.environ.get("BRANCH_NAME")
    if env_branch:
        return env_branch
    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True
        ).strip() or "main"
    except Exception:
        return "main"


def github_tree_url(rel_path: str, branch: str) -> str:
    clean_path = str(rel_path).strip().strip("/")
    if not clean_path:
        return None
    # Quote branch/path components so branch names like aime/xxx and paths with
    # spaces produce stable GitHub links without changing the repository model.
    branch_part = quote(branch or "main", safe="")
    path_part = quote(clean_path, safe="/")
    return f"https://github.com/Sam1224/CCReproduce/tree/{branch_part}/{path_part}"


def extract_day_metadata(loaded, fallback_date: str) -> dict:
    if not isinstance(loaded, dict):
        return {
            "inspection_date": fallback_date,
            "notes_zh": None,
            "notes_en": None,
            "window_start": None,
            "window_end": None,
        }
    window = loaded.get("window") or {}
    return {
        "inspection_date": loaded.get("inspection_date") or fallback_date,
        "notes_zh": loaded.get("notes_zh") or loaded.get("note_zh"),
        "notes_en": loaded.get("notes_en") or loaded.get("note_en"),
        "window_start": window.get("start") if isinstance(window, dict) else None,
        "window_end": window.get("end") if isinstance(window, dict) else None,
    }


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

    # Also persist the inspection dates themselves so the UI can show days
    # even when a given day has zero papers.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS days (
          inspection_date TEXT PRIMARY KEY,
          paper_count INTEGER NOT NULL,
          notes_zh TEXT,
          notes_en TEXT,
          window_start TEXT,
          window_end TEXT,
          repo_branch TEXT
        );
        """
    )


def main() -> None:
    out_path = WEBAPP_DIR / "data" / "papers.sqlite"
    json_out_path = WEBAPP_DIR / "data" / "papers.json"
    js_out_path = WEBAPP_DIR / "data" / "papers-inline.js"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        out_path.unlink()

    conn = sqlite3.connect(str(out_path))
    ensure_schema(conn)
    repo_branch = current_repo_branch()
    exported_days = []
    exported_papers = []

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

        day_meta = extract_day_metadata(loaded, inspection_date)
        inspection_date = day_meta["inspection_date"]

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
            if reproduce and reproduce.get("path"):
                reproduce_url = github_tree_url(reproduce["path"], repo_branch)

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

            paper_row = {
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
            exported_papers.append(paper_row)

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
                    paper_row["inspection_date"],
                    paper_row["paper_id"],
                    paper_row["title"],
                    json.dumps(paper_row["authors"], ensure_ascii=False),
                    json.dumps(paper_row["affiliations"], ensure_ascii=False),
                    paper_row["source"],
                    paper_row["published"],
                    json.dumps(paper_row["links"], ensure_ascii=False),
                    json.dumps(paper_row["tags"], ensure_ascii=False),
                    paper_row["score_total"],
                    json.dumps(paper_row["score_breakdown"], ensure_ascii=False),
                    paper_row["rationale_zh"],
                    paper_row["rationale_en"],
                    paper_row["method_overview_zh"],
                    paper_row["method_overview_en"],
                    paper_row["story_zh"],
                    paper_row["story_en"],
                    paper_row["innovation_zh"],
                    paper_row["innovation_en"],
                    paper_row["key_metrics_zh"],
                    paper_row["key_metrics_en"],
                    paper_row["reproduce_url"],
                    paper_row["figure_path"],
                    paper_row["exp_figure_path"],
                ),
            )

        day_row = {
            "inspection_date": inspection_date,
            "paper_count": paper_count,
            "notes_zh": day_meta.get("notes_zh"),
            "notes_en": day_meta.get("notes_en"),
            "window_start": day_meta.get("window_start"),
            "window_end": day_meta.get("window_end"),
            "repo_branch": repo_branch,
        }
        exported_days.append(day_row)

        conn.execute(
            """
            INSERT OR REPLACE INTO days (
              inspection_date, paper_count, notes_zh, notes_en, window_start, window_end, repo_branch
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                day_row["inspection_date"],
                day_row["paper_count"],
                day_row["notes_zh"],
                day_row["notes_en"],
                day_row["window_start"],
                day_row["window_end"],
                day_row["repo_branch"],
            ),
        )

    conn.commit()
    conn.close()

    payload = json.dumps(
        {"days": exported_days, "papers": exported_papers},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    json_out_path.write_text(payload, encoding="utf-8")
    js_out_path.write_text(f"window.__PAPER_DATA__ = {payload};\n", encoding="utf-8")

    print(f"wrote: {out_path}")
    print(f"wrote: {json_out_path}")
    print(f"wrote: {js_out_path}")


if __name__ == "__main__":
    main()

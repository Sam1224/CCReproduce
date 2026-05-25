import json
import os
import sqlite3
import time
import urllib.request
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SEED_DIR = os.path.join(DATA_DIR, "seed")
STATIC_DIR = os.path.join(BASE_DIR, "static")
THUMB_DIR = os.path.join(STATIC_DIR, "thumbnails")
DB_PATH = os.path.join(DATA_DIR, "paper_hub.sqlite3")


def _ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SEED_DIR, exist_ok=True)
    os.makedirs(THUMB_DIR, exist_ok=True)


def _connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uid TEXT NOT NULL,
            date TEXT NOT NULL,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            authors TEXT,
            affiliations TEXT,
            paper_url TEXT,
            pdf_url TEXT,
            code_url TEXT,
            project_url TEXT,
            tags_json TEXT,
            score INTEGER,
            score_reason_zh TEXT,
            score_reason_en TEXT,
            method_zh TEXT,
            method_en TEXT,
            innovation_zh TEXT,
            innovation_en TEXT,
            metrics_zh TEXT,
            metrics_en TEXT,
            thumbnail_url TEXT,
            local_thumbnail TEXT,
            created_at INTEGER NOT NULL,
            UNIQUE(date, uid)
        )
        """
    )
    conn.commit()


def _download_thumbnail(uid: str, url: str) -> Optional[str]:
    if not url:
        return None

    ext = ".png"
    if url.lower().endswith(".jpg") or url.lower().endswith(".jpeg"):
        ext = ".jpg"
    elif url.lower().endswith(".webp"):
        ext = ".webp"

    dst = os.path.join(THUMB_DIR, f"{uid}{ext}")
    if os.path.exists(dst):
        return f"/static/thumbnails/{uid}{ext}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "paper-hub/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            content = resp.read()
        with open(dst, "wb") as f:
            f.write(content)
        return f"/static/thumbnails/{uid}{ext}"
    except Exception:
        return None


def _load_seed_files(conn: sqlite3.Connection) -> None:
    if not os.path.isdir(SEED_DIR):
        return

    files = [
        os.path.join(SEED_DIR, name)
        for name in os.listdir(SEED_DIR)
        if name.endswith(".json")
    ]
    for path in sorted(files):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        items = payload.get("papers", []) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            continue
        _upsert_papers(conn, items)


def _upsert_papers(conn: sqlite3.Connection, papers: List[Dict[str, Any]]) -> None:
    cursor = conn.cursor()
    now = int(time.time())

    for p in papers:
        uid = str(p.get("uid") or p.get("arxiv_id") or p.get("openreview_id") or "").strip()
        date = str(p.get("date") or "").strip()
        title = str(p.get("title") or "").strip()
        source = str(p.get("source") or "").strip()
        if not uid or not date or not title or not source:
            continue

        thumb_url = str(p.get("thumbnail_url") or "").strip()
        local_thumb = _download_thumbnail(uid, thumb_url) if thumb_url else None

        cursor.execute(
            """
            INSERT OR IGNORE INTO papers (
                uid, date, source, title, authors, affiliations,
                paper_url, pdf_url, code_url, project_url, tags_json,
                score, score_reason_zh, score_reason_en,
                method_zh, method_en, innovation_zh, innovation_en,
                metrics_zh, metrics_en, thumbnail_url, local_thumbnail, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uid,
                date,
                source,
                title,
                str(p.get("authors") or ""),
                str(p.get("affiliations") or ""),
                str(p.get("paper_url") or ""),
                str(p.get("pdf_url") or ""),
                str(p.get("code_url") or ""),
                str(p.get("project_url") or ""),
                json.dumps(p.get("tags") or [], ensure_ascii=False),
                int(p.get("score") or 0),
                str(p.get("score_reason_zh") or ""),
                str(p.get("score_reason_en") or ""),
                str(p.get("method_zh") or ""),
                str(p.get("method_en") or ""),
                str(p.get("innovation_zh") or ""),
                str(p.get("innovation_en") or ""),
                str(p.get("metrics_zh") or ""),
                str(p.get("metrics_en") or ""),
                thumb_url,
                local_thumb or "",
                now,
            ),
        )

    conn.commit()


_ensure_dirs()
conn = _connect_db()
_init_schema(conn)

app = FastAPI(title="Paper Hub")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def _startup() -> None:
    _load_seed_files(conn)


@app.get("/v1/ping")
def ping_handler() -> str:
    return "ok"


@app.get("/api/dates")
def list_dates() -> Dict[str, Any]:
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT date FROM papers ORDER BY date DESC")
    rows = cursor.fetchall()
    return {"dates": [r[0] for r in rows]}


@app.get("/api/papers")
def list_papers(date: str = Query(..., description="YYYY-MM-DD")) -> Dict[str, Any]:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            id, uid, date, source, title, authors, affiliations,
            paper_url, pdf_url, code_url, project_url, tags_json,
            score, score_reason_zh, score_reason_en,
            method_zh, method_en, innovation_zh, innovation_en,
            metrics_zh, metrics_en, local_thumbnail
        FROM papers
        WHERE date = ?
        ORDER BY score DESC, id ASC
        """,
        (date,),
    )
    rows = cursor.fetchall()
    items: List[Dict[str, Any]] = []
    for r in rows:
        items.append(
            {
                "id": r["id"],
                "uid": r["uid"],
                "date": r["date"],
                "source": r["source"],
                "title": r["title"],
                "authors": r["authors"],
                "affiliations": r["affiliations"],
                "paper_url": r["paper_url"],
                "pdf_url": r["pdf_url"],
                "code_url": r["code_url"],
                "project_url": r["project_url"],
                "tags": json.loads(r["tags_json"] or "[]"),
                "score": r["score"],
                "score_reason_zh": r["score_reason_zh"],
                "score_reason_en": r["score_reason_en"],
                "method_zh": r["method_zh"],
                "method_en": r["method_en"],
                "innovation_zh": r["innovation_zh"],
                "innovation_en": r["innovation_en"],
                "metrics_zh": r["metrics_zh"],
                "metrics_en": r["metrics_en"],
                "thumbnail": r["local_thumbnail"] or "",
            }
        )

    return {"papers": items}


@app.get("/api/papers/{paper_id}")
def get_paper(paper_id: int) -> Dict[str, Any]:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="paper not found")

    return {k: row[k] for k in row.keys()}


# ---------------------------DO NOT EDIT CODE BELOW THIS LINE---------------------------------
# This is the entry point for the FastAPI application.
if __name__ == "__main__":
    port = int(os.environ.get("_BYTEFAAS_RUNTIME_PORT", 8000))
    config = uvicorn.Config("main:app", port=port, log_level="info", host=None)
    server = uvicorn.Server(config)
    server.run()
# --------------------------------------------------------------------------------------------

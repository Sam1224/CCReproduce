from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_WEBAPP = REPO_ROOT / "paper_webapp"
SOURCE_DATA = SOURCE_WEBAPP / "data" / "papers.json"
OUTPUT_DIR = REPO_ROOT / "paper_webapp_deploy"
CORE_FILES = ("index.html", "app.js", "styles.css")
SQL_JS_TAG = '    <script src="https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.10.2/sql-wasm.js" defer></script>\n'
OLD_LOADERS = """  const loaders = protocol === \"file:\"\n    ? [loadInlineData, loadJsonFallback, loadSqliteDatabase]\n    : [loadSqliteDatabase, loadJsonFallback, loadInlineData];"""
NEW_LOADERS = """  const loaders = protocol === \"file:\"\n    ? [loadInlineData, loadJsonFallback]\n    : [loadJsonFallback, loadInlineData];"""
OLD_ERROR = "加载失败，请检查 data/papers-inline.js / data/papers.sqlite / data/papers.json 是否可访问。"
NEW_ERROR = "加载失败，请检查 data/papers-inline.js / data/papers.json 是否可访问。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a slim deploy bundle for paper_webapp."
    )
    parser.add_argument(
        "--figure-date",
        default="2026-08-26",
        help="Only copy figure assets for this inspection date.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Output directory for the deploy bundle.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_allowed_ids(figure_date: str) -> set[str]:
    day_path = REPO_ROOT / figure_date / "papers.json"
    payload = load_json(day_path)
    papers = payload if isinstance(payload, list) else payload.get("papers", [])
    return {
        paper.get("id")
        for paper in papers
        if isinstance(paper, dict) and paper.get("id")
    }


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "assets" / "figures").mkdir(parents=True, exist_ok=True)


def write_core_files(output_dir: Path) -> None:
    for name in CORE_FILES:
        src = SOURCE_WEBAPP / name
        dst = output_dir / name
        text = src.read_text(encoding="utf-8")
        if name == "index.html":
            text = text.replace(SQL_JS_TAG, "")
        elif name == "app.js":
            text = text.replace(OLD_LOADERS, NEW_LOADERS)
            text = text.replace(OLD_ERROR, NEW_ERROR)
        dst.write_text(text, encoding="utf-8")


def maybe_copy_asset(output_dir: Path, rel_path: str | None) -> str | None:
    if not rel_path:
        return None
    src = SOURCE_WEBAPP / rel_path
    if not src.exists() or not src.is_file():
        return None
    dst = output_dir / "assets" / "figures" / src.name
    shutil.copy2(src, dst)
    return src.name


def normalize_figure_path(rel_path: str | None, copied_assets: set[str]) -> str | None:
    if not rel_path:
        return None
    name = Path(rel_path).name
    if name not in copied_assets:
        return None
    return f"assets/figures/{name}"


def build_bundle_data(
    source_payload: dict[str, Any], figure_date: str, output_dir: Path
) -> tuple[dict[str, Any], int]:
    allowed_ids = load_allowed_ids(figure_date)
    copied_assets: set[str] = set()
    source_papers = source_payload.get("papers") or []
    for paper in source_papers:
        if not isinstance(paper, dict):
            continue
        if paper.get("inspection_date") != figure_date:
            continue
        if paper.get("paper_id") not in allowed_ids:
            continue
        for key in ("figure_path", "exp_figure_path"):
            copied = maybe_copy_asset(output_dir, paper.get(key))
            if copied:
                copied_assets.add(copied)

    deploy_payload = {
        "days": source_payload.get("days") or [],
        "papers": [],
    }
    for paper in source_papers:
        if not isinstance(paper, dict):
            continue
        copied_paper = dict(paper)
        copied_paper["figure_path"] = normalize_figure_path(
            copied_paper.get("figure_path"), copied_assets
        )
        copied_paper["exp_figure_path"] = normalize_figure_path(
            copied_paper.get("exp_figure_path"), copied_assets
        )
        deploy_payload["papers"].append(copied_paper)
    return deploy_payload, len(copied_assets)


def write_data_files(output_dir: Path, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    (output_dir / "data" / "papers.json").write_text(body, encoding="utf-8")
    (output_dir / "data" / "papers-inline.js").write_text(
        f"window.__PAPER_DATA__ = {body};\n", encoding="utf-8"
    )


def count_files(directory: Path) -> int:
    return sum(1 for path in directory.rglob("*") if path.is_file())


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    source_payload = load_json(SOURCE_DATA)

    prepare_output_dir(output_dir)
    write_core_files(output_dir)
    deploy_payload, copied_asset_count = build_bundle_data(
        source_payload, args.figure_date, output_dir
    )
    write_data_files(output_dir, deploy_payload)

    print(f"output_dir={output_dir}")
    print(f"days={len(deploy_payload.get('days') or [])}")
    print(f"papers={len(deploy_payload.get('papers') or [])}")
    print(f"copied_figure_assets={copied_asset_count}")
    print(f"file_count={count_files(output_dir)}")


if __name__ == "__main__":
    main()

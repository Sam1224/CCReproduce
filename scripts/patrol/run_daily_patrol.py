from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date-dir", required=True, help="Date directory such as 2026-08-18")
    parser.add_argument("--verify-reproductions", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd or REPO_ROOT, check=True)


def load_papers(date_dir: Path) -> list[dict]:
    payload = json.loads((date_dir / "papers.json").read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("papers", [])


def verify_reproductions(date_dir: Path, papers: list[dict]) -> None:
    for paper in papers:
        reproduce = paper.get("reproduce") or {}
        if reproduce.get("status") != "implemented":
            continue
        rel_path = reproduce.get("path")
        if not rel_path:
            raise RuntimeError(f"implemented paper {paper['id']} is missing reproduce.path")
        workdir = REPO_ROOT / rel_path
        for required in ["README.md", "data.py", "model.py", "train.py", "test.py"]:
            if not (workdir / required).exists():
                raise FileNotFoundError(f"missing {required} in {workdir}")
        run([sys.executable, "train.py"], cwd=workdir)
        run([sys.executable, "test.py"], cwd=workdir)


def main() -> None:
    args = parse_args()
    date_dir = (REPO_ROOT / args.date_dir).resolve()
    if not date_dir.exists():
        raise FileNotFoundError(f"date directory not found: {date_dir}")
    if not (date_dir / "papers.json").exists():
        raise FileNotFoundError(f"missing papers.json in {date_dir}")

    papers = load_papers(date_dir)
    if args.verify_reproductions:
        verify_reproductions(date_dir, papers)

    run([sys.executable, "scripts/patrol/generate_assets.py", "--date-dir", str(date_dir)])
    run([sys.executable, "paper_webapp/build_db.py"])
    print(f"patrol build completed for {date_dir.name}")


if __name__ == "__main__":
    main()

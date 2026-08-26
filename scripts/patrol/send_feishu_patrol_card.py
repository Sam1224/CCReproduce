#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_MIN_SCORE = 40
DEFAULT_REPO_SLUG = "Sam1224/CCReproduce"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or send a concise Feishu interactive card from a date directory papers.json."
    )
    parser.add_argument("--date-dir", required=True, help="Date directory containing papers.json")
    parser.add_argument("--target-email", help="Feishu target email; required unless --dry-run is used")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help="Minimum score threshold")
    parser.add_argument("--web-url", help="Optional deployed page URL; supports {date} placeholder")
    parser.add_argument("--branch-name", help="Optional GitHub branch name for date directory links")
    parser.add_argument("--output", help="Optional path to write the generated send payload JSON")
    parser.add_argument("--dry-run", action="store_true", help="Only generate payload; do not resolve/send")
    parser.add_argument(
        "--print-payload",
        action="store_true",
        help="Print generated payload JSON to stdout (enabled automatically for --dry-run unless --quiet is set)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress informational stdout logs")
    return parser.parse_args()


def load_payload(date_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    papers_path = date_dir / "papers.json"
    if not papers_path.exists():
        raise FileNotFoundError(f"missing papers.json: {papers_path}")
    payload = json.loads(papers_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {}, payload
    if isinstance(payload, dict):
        papers = payload.get("papers")
        if isinstance(papers, list):
            return payload, papers
        raise ValueError(f"papers field is missing or not a list in {papers_path}")
    raise ValueError(f"unsupported papers.json format in {papers_path}")


def get_inspection_date(meta: dict[str, Any], date_dir: Path) -> str:
    for key in ("inspection_date", "patrol_date", "date"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            return value
    return date_dir.name


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return "；".join(normalize_text(item) for item in value if normalize_text(item))
    return " ".join(str(value).split())


def shorten_text(text: str, limit: int = 110) -> str:
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def get_score_value(paper: dict[str, Any]) -> float | None:
    score = paper.get("score")
    if isinstance(score, (int, float)):
        return float(score)
    if isinstance(score, dict):
        total = score.get("total")
        if isinstance(total, (int, float)):
            return float(total)
        numeric_values = [value for value in score.values() if isinstance(value, (int, float))]
        if numeric_values:
            return float(sum(numeric_values))
    breakdown = paper.get("score_breakdown")
    if isinstance(breakdown, dict):
        total = breakdown.get("total")
        if isinstance(total, (int, float)):
            return float(total)
        numeric_values = [value for value in breakdown.values() if isinstance(value, (int, float))]
        if numeric_values:
            return float(sum(numeric_values))
    return None


def get_paper_id(paper: dict[str, Any]) -> str:
    for key in ("id", "arxiv_id", "paper_id"):
        value = paper.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def get_paper_url(paper: dict[str, Any]) -> str | None:
    links = paper.get("links")
    if isinstance(links, dict):
        for key in ("paper", "abs", "arxiv", "pdf"):
            value = links.get(key)
            if isinstance(value, str) and value:
                return value
    for key in ("arxiv_abs", "paper_url", "url", "arxiv_pdf"):
        value = paper.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def get_summary_excerpt(paper: dict[str, Any]) -> str:
    nested_candidates = [
        ("summary", "zh", "story"),
        ("summary", "zh", "method_overview"),
        ("summary", "zh", "innovation"),
        ("summary_zh", "story"),
        ("summary_zh", "method"),
        ("summary_zh", "innovation"),
    ]
    for path in nested_candidates:
        value: Any = paper
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        text = normalize_text(value)
        if text:
            return shorten_text(text)
    flat_candidates = [
        "method_overview_zh",
        "innovation_zh",
        "score_rationale_zh",
        "summary_zh",
        "abstract_zh",
    ]
    for key in flat_candidates:
        text = normalize_text(paper.get(key))
        if text:
            return shorten_text(text)
    return "暂无中文摘要。"


def get_tags_text(paper: dict[str, Any]) -> str:
    tags = paper.get("tags")
    if isinstance(tags, list):
        joined = " / ".join(normalize_text(tag) for tag in tags if normalize_text(tag))
        return joined
    return normalize_text(tags)


def escape_lark_md(text: str) -> str:
    text = text.replace("\\", "\\\\")
    for char in ("[", "]", "(", ")", "*", "_", "`", "<", ">"):
        text = text.replace(char, f"\\{char}")
    return text


def load_filtered_papers(papers: list[dict[str, Any]], min_score: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for paper in papers:
        score = get_score_value(paper)
        if score is None or score < min_score:
            continue
        rows.append(
            {
                "paper": paper,
                "score": score,
                "paper_id": get_paper_id(paper),
                "paper_url": get_paper_url(paper),
                "summary": get_summary_excerpt(paper),
                "tags": get_tags_text(paper),
                "title": normalize_text(paper.get("title")) or get_paper_id(paper),
                "source": normalize_text(paper.get("source")) or "Unknown",
            }
        )
    rows.sort(key=lambda item: (-item["score"], item["paper_id"]))
    return rows


def render_web_url(web_url: str | None, inspection_date: str) -> str | None:
    if not web_url:
        return None
    if "{date}" in web_url:
        return web_url.format(date=inspection_date)
    return web_url


def guess_repo_http_url() -> str:
    default_url = f"https://github.com/{DEFAULT_REPO_SLUG}"
    git = shutil.which("git")
    if not git:
        return default_url
    try:
        result = subprocess.run(
            [git, "config", "--get", "remote.origin.url"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return default_url
    remote = result.stdout.strip()
    if not remote:
        return default_url
    remote = remote.removesuffix(".git")
    if remote.startswith("git@") and ":" in remote:
        host, path = remote.split(":", 1)
        if host.endswith("github.com"):
            return "https://github.com/" + path
    if remote.startswith("https://github.com/") or remote.startswith("http://github.com/"):
        return remote.replace("http://", "https://", 1)
    return default_url


def build_branch_dir_url(date_dir: Path, branch_name: str | None) -> str | None:
    if not branch_name:
        return None
    repo_url = guess_repo_http_url().rstrip("/")
    rel_path = date_dir.relative_to(REPO_ROOT).as_posix()
    return f"{repo_url}/tree/{branch_name}/{rel_path}"


def build_card(
    inspection_date: str,
    filtered_papers: list[dict[str, Any]],
    total_count: int,
    min_score: float,
    web_url: str | None,
    branch_dir_url: str | None,
) -> dict[str, Any]:
    title = f"论文巡检 · {inspection_date}"
    summary_line = f"阈值 ≥ {int(min_score) if float(min_score).is_integer() else min_score} · 入选 {len(filtered_papers)}/{total_count} 篇"
    elements: list[dict[str, Any]] = [
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": summary_line,
            },
        }
    ]

    buttons = []
    if web_url:
        buttons.append(
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "查看部署页"},
                "type": "default",
                "url": web_url,
            }
        )
    if branch_dir_url:
        buttons.append(
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "查看 GitHub 目录"},
                "type": "default",
                "url": branch_dir_url,
            }
        )
    if buttons:
        elements.append({"tag": "action", "actions": buttons})

    if not filtered_papers:
        elements.append(
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": "本次没有分数达到阈值的论文。",
                },
            }
        )
    else:
        for index, item in enumerate(filtered_papers, start=1):
            score = int(item["score"]) if float(item["score"]).is_integer() else round(item["score"], 1)
            title_text = escape_lark_md(item["title"])
            source_text = escape_lark_md(item["source"])
            summary_text = escape_lark_md(item["summary"])
            tags_text = escape_lark_md(item["tags"]) if item["tags"] else "无"
            lines = [
                f"**{index}. {title_text}**",
                f"分数：`{score}` · 来源：{source_text}",
                summary_text,
                f"标签：{tags_text}",
            ]
            if item["paper_url"]:
                lines.append(f"[论文链接]({item['paper_url']})")
            elements.append(
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": "\n".join(lines),
                    },
                }
            )
            if index != len(filtered_papers):
                elements.append({"tag": "hr"})

    return {
        "config": {"wide_screen_mode": True, "enable_forward": True},
        "header": {
            "template": "blue",
            "title": {"tag": "plain_text", "content": title},
        },
        "elements": elements,
    }


def build_send_payload(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "msg_type": "interactive",
        "content": json.dumps(card, ensure_ascii=False),
    }


def extract_open_id(raw_output: str) -> str:
    if "not visible to the bot" in raw_output.lower():
        raise RuntimeError("目标用户对机器人不可见，无法发送飞书消息")
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        parsed = None

    def walk(value: Any) -> str | None:
        if isinstance(value, dict):
            for key in ("open_id", "openId", "openID", "user_id", "userId"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.startswith("ou_"):
                    return candidate
            for nested in value.values():
                found = walk(nested)
                if found:
                    return found
        elif isinstance(value, list):
            for nested in value:
                found = walk(nested)
                if found:
                    return found
        elif isinstance(value, str):
            match = re.search(r"\bou_[A-Za-z0-9]+\b", value)
            if match:
                return match.group(0)
        return None

    found = walk(parsed if parsed is not None else raw_output)
    if found:
        return found
    raise RuntimeError(f"无法从用户信息结果中解析 open_id: {raw_output}")


def resolve_open_id_from_email(email: str) -> str:
    helper = WORKSPACE_ROOT / "inner_skills" / "lark" / "mcp_lark_lark_user_info.py"
    if not helper.exists():
        raise FileNotFoundError(f"missing helper script: {helper}")
    cmd = [sys.executable, str(helper), json.dumps({"emails": [email]}, ensure_ascii=False)]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return extract_open_id(result.stdout.strip())


def send_card_with_lark_cli(open_id: str, send_payload: dict[str, Any]) -> dict[str, Any]:
    lark_cli = shutil.which("lark-cli")
    if not lark_cli:
        raise FileNotFoundError("lark-cli not found in PATH")
    cmd = [
        lark_cli,
        "im",
        "+messages-send",
        "--as",
        "bot",
        "--user-id",
        open_id,
        "--msg-type",
        send_payload["msg_type"],
        "--content",
        send_payload["content"],
        "--format",
        "json",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def write_payload_if_needed(payload: dict[str, Any], output_path: str | None) -> None:
    if not output_path:
        return
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.dry_run and not args.target_email:
        raise SystemExit("--target-email is required unless --dry-run is used")

    date_dir = (REPO_ROOT / args.date_dir).resolve()
    if not date_dir.exists():
        raise FileNotFoundError(f"date directory not found: {date_dir}")
    if REPO_ROOT not in date_dir.parents and date_dir != REPO_ROOT:
        raise ValueError(f"date directory must be inside repo: {date_dir}")

    meta, papers = load_payload(date_dir)
    inspection_date = get_inspection_date(meta, date_dir)
    filtered_papers = load_filtered_papers(papers, args.min_score)
    web_url = render_web_url(args.web_url, inspection_date)
    branch_dir_url = build_branch_dir_url(date_dir, args.branch_name)
    card = build_card(
        inspection_date=inspection_date,
        filtered_papers=filtered_papers,
        total_count=len(papers),
        min_score=args.min_score,
        web_url=web_url,
        branch_dir_url=branch_dir_url,
    )
    send_payload = build_send_payload(card)
    write_payload_if_needed(send_payload, args.output)

    should_print = args.print_payload or (args.dry_run and not args.quiet)
    if should_print:
        print(json.dumps(send_payload, ensure_ascii=False, indent=2))

    if args.dry_run:
        return

    open_id = resolve_open_id_from_email(args.target_email)
    response = send_card_with_lark_cli(open_id, send_payload)
    if not args.quiet:
        print(json.dumps({"target_email": args.target_email, "open_id": open_id, "response": response}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

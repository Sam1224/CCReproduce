#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def trim(text, limit=220):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def text_md(content, tag="lark_md"):
    return {"tag": "markdown", "content": content, "text_align": "left", "text_size": "normal"}


def build_card(papers_path: Path, min_score: int, web_url: str | None):
    payload = json.loads(papers_path.read_text(encoding="utf-8"))
    papers = [p for p in payload["papers"] if p["score"]["total"] >= min_score]
    elements = []
    if web_url:
        elements.append({
            "tag": "button",
            "text": {"tag": "plain_text", "content": "打开今日 Paper Web 应用"},
            "type": "primary",
            "width": "default",
            "behaviors": [{"type": "open_url", "default_url": web_url, "pc_url": web_url, "ios_url": web_url, "android_url": web_url}],
            "margin": "0px 0px 12px 0px",
        })
    if not papers:
        elements.append(text_md("今日未发现评分 ≥40 的相关新论文。"))
    for index, paper in enumerate(papers, start=1):
        zh = paper["summary"]["zh"]
        links = paper.get("links", {})
        code = paper.get("reproduce", {}).get("github_url") or paper.get("reproduce", {}).get("official_code") or paper.get("reproduce", {}).get("path", "")
        block = (
            f"**{index}. 📄 标题：** [{paper['title']}]({links.get('abs', '')})\n"
            f"**👥 作者：** {', '.join(paper.get('authors', [])[:8])}\n"
            f"**🏢 单位：** {', '.join(paper.get('affiliations', [])[:4])}\n"
            f"**🏷️ 标签：** {', '.join(paper.get('tags', []))} ｜ **日期：** {paper.get('published', '')} ｜ **得分：** {paper['score']['total']}/100\n"
            f"**🔗 链接：** {links.get('abs', '')}\n"
            f"**📝 方法概述：** {trim(zh.get('method_overview', ''), 260)}\n"
            f"**💡 创新性分析：** {trim(zh.get('innovation', ''), 220)}\n"
            f"**📊 关键指标：** {trim(zh.get('key_metrics', ''), 220)}\n"
            f"**🧪 代码：** {code or '未提供'}"
        )
        elements.append(text_md(block))
        if index != len(papers):
            elements.append({"tag": "hr"})
    card = {
        "schema": "2.0",
        "config": {"update_multi": True, "style": {"text_size": {"normal_v2": {"default": "normal", "pc": "normal", "mobile": "normal"}}}},
        "header": {
            "title": {"tag": "plain_text", "content": f"电商内容生态&达人治理 Paper 巡检｜{payload['inspection_date']}"},
            "subtitle": {"tag": "plain_text", "content": f"评分≥{min_score} 共 {len(papers)} 篇；高分论文已复现或核验官方代码"},
            "template": "blue",
            "padding": "12px 12px 12px 12px",
        },
        "body": {"direction": "vertical", "padding": "12px 12px 12px 12px", "elements": elements},
    }
    return card


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--min-score", type=int, default=40)
    parser.add_argument("--web-url")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    card = build_card(Path(args.papers), args.min_score, args.web_url)
    command = [
        "lark-cli", "im", "+messages-send",
        "--as", "bot",
        "--user-id", args.user_id,
        "--msg-type", "interactive",
        "--content", json.dumps(card, ensure_ascii=False),
        "--idempotency-key", f"paper-{Path(args.papers).parent.name}",
    ]
    if args.dry_run:
        command.append("--dry-run")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

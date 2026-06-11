# LLM-Based User Personas (Engineering Prototype)

- Paper: **LLM-Based User Personas for Recommendations at Scale**
- arXiv: https://arxiv.org/abs/2606.12198

## What’s implemented

原论文的核心贡献更偏“工业系统设计 + 大规模线上实验”。本目录提供一个 **可运行的工程化原型**，把论文里的关键链路拆成可替换模块：

- **用户历史聚类**（embedding-based clustering）：将 watch history 按语义聚成若干簇。
- **Persona 生成接口**：
  - `RuleBasedPersonaGenerator`：不依赖 LLM 的可运行 baseline（从标题/主题统计生成“总结兴趣 + 探索兴趣”）。
  - `LLMPersonaGenerator`：保留论文中的 LLM 生成接口位置（此处用伪代码/注释说明，实际可替换为公司内部/开源 LLM）。
- **异步生成服务骨架**：用 `asyncio.Queue` 模拟“异步 persona 推理”把生成从在线请求中解耦（对应论文的 asynchronous inference）。
- **基于 persona 的候选召回示例**：把 persona 映射回 item space（用 embedding 近邻检索）输出候选列表。

## Limitations

- 原论文包含 teacher→student 蒸馏、量化、线上安全机制与真实召回/排序链路；此处保留接口与工程结构，但不包含大模型蒸馏训练与线上系统依赖。

## Quickstart

```bash
cd 2026-06-11/LLMUserPersonas
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python demo.py
```

你会看到：

- 某个用户的若干 persona（总结兴趣 + 探索兴趣，含置信度/证据视频）
- 对应 persona 引导的候选召回结果（top-K videos）

# OmniRetrieval（论文复现目录）

- Paper: **OmniRetrieval: Unified Retrieval across Heterogeneous Knowledge Sources**
- ArXiv: https://arxiv.org/abs/2605.29250
- Upstream Code（作者仓库，已确认包含 `src/`、`main.py`、`evaluate.py` 等实现）：https://github.com/JinheonBaek/OmniRetrieval

## 为什么这里不重复实现
论文已提供较完整的官方实现，因此本目录以“可复用说明 + 运行入口整理”为主，不再重写一份可能与官方不一致的代码。

## 快速复现（建议）
```bash
git clone https://github.com/JinheonBaek/OmniRetrieval.git
cd OmniRetrieval
pip install -r requirements.txt
python main.py --demo
```

## 与电商内容生态/达人治理的结合点（思路）
OmniRetrieval 的核心价值是把 **文档语料（policy/公告/规则）**、**结构化表（商品/订单/达人画像）**、**知识图谱（实体关系/违规链路）**、**属性图（社交关系/供应链关系）** 通过同一层“编排”统一检索出来，并且保持每类数据的原生查询能力（SQL/SPARQL/Cypher/向量检索）。

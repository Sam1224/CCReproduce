# CMTA: Leveraging Cross-Modal Temporal Artifacts for Generalizable AI-Generated Video Detection

- **arXiv**: https://arxiv.org/abs/2605.00630
- **代码仓库**: https://github.com/hwang-cs-ime/CMTA  ⚠️ **空仓库**, 仅 README + LICENSE，需要复现
- **作者**: Hang Wang, Chao Shen, Chenhao Lin, Minghui Yang, Lei Zhang, Cong Wang
- **领域标签**: `AIGC Detection` `VLM` `Video Forensics` `内容治理` `Cross-Modal`
- **桶位**: STRONG

## 方法概述
CMTA 提出 AIGV 检测领域一个新的 fingerprint — **Cross-Modal Temporal Artifact**：真实视频的视觉-文本对齐随语义自然波动，而 AI 生成视频被 prompt 约束，跨模态对齐"过于稳定"。CMTA 用 BLIP 给每帧打 caption，CLIP 把 (帧, caption) 编码到共享空间；GRU 分支建模粗粒度时间对齐稳定度，Transformer 分支建模细粒度帧间变化，两路融合做二分类。

## 故事
现有 AIGV 检测 → **只看 uni-modal 或时空伪影**，忽视跨模态对齐随时间的稳定性 → CMTA 把"prompt-induced 语义稳态"作为新指纹，用双路时序建模捕获。

## 创新性分析
- 将"跨模态对齐稳定性"作为 AIGV 指纹是一个有论文级别新颖度的观察，跟现有 BNet、F3Net 这类时空伪影路线正交。
- BLIP→CLIP→双时序分支 的组合是经典模块的新组装，但概念层面的贡献明确。
- 跨生成器泛化是 AIGV 检测里最难的指标，文章把它放在 headline 上。

## 关键指标
- 数据集: **GenVideo / EvalCrafter / VideoPhy / VidProM**, 共 4 个大规模集 / 40 个子集
- 在所有子集设新 SOTA，cross-generator 泛化优势显著 (论文摘要未给具体百分点)

## 评分 (满分 100)
| 维度 | 分 / 满分 | 理由 |
|------|----------|------|
| 创新性 | 23 / 30 | 跨模态时序稳态 = 新观察，框架是经典模块组合 |
| 实验指标 | 13 / 15 | 4 数据集 / 40 子集 SOTA |
| 实验质量 | 13 / 15 | 数据集广度强；ablation 待论文细看 |
| 效率 | 5 / 10 | BLIP+CLIP 的双 backbone 推理代价高 |
| 泛化性 | 5 / 5 | 显式优化跨生成器泛化 |
| 相关性 | 22 / 25 | AIGC 视频在电商达人内容中爆发，对违规/虚假治理直接相关 |
| **合计** | **81** | ✅ ≥80，进入复现 (官方仓库空) |

## 复现状态
- `code/CMTA/`: PyTorch 玩具实现，包括 BLIP/CLIP feature 提取占位、双分支 (GRU + Transformer) 建模、二分类头、训练 + 推理脚本，附 toy data。
- 大模型 backbone 用 stub，留 TODO 接真实 BLIP / CLIP。

# PaSBench-Video: A Streaming Video Benchmark for Proactive Safety Warning

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | PaSBench-Video: A Streaming Video Benchmark for Proactive Safety Warning |
| **Authors** | Yusong Zhao, Yuejin Xie, Youliang Yuan, Junjie Hu, Jitian Guo, Yujiu Yang, Pinjia He |
| **Affiliations** | The Chinese University of Hong Kong, Shenzhen; Tsinghua University |
| **Link** | https://arxiv.org/abs/2606.02443 |
| **Submission Date** | 2026-06-01 (GMT+8: 2026-06-02 listing) |
| **Domain Bucket** | **STRONG** — proactive video safety monitoring, direct relevance to content governance |
| **Total** | **73 / 100** |

---

## 方法概述 / Method Overview

### 问题背景 (Problem)
现有视频基准测试普遍存在三大缺陷：(1) 依赖静态输入而非在线流式观察；(2) 忽略告警时机的精确性（何时发出告警）；(3) 对"无风险视频"缺乏误报率衡量。因此，没有任何基准能够有效测试多模态大语言模型（MLLM）能否在**危险发生前**（风险显现到事故发生的窗口期）主动发出告警。

> X is insufficient: No benchmark captures whether MLLMs can serve as always-on safety monitors that warn **before** an accident occurs in streaming video.

### 设计 (Design)
PaSBench-Video 构建了 740 段流式视频（481 段风险视频 + 259 段无风险视频），覆盖四大场景：
- **Driving**（自动驾驶危险预警）
- **Healthcare**（医疗操作安全）
- **Daily Life**（日常生活意外）
- **Industrial Production**（工业生产安全）

关键注解：每段风险视频均标注**帧级风险发生时刻**和**事故边界**，模型只能观察当前及历史帧（因果约束，不能"偷看"未来帧）。

**三级递进评测指标：**
1. **Detection Hit** — 是否在事故前发出任何告警
2. **Temporally-Calibrated Hit Rate** — 告警是否在合理的时间窗口内发出
3. **Content-Grounded Accuracy** — 告警内容是否准确描述了危险类型（最严格）

无风险视频用于衡量**误报率（False Positive Rate, FPR）**。

---

## 故事弧 / Story Arc

> *"Between the first visible sign of danger and the moment an accident occurs, there is often a window where intervention remains possible — yet no benchmark tests whether MLLMs can serve as always-on safety monitors during this window."*

现有基准只问"是否发生了危险"，PaSBench-Video 问"**何时**、**用什么理由**告警"。
这与内容治理平台中的直播安全监控场景高度契合：平台不仅需要检测是否存在违规，更需要**在关键时刻及时、准确地干预**。

---

## 创新点 / Innovation

1. **流式评估协议（Streaming Protocol）**：模型无法预知未来帧，必须实时决策，与真实部署场景一致。
2. **时间校准指标**：不仅考察"发现了吗"，还考察"发现得及时吗"，是业界首次系统设计该类指标。
3. **误报测量**：259 段无风险视频确保基准不被"全告警"策略刷分。
4. **跨域设计**：4 个垂直场景揭示了模型依赖场景活动特征（表面线索）而非真正推理风险。

与已有工作比较：
| 基准 | 流式 | 时间精度 | FPR测量 | 多场景 |
|------|------|----------|---------|--------|
| Video-SafetyBench | ❌ | ❌ | ❌ | ❌ |
| StreamingBench | ✅ | ❌ | ❌ | 部分 |
| **PaSBench-Video** | ✅ | ✅ | ✅ | ✅ |

---

## 关键指标 / Key Metrics

| 模型 | 最严格指标 (Content-Grounded Acc) | FPR |
|------|----------------------------------|-----|
| **最优 MLLM（未披露名称）** | < 20% | 与 Recall 正相关 (r=0.64) |
| Daily Life 域（最佳场景） | 相对较高 | 可接受 |
| Driving 域 | 低 | 高（模型误将正常驾驶场景告警）|

**结论**：Pearson r = 0.64，模型无法在低误报率前提下维持高召回率，表明当前 MLLM 依赖表面场景线索而非真正理解"风险是否会升级为伤害"。

---

## 打分明细 / Scoring Breakdown

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 22 | 30 | 填补流式主动安全告警空白；时间精度指标首创 |
| Experimental SOTA Delta | 8 | 15 | 基准论文，展示所有模型均失败（<20%），无直接SOTA提升 |
| Experimental Quality/Ablations | 13 | 15 | 740视频、13 MLLM、4域、三级指标、无风险集 |
| Efficiency | 3 | 10 | 基准论文，不适用 |
| Generalization | 4 | 5 | 4场景，有一定泛化性 |
| Domain Relevance | 23 | 25 | 直播流式内容安全、实时违规预警，与内容治理高度相关 |
| **Total** | **73** | **100** | |

---

## 与电商/内容治理的关联

PaSBench-Video 的"流式主动安全告警"框架可直接迁移至：
- **直播平台违规预警**：在主播违规行为升级为严重事件前及时告警
- **短视频内容审核**：逐帧分析而非事后全片检测，降低延迟
- **商品展示合规**：实时监控达人直播中的违规商品展示、虚假宣传等

关键发现：现有 MLLM 不足以胜任该任务，为专用内容治理模型（需时间感知 + 规则理解能力）提供了明确的研究方向。

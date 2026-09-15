# SyntheticUsersDR（toy but runnable）

复现目录：`CCReproduce/2026-09-15/SyntheticUsersDR`

目标论文：**When Can You Trust Your Synthetic Users? Diagnostics and Corrections for LLM Consumer Panels**

本目录提供一个 **toy-but-runnable** 的最小实现，用来演示：

- **合成用户面板（synthetic panel）** vs **人类用户样本（human sample）**
- 两类分布偏移：
  - **Covariate shift**：`P_s(X) != P_h(X)`（合成用户画像分布和真实人群不同）
  - **Concept shift**：`P_s(Y|X,T) != P_h(Y|X,T)`（合成用户对处理/产品的反应机制不同）
- 诊断指标：
  - covariate shift：用域分类器（propensity）区分 synthetic/human 的 **AUC**，以及重要性权重的 ESS
  - concept shift：synthetic outcome 模型在 human 上的 **泛化损失差（gap）**
- 纠偏：
  - **IPW**（importance weighting）纠 covariate shift
  - **AIPW / doubly-robust（DR）**：用 *weighted synthetic model term* + *human augmentation term* 同时处理 covariate shift + concept shift
- “信任/纠偏决策”：根据诊断阈值决定是直接信任 naive synthetic 估计，还是采用 DR 纠偏

> 备注：这里的 `T` 可以理解成“产品版本/策略/曝光处理（0/1）”；`Y` 是用户满意（0/1）。

---

## 文件说明

- `data.py`：生成 toy 数据（synthetic/human），内置 covariate/concept shift，并提供 human 真实 ATE 的计算函数。
- `model.py`：三个小网络
  - `PropensityNet`：域分类器，估计 `e(x)=P(S=synthetic|x)`
  - `OutcomeNet`：synthetic outcome 模型，预测 `p(y=1|x,t)`
  - `BiasNet`：用少量 human 标注样本拟合残差 `y - p_hat`（concept shift 校正）
- `train.py`：训练 propensity/outcome/bias 三个模块并保存到 `artifacts/ckpt.pt`
- `test.py`：在 held-out toy data 上输出 `naive vs IPW vs DR(AIPW)` 的 ATE 误差对比，并给出“是否信任 synthetic”的决策

---

## 运行方法

在本目录下执行：

```bash
python train.py
python test.py
```

期望输出（示例）：

- covariate shift AUC（越高越偏）
- concept shift gap（synthetic 模型在 human 上更差）
- true ATE vs naive / IPW / DR(AIPW) 的估计与绝对误差
- 最终决策：trust naive or apply correction

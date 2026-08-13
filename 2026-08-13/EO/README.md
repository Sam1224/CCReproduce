# EO toy PyTorch reproduction

本目录复现论文 **Dynamic Governance of Multi-LLM Agent Systems for Collaborative Conversational Outcomes** 的一个可运行 toy 版本，核心目标是把论文里的 **Contextual Bandit + POMDP belief tracker + PID controller** 组合成简化的 Experience Orchestrator（EO）。

## 文件说明

- `data.py`：合成 visitor session 数据；包含 context vector、隐藏 intent state、4 个 content arms、resistance / friction 的环境转移，以及闭环评测逻辑。
- `model.py`：`ExperienceOrchestrator`，由 GRU belief tracker、bandit/policy head、PID 控制项组成；同时提供 naive baseline。
- `train.py`：用 oracle 轨迹做 toy 行为克隆训练，并在闭环仿真里验证 advisor contact 指标。
- `test.py`：加载 checkpoint，对比 naive baseline 与 EO，输出 `advisor_contact_rate / genuine_contact_rate / avg_resistance_drop`。

## 与原论文一致的部分

- 保留了论文最核心的系统分层：
  1. **belief tracker** 维护访客潜在意图；
  2. **contextual bandit / policy** 选择 4 个内容 arm；
  3. **PID controller** 根据 resistance 偏差调节 CTA 强度。
- 数据生成里显式模拟了：
  - session context vector；
  - hidden visitor intent / belief；
  - 4 个内容动作臂；
  - resistance 与 friction 的动态演化；
  - advisor contact 与 genuine contact 两类 outcome。
- 测试时采用闭环 rollout，而不是只看离线分类准确率，更贴近论文“governed trajectory”的评估口径。

## 这里做的简化

- 原论文是 LLM-to-LLM financial-services conversation simulation；这里改成了 **可控的合成环境**，不直接调用 LLM。
- belief tracker 用 GRU 追踪连续 latent intent，而不是完整的文本级 POMDP 观测模型。
- contextual bandit 用一个小型 MLP policy head 近似；训练方式是 oracle policy 的 toy 行为克隆，不做真实在线 bandit 更新。
- PID controller 不对生成 schema 做硬约束，而是通过 **对 4 个 arm 的 logit bias** 实现“过高 resistance 时抑制 advisor CTA、鼓励降阻内容”的控制效果。
- genuine contact 在 toy 环境中定义为“成功 contact 且 contact 时 readiness 足够高、resistance 足够低”，用于近似论文中的 high-intent conversion。

## 4 个 content arms

1. `educate`：解释产品/流程，主要降低 `info_need` 和部分 friction。
2. `empathize`：缓和顾虑，主要降低 `trust_need` 与 resistance。
3. `social_proof`：展示案例/背书，适合中后期把 readiness 往 contact 推进。
4. `advisor_cta`：直接推动联系 advisor；时机合适时转化最高，过早会抬升 resistance。

## 运行方式

```bash
cd CCReproduce/2026-08-13/EO
python train.py --epochs 8 --train-sessions 1200 --eval-sessions 256
python test.py --checkpoint eo_toy.pt --eval-sessions 384
```

## 预期现象

- naive baseline 能完成一部分高 readiness session，但在高 resistance / 高 friction 访客上容易过早或过晚 CTA。
- EO 通过 belief + PID 的组合，通常会带来更高的 `advisor_contact_rate`、更高的 `genuine_contact_rate`，以及更明显的 `avg_resistance_drop`。

# Xuanwu_VL_2B

面向论文 **Xuanwu: Evolving General Multimodal Models into an Industrial-Grade Foundation for Content Ecosystems** 的可运行 toy reproduction。

本目录不尝试复现工业级 Xuanwu 全量模型，而是在一个很小的 PyTorch 示例中保留论文核心故事：从通用多模态底座出发，加入内容治理需要的细粒度感知、OCR 对抗鲁棒性与轻量部署接口。

## 复现内容

- **通用多模态底座**：`VisualEncoder + TokenEncoder + GatedFusion` 模拟视觉、正文、OCR 三路输入的通用 VLM 表征。
- **细粒度感知**：除了 coarse policy 分类，还预测 `red_region / edge_tool / qr_grid / money_word / skin_region / safe_scene` 等细粒度证据。
- **对抗 OCR 鲁棒性**：toy 数据同时提供 clean OCR 与 adversarial OCR，例如 `knife -> k1nife`、`money -> m0ney`、`adult -> ad ult`；post 阶段加入 clean/adv 一致性损失。
- **三阶段训练接口**：`--stage pre|mid|post|all` 分别模拟底座预热、治理中训、鲁棒后训。
- **轻量部署**：模型只有几十万参数，并提供 `deploy_head` 蒸馏式轻量输出，`test.py` 会打印参数量与 deploy accuracy。

## 文件说明

- `dataset.py`：合成多模态内容治理数据集，包含图片、正文 token、OCR token、对抗 OCR token、粗粒度标签与细粒度证据标签。
- `model.py`：轻量视觉编码器、OCR/text encoder、门控融合模块、三阶段 loss 与参数量统计。
- `train.py`：训练脚本，支持 `pre/mid/post/all` stage 参数。
- `test.py`：评估 clean accuracy、adversarial OCR accuracy、robust gap、fine F1、deploy accuracy 与模态权重。

## 运行方式

```bash
python train.py --stage all --epochs 3 --train-samples 192 --output xuanwu_vl_2b_toy.pt
python test.py --checkpoint xuanwu_vl_2b_toy.pt --test-samples 96
```

快速 smoke test 可以使用更小配置：

```bash
python train.py --stage all --epochs 1 --train-samples 32 --batch-size 8 --hidden-dim 32 --embed-dim 24 --output /tmp/xuanwu_smoke.pt
python test.py --checkpoint /tmp/xuanwu_smoke.pt --test-samples 16 --batch-size 8
```

## 预期现象

训练后通常可以观察到：

- clean 与 adversarial OCR accuracy 都能正常输出，`robust_gap` 用来衡量 OCR 扰动导致的性能下降；
- `fine_f1` 反映模型是否学到局部证据，而不是只靠类别先验；
- `modality_weight` 展示门控融合对视觉、正文与 OCR 的动态依赖；
- `deploy_acc` 展示轻量部署头在 toy 场景下可直接推理。

## 与原论文的差异

- 原论文是工业级多模态基础模型，这里只用合成数据和小 CNN/GRU 保留结构与训练故事。
- 原论文包含大规模真实内容生态与复杂安全策略，本 toy 使用四类内容治理标签：`safe / violence / scam / adult`。
- OCR 鲁棒性以可控字符串扰动模拟，便于本地 CPU 快速运行。
- 轻量部署通过参数量统计与小型 deploy head 表示，不涉及真实量化、编译或线上服务。

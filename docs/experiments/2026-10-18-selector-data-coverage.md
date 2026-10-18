# 2026-10-18：固定更新预算，扩大 selector 训练图片覆盖

## 目的与预先固定的设计

从 I4 的教师拟合泛化差距和 I5 的有效任务依赖出发，检验增加不同训练图片、保持训练更新次数不变，是否提高条件化 selector 相对图像版的优势。采用 [预先固定的协议](../selector-data-coverage.md) 和 `configs/pets_selector_scale.json`。这对应 proposal H3；任务交换是 H1 与 H3 的衔接，本轮不增加 H2 结论。

固定 CLIP ViT-B/16、block 6、49/196 个 patch、同一 348,033 参数架构和软重要性 KL 目标。592 张旧训练图片严格嵌套于 2,944 张全部训练图片；两组都是 532 个训练类别组合。覆盖 × 模型 × 三种子共 12 次训练，各 3,990 更新、batch 128。任务／类别各出现 480 次；每种子四次运行共享初始化和任务／类别／图片分位数序列，同覆盖的两模型共享实际图片序列。旧小覆盖模型用新采样方式重训，避免将采样方式变化混入规模对照。固定最终步 checkpoint。

完整开发评估包含 736 张图片。未见的 66 个组合涉及其中 716 张图、2,624 个图片／任务样本；已见的 532 个组合涉及全部 736 张图、21,162 个样本。原有未见面板为 144 张图／528 个样本，新增未见面板为 572 张图／2,096 个样本。最终测试图片与 68 个测试组合保持预留。

主效应为 `(条件化 − 图像版)_大覆盖 − (条件化 − 图像版)_小覆盖`。每个任务和任务内类别等权；三种子先平均，再进行 2,000 次配对图片聚类 bootstrap，seed 2052。图片等权、面板拆分、已见组合和任务交换均单独标为辅助分析。

## 执行来源与成本

运行服务器目录 `/data2/zhao.tong/experiments`，单张 RTX A6000（physical GPU 4），Python 3.10.12、PyTorch 2.6.0+cu124、NumPy 1.26.4。代码先提交再运行。

| 阶段 | 代码来源 | 输出目录 | 进程 wall-clock |
| --- | --- | --- | ---: |
| 特征与监督准备 | `fa4ed58` | `runs/pets-selector-scale-supervision-v1` | 224.823 s |
| 配对训练 | `604150c` | `runs/pets-selector-scale-training-v1` | 362.477 s |
| 实际 CLIP 评估 | `8b8a231` | `runs/pets-selector-scale-eval-v1` | 83.019 s |

三阶段合计 670.319 秒，约 **0.1862 单卡小时**；该占用时间估计包含阶段内 CPU/I/O，不含外部传输、备份及 CPU 分析，也不是部署延迟。训练阶段中优化更新合计 316.893 秒，周期验证 17.398 秒，峰值 GPU 分配约 1.85 GB。

准备阶段复用 592 张旧训练图片和 144 张旧开发图片，为额外 2,352／592 张图片生成监督，共新增 1,507,328 次 mask 后缀推理，避免重复 376,832 次。原有 144 张开发图没有保留的原始探测响应以 availability 字段标记，不将占位零视为真实响应。最终 ridge 系数和前缀已正确复用。

复现入口如下；先准备已有数据、基线和旧缓存，GPU 阶段使用本轮的固定配置：

```bash
.venv/bin/python scripts/prepare_selector_scale.py --config configs/pets_selector_scale.json
.venv/bin/python scripts/train_selector_scale.py --config configs/pets_selector_scale.json --output runs/pets-selector-scale-training-v1
.venv/bin/python scripts/evaluate_selector_scale.py --config configs/pets_selector_scale.json --models runs/pets-selector-scale-training-v1 --output runs/pets-selector-scale-eval-v1
.venv/bin/python scripts/analyze_selector_scale.py --source runs/pets-selector-scale-eval-v1 --training runs/pets-selector-scale-training-v1 --output results/pets-selector-scale-v1 --plot
```

## 结果与解释

在完整未见组合评估中，大覆盖条件化准确率 **98.9919%**，图像版 **98.9906%**，差值 +0.0013 个百分点，95% 区间 [−0.2186, +0.2141]。预定的条件化优势变化为 +0.1887 个百分点 [−0.1559, +0.5645]，仍不能建立增加覆盖后的可靠优势。大覆盖条件化的平均分类间隔比图像版低 0.000919 [−0.001564, −0.000228]。

监督拟合则明显改善：条件化的训练／验证 KL 从 0.1418／0.8330 变为 0.2696／0.4995；图像版从 0.3349／0.6478 变为 0.4285／0.5506。条件化验证 KL 在三个种子中均由较差变为较好，但不能据此推断实际分类更好。这是本轮最清楚的认识：扩大覆盖缓解了教师拟合的泛化差距，仍不足以证明任务条件化优于强共享选择。

大覆盖条件化相对图像版，在原有面板准确率差 +0.3157 个百分点，在新增面板为 −0.0779 个百分点，两者区间均跨零。小覆盖分别为 +0.2525 与 −0.3020 个百分点。旧小面板的正向均值不能代表全部开发图片。

任务交换使用不同的统计对象：616 张支持交换的图片、每种子 4,315 对有效任务，图片等权。正确描述相对交换描述的 margin 效应在小／大覆盖分别为 +0.003699 和 +0.001795，两个区间都高于零；覆盖增加使该效应减少 0.001904 [−0.002304, −0.001542]。大覆盖交换准确率效应 +0.1165 个百分点，区间跨零。两组图像版交换效应均严格为零。原有 124 张和新增 492 张交换图片中的 margin 效应都保持正向。

因此，有用的候选信息依赖仍然存在，但更好地模仿教师并没有带来更强的候选依赖或可靠的分类领先。该变化值得结合 token 预算、排序目标和特征几何继续解释，现阶段不能指定其内部机制。详细表格、逐种子结果和图片见 [结果说明](../../results/pets-selector-scale-v1/README.md)；新增洞见记录为 I6。

## 验证与证据保存

[准备核验](../../results/pets-selector-scale-v1/preparation_audit.json)确认旧缓存精确对齐、抽样 ridge 重算一致、训练／开发图片与任务划分不交叉且未进入最终测试。[训练核验](../../results/pets-selector-scale-v1/training_audit.json)独立重建全部采样序列和曝光计数：每任务／类别恰好 480 次，全部 592／2,944 张图片均实际使用；12 个最终权重可加载且数值有限，更新数、参数量与曲线完整。每种子同一模型类型在两种覆盖下的初始验证 KL 精确一致；初始化张量一致性由训练时断言检查，未另存初始 checkpoint。[评估核验](../../results/pets-selector-scale-v1/evaluation_audit.json)独立检查全部 356,790 行预测，并重算加权主效应和配对区间，结果在浮点精度内一致；两张汇总图已检查。

结果摘要、图表、曲线和审计 JSON 保存在 `results/pets-selector-scale-v1/`。完整监督、12 个最终权重、逐样本 37 类分数、选择位置和图像嵌入归档于项目父目录的 `artifacts/pets-selector-scale-v1/`，分别为 `pets-selector-scale-supervision-v1.zip` 和 `pets-selector-scale-models-and-evaluation-v1.zip`；服务器保留原始结果。两 ZIP 约 421 MiB 和 898 MiB，是长期证据备份，不是待删除的传输中间件。GitHub Release `pets-selector-scale-v1` 的发布和目的地恢复核验正在完成，以结果目录中的后续核验记录为准。

## 下一步

本轮完成了训练图片覆盖的受控比较，但尚未建立 3B 的强共享基线优势。下一轮优先补齐 49／98／147 token 的成对实验，并开展 H2 的归一化特征变化分析，解释监督拟合与分类表现为何分离。随后完成 Flowers102 和已升为必做项的 ViT-L/14 定向验证；保留最终测试用于方案固定后的确认。

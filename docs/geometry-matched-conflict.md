# 在同一合成图内匹配任务方向与完整模型难度

目的：检验第一轮“不同对象的选择冲突更大”是否超出了文本决策方向和任务难度的差异。复用 `results/evidence-conflict-v1` 的完整类别分数，无新图片、无模型推理、无 selector 训练。

## 在查看新配对的剪枝结果前固定的设计

1. 对每张既有双对象合成图，枚举两个真实来源类别的全部**预先划分为val的候选类别对**。干扰类不能是图中任一来源类别；不使用预留test类别对。
2. 分别构造询问同一对象的任务对与询问不同对象的任务对。任务方向统一指向当前真实来源类别；margin为完整CLIP的几何margin。
3. 仅在**同一张合成图**内匹配。两个任务对的方向夹角相差至多5°；各自排序后的两个完整模型margin分别相差至多0.015。这三个条件同时满足，才允许匹配。
4. 每图最多保留一组“同对象任务对—不同对象任务对”。选择三项差异分别除以其阈值后的平方和最小者；若并列，按任务标识排序。选择完全不读取剪枝结果。
5. 分别固定两个面板：不限制完整模型是否正确；以及双方的两个任务均被完整模型答对。它们各自按相同规则选一组，并非事后挑选有利面板。

先写出 `matched_design.json` 和 `coverage.json`，之后才加载压缩预测。每张未匹配图片记录候选数量、单项与联合阈值通过数量、最近候选的距离及失败原因。首轮预定20张匹配图作为覆盖检查，**该数目不是统计功效保证**；不足时仍可保留描述结果，但不把小样本无差异解释为没有效应。不会依据剪枝结果放宽阈值。

## 评估

沿用两个剪枝层（3/6）、两个预算（16/49）和固定35-mask库：32随机、空间均匀、两个半图mask。排除attention，并核对两个剪枝层的mask位置完全相同。

主要结果为每组匹配中“不同对象减同对象”的共同选择冲突率：各题分别存在正确mask，但没有同一mask同时答对两题。同步报告分别可行率、共同可行率，避免把两题都无法解答误写成没有冲突。最佳margin差距仅为次要描述，不能将其由优化定义保证的非负性当作发现。

置信区间按合成图进行配对bootstrap（种子2111，2,000次），条件于当前类别、候选任务和来源图片面板。一些合成图复用了来源图片，因此这不估计独立新来源图片上的总体不确定性；全零事件产生的零宽bootstrap区间也不是等效性证据。每张图的来源ID和逐条件结果均保留。

## 设计阶段覆盖结果

仅查看完整模型分数后：Pets匹配19/64张（两个面板均19）；Flowers匹配7/64张，限定完整模型正确后为6/64张。主要障碍是方向夹角缺乏重叠，两个数据集均未达到预定覆盖检查。此时尚未读取这些新配对的剪枝结果。

可单独执行设计阶段：

```bash
python3 scripts/analyze_geometry_matched_conflict.py \
  --design-only --output results/geometry-matched-conflict-v1
```

完整执行仍先冻结上述设计，再读取已保存的剪枝分数：

```bash
python3 scripts/analyze_geometry_matched_conflict.py \
  --output results/geometry-matched-conflict-v1
```

下一阶段如要扩大可比范围，应独立预先定义新的任务池或合成面板，例如使用train+val开发类别对；这会改变实验目标覆盖，必须保留本轮全部结果，并继续隔离test类别对。当前脚本仅实现val任务池，不自动扩展。

## 第二阶段：独立冻结更大的开发任务池

第一阶段覆盖不足后，在**尚未查看新匹配的剪枝结果时**，固定第二阶段：允许预先划分为train或val的全部开发类别对，仍严格排除test类别对；来源合成图、5°／0.015阈值、匹配距离和选择规则全部不变。这一步针对稀疏val邻居导致的几何支持不足，不通过剪枝表现选择任务。目标是机制诊断，**不称作未见类别组合泛化验证**。

第二阶段设计固定在 `results/geometry-matched-conflict-devpool-v1/`，第一阶段的 `results/geometry-matched-conflict-v1/` 完整保留。扩展后Pets两个面板均匹配64/64张；Flowers不限制正确性时为56/64张，限定完整模型正确时为54/64张。

`matching_balance.json` 记录每个坐标的两组均值、配对绝对差与最大差，以及来源图片复用。第二阶段完整模型正确面板中，Pets匹配方向角的平均绝对差为0.18°、两个margin分别约0.00041/0.00042；Flowers为1.23°、约0.00233/0.00231。Pets复用了10张来源图（最多2次），Flowers复用了5张（最多3次）。统计结果仍条件于当前来源面板。

```bash
# 仅设计，不读取压缩分数
python3 scripts/analyze_geometry_matched_conflict.py \
  --design-only --task-pool train_val \
  --output results/geometry-matched-conflict-devpool-v1

# 提交冻结设计后，分别评价两套已保存设计；此模式不重算或覆盖匹配
python3 scripts/analyze_geometry_matched_conflict.py \
  --evaluate-only --output results/geometry-matched-conflict-v1
python3 scripts/analyze_geometry_matched_conflict.py \
  --evaluate-only --output results/geometry-matched-conflict-devpool-v1
```

新增任务只改变读取已保存类别分数的方式，所有结果均由同一批原始完整／压缩预测计算，无额外GPU工作。

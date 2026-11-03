# 几何与难度匹配：val任务池

此阶段先用完整模型分数冻结同图匹配，再读取既有压缩预测；匹配阈值为方向夹角差≤5°，两个排序后的几何margin分别相差≤0.015。每张图最多一组“同对象任务对—不同对象任务对”，仅使用原定val类别对。详见[协议](../../docs/geometry-matched-conflict.md)。

**结果首先暴露了可比任务不足：**Pets两个面板均匹配19/64张，Flowers不限制原模型正确性时匹配7/64张，限定双方原模型均正确时匹配6/64张。方向夹角支持不足是主要原因。因此本轮保留为小样本开发诊断。

在原模型正确面板中，两数据集、两层（3/6）、两预算（16/49）的同对象与不同对象任务对均未出现共同选择冲突；全部匹配图均能在固定35-mask库中找到共同正确选择。这里的零事件不说明总体不存在差异。不限制原模型正确性时，仅Flowers block6/16-token出现1个同对象冲突、0个不同对象冲突；完整结果保留在CSV与JSON中。

margin取舍差异方向并不完全稳定：Pets不同对象减同对象的差距为0.00202–0.00920；Flowers为−0.00141–0.00520。稀疏匹配限制了跨数据集解释。

在查看这些压缩结果**之前**，已另行冻结扩大train+val候选池的阶段，以增加几何重叠而不改变阈值或图片。两套结果独立保留；[扩展任务池结果](../geometry-matched-conflict-devpool-v1/README.md)覆盖更充分，目标为机制诊断，不作为未见类别组合泛化验证。

- `matched_design.json`：冻结匹配与每张未匹配图的原因。
- `coverage.json`、`matching_balance.json`：覆盖、匹配偏差、来源图复用。
- `paired_outcomes.csv`、`summary.json`：全部逐图结果及配对bootstrap汇总。

```bash
python3 scripts/analyze_geometry_matched_conflict.py \
  --evaluate-only --output results/geometry-matched-conflict-v1
```

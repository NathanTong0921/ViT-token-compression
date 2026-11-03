# 机制线索诊断与研究扩展

- 状态：三项诊断完成；新任务、新训练与方法干预尚未运行。
- 代码版本：`6e19750dfed4d1384285c21aefa43cff4a79dd0a`。
- 执行：服务器项目 Python 环境，CPU；不新增 GPU 训练、监督生成或正式测试访问。
- 输入：已保存 Pets49／Flowers98 图像版检查点及开发缓存、训练文本字典、四设置 KL／ranking 逐例结果。
- 配置：固定种子2041／2042／2043；折叠核对每设置32图；类别身份核对所有字典类别对；教师转移保留全部48条汇总记录。

运行入口为 `scripts/diagnose_capacity_folding.py`、`scripts/diagnose_descriptor_identity.py`、`scripts/diagnose_teacher_transfer.py`，输出统一位于 `results/mechanism-diagnostics-v1/`。前两项分别核对固定上下文参数折叠与原始文本均值的信息保留，第三项对照教师预测和实际 CLIP 的目标变更效果。

结果、精确输入位置、逐种子数值与解释见[结果记录](../../results/mechanism-diagnostics-v1/README.md)。与原proposal的关系及新实验安排见[扩展方案](../research-expansion.md)和研究洞见 I17。本轮为正面推进：得到可区分原因的线索，尚未产生新的分类优势。

备份：小型诊断结果与脚本随Git同步本地、服务器、GitHub；输入沿用既有归档，不创建重复大文件。传输用bundle在同步后删除。

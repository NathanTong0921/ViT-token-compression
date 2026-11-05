# 外部对照：Patch Ranking 的适配与可借鉴设计

更新：2026-11-05。本轮完成论文与公开代码评估；未启动新训练。

## 结论与选择

**优先补 Patch Ranking 的 feature-preservation predictor，采用冻结 CLIP、无 prompt tuning 的版本。** 它与本项目共享 CLIP ViT-B/16、patch 排序和轻量预测器这一设定，比引入一个依赖其他主干重训的方法更直接。第一轮在 Pets、Flowers 的 block 6、98/196-token 下比较，回答：当前强共享选择相对于独立来源的排序方法处在什么水平？

这项工作的重点是接好并训练独立 predictor。我们早期已经实现了与其相关的离线探测：2×2 滑动删除、按覆盖次数平均 patch 分数，以及 representation-preservation 排序；代码在 [selection.py](../src/token_compression/selection.py)，结果在 [Pets 离线实验](../results/pets-selection-v1/README.md)。当前 C/U 则学习目标预算随机掩码的任务效用教师。两者有共同技术基础，但教师和预测器训练路线不同。

建议表中名称采用 **Patch Ranking–Preservation (adapted)**；同表给出 C、原 image-only、wide U、attention 与完整 CLIP。论文原表以多分类任务及其他剪枝位置为主，我们重新在同一二分类面板上测量各方法。

## 已核对的论文与代码

论文将排序监督、预测器和压缩后的模型适配分为三个阶段。监督包含 label、confidence、feature preservation 三种目标；块删除比单 token 删除产生更清楚的信号。核心可借鉴处是把离线选择效果、排序可预测性和实际分类效果分开观察。[论文 §3、Table 1](https://openaccess.thecvf.com/content/WACV2025/papers/Wu_Patch_Ranking_Token_Pruning_as_Ranking_Prediction_for_Efficient_CLIP_WACV_2025_paper.pdf)

公开实现值得复用，但目前需要补齐训练入口与材料：

| 核对项 | 现状与实现选择 |
| --- | --- |
| README 的启动方式 | 指向 `configs/train_predictor.yaml`，当前 configs 下只有 datasets、trainers 两个目录；应接入本项目现有数据读取和配置。 |
| 排序材料 | `train_predictor.py` 读取硬编码 Pets train/test ranking JSON；当前 `test/` 只有路径替换脚本。使用本项目删除窗口路径生成训练排序。 |
| predictor | 代码将通道池化到 64 维，使用 `MixerBlock(64,197)` 和线性评分头；包含跨 token 混合，与本项目逐 patch 的 MLP 不同。 |
| 训练目标 | 公开入口将离散排序转换为分布并训练 KL；论文正文给出标准化分数的 sigmoid 目标。适配版采用公开入口的 KL 路线，明确这一选择。 |
| 表示位置 | 代码用 timm 的 `_intermediate_layers(..., n=10)` 后取 `feature[-1]`，没有直接指定正文 Table 1 的第 4 层。适配中明确取本项目第 6 block 输出。 |
| 权重发布 | 本轮查看的 README、目录与 Releases 页未提供可直接使用的 predictor checkpoint；需要训练。 |

来源：[官方 README](https://github.com/CEWu/PatchRanking/blob/main/README.md)、[训练入口](https://github.com/CEWu/PatchRanking/blob/main/train_predictor.py)、[configs](https://github.com/CEWu/PatchRanking/tree/main/configs)、[test](https://github.com/CEWu/PatchRanking/tree/main/test)、[Releases](https://github.com/CEWu/PatchRanking/releases)。这些差异意味着应交付一个清楚标注的论文方法适配，而不是把 README 命令直接套到现有实验。

## 最小实施方案

1. **排序教师。** 复用冻结前半网络与后半网络接口，逐图删除 2×2 窗口，按完整／删减图像嵌入的 cosine 差生成 preservation 分数。同一图片只生成一份排序，可供全部候选任务复用。早期 144 张开发图片的探测数据用于接口验证；训练仍来自正式训练图片。
2. **预测器。** 接入官方池化＋MixerBlock＋评分头及其排序到目标分布的转换，训练 3 个固定种子。采用公开入口 Adam、50 epochs 的路线；第一轮记录实际收敛与成本，不把现有 C/U 的训练步数硬当作该方法的最佳配方。
3. **主比较。** 两数据集、block 6、98-token，总计 6 个新模型；在现有开发图片／未见类别组合上与固定 C、wide U 等比较准确率、margin、参数量、推理时间。所有方法使用同一图像预处理、CLIP 权重和分类文本。
4. **定位结果。** 如果 preservation predictor 明显优于 wide U，更新论文的共享参照；如果接近或弱于 wide U，说明当前参照已有相当竞争力。两种结果都直接回答外部定位问题。49-token 的扩展接在这一步之后：它是更强压缩条件，原文主要表格从 50% keep rate 起。

这里新增的核心开销是训练图片的删除探测；现有 prefix 缓存、窗口构造、剪枝 suffix、评价面板都能复用。先实际计时一小批，再估算完整运行，避免凭模型大小猜 GPU hours。

**备选只保留 ToMe。** 它适合充当独立压缩方法参照，官方提供无额外训练的合并算法；但它合并特征、常用逐层调度，研究对象已超出保留位置选择。因此需要按总计算／实测时间比较，不能只对齐最终 token 数。官方接口支持 timm／SWAG／MAE，本项目 OpenAI CLIP 路径仍需适配。当前优先 Patch Ranking，不同时展开两套外部方法。[ToMe 官方实现](https://github.com/facebookresearch/ToMe)

## 可以直接借鉴的实验组织

| Patch Ranking 的组织方式 | 本项目对应问题与材料 |
| --- | --- |
| Table 1：离线教师 → matching rate → 实际分类 | 将现有离线潜力、集合恢复和实际 C/U 表现连成证据链；对同一教师显示强共享起点，区分恢复教师与超过共享。 |
| Figure 2：三阶段方法流程 | 画本项目的“同图不同任务 → 训练／交叉选择 → 资源匹配与泛化评价”研究框架。它是流程图，现有实现已足够作图。 |
| 跨数据集 predictor 转移 | 现有 Flowers 是重新训练后验证结论；额外的 Pets-trained → Flowers inference 会回答不同的迁移问题，可在外部 predictor 接入后低成本评估。 |
| 同一预算下列出不同排序／架构 | 将输入任务信息、图像处理能力和选择教师分别改变，让每张表对应一个清楚的问题。 |

这些借鉴主要提升实验表达与可比性。四页正文仍围绕本项目的三个问题组织：任务信息是否被使用、是否比额外图像能力更值得、什么条件下保留额外价值。

## 在 Patch Ranking 基础上进一步研究

### 优先方向：从保留整个图像表示，改为保留候选类别之间的判别关系

现有任务效用教师用真实标签引导选择，提高正确类别相对竞争者的间隔。另一条路线是：**让压缩模型维持完整 CLIP 对当前候选类别的相对判断**。它需要图片和候选文本，不需要知道正确答案。

设归一化完整／压缩图像嵌入为 \(z,z_S\)，两类文本为 \(t_a,t_b\)，令 \(d=(t_a-t_b)/\|t_a-t_b\|\)。可以比较：

\[
E_{\mathrm{global}}(S)=\|z_S-z\|^2,
\qquad
E_{\mathrm{task}}(S)=\big((z_S-z)^\top d\big)^2.
\]

前者要求整个表示保持接近；后者只要求当前分类方向保持接近。多类别可用中心化候选文本张成的子空间投影，得到同一思想的扩展。候选顺序交换不改变这个目标。

**为何值得试：** 它把“利用答案寻找更好的选择”与“利用任务保留已有判别信息”分开，也直接连接我们的主问题。论文已比较全局 preservation 与 label/confidence，所检查正文未把候选子空间 preservation 作为一个独立目标；这里将其作为待检验方案，不预设其新颖程度或性能优势。

**第一步：** 用已有窗口后的类别 cosine 分数和完整分数，计算两种保存误差，先看它们是否产生不同且有用的保留集合。需要实际后半网络评价新集合；已有分数只负责构造目标。随后再训练同等容量的条件化／共享预测器，区分教师本身更好与任务输入使教师更容易被预测。完整 CLIP 原有错误也会被该目标保留，因此主要指标同时列真实准确率与相对完整模型的偏离。

这比继续微调同一 KL／集合目标更换了问题本身；它也可与已讨论的低维任务输入结合，但应先分别看清教师和输入的作用。

### 第二方向：任务条件化的收益是否受跨 token 信息交互限制？

本项目评分头读单个 patch、CLS 和任务描述；Patch Ranking 的 Mixer 同时处理整个 token 序列。直接集合监督在训练图片上仍有恢复差距，提供了检查交互能力的动机。

保留同一教师与训练数据，比较逐 patch MLP 和 Mixer 的条件化／共享版本，维持近似参数预算。若 Mixer 同时改善 C/U，收益来自图像交互能力；若它更明显地改善 C−U，说明任务信息的效用依赖选择器怎样汇总其他 patch。可以先复用现有混合教师与缓存，不重新做完整搜索。

这条与“再加宽网络”不同：它改变信息如何在 patch 之间交换。它的研究价值是给当前资源分配结论加入架构解释；是否值得展开，由外部 predictor 的表现决定。

## 推荐顺序

先完成 preservation predictor 的外部适配与当前基线比较；同时仅做候选子空间目标的离线小检查。若后者产生清楚的保留集合差异和分类价值，再进入配对学习；Mixer 条件化作为外部比较引出的后续选择。已有目标预算随机掩码路线继续作为工具复用，不再将“预算匹配教师”另列为新贡献。

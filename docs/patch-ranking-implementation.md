# Patch Ranking：公开实现与本项目的对应关系

核对日期：2026-11-05。当前对照使用 **Patch Ranking–Preservation (adapted)**：保留其 preservation 教师思想、公开 Mixer predictor 和 KL 训练路线，接入本项目冻结 CLIP 的 block 6、98-token 与二分类评价面板。

| 部分 | 官方论文／代码 | 本项目实现 |
| --- | --- | --- |
| 删除位置 | `clip/model.py` 在 `blk_id in pruning_loc` 时先删除，再执行该 block。循环从 0 计数。官方运行脚本设 `plocs=(3)`，即第 4 个 block 前。 | 使用前 6 个 block 的缓存，删除后运行第 7–12 个 block；对应官方 `pruning_loc=6`。可以复用 prefix，不需要为每个窗口重复前 6 层。 |
| 删除窗口 | 论文 §3.1 描述重叠滑动与覆盖平均；公开 `generate_masks` 用步长 `k` 的非重叠块，`custom_avg_pool` 将每块分数复制回各 patch。官方示例设 `blocksizes=(2)`。 | 2×2、步长 1、按覆盖次数平均，采用论文描述的滑动方式。14×14 patch 网格共有 169 个窗口。 |
| 排序方向 | 删除后与完整表示的 cosine 越高，说明被删 token 越不重要。官方剪枝代码删除 cosine 分数最高的部分；训练函数说明 `ranking` 按重要性从高到低排列。 | 重要性取 `1 − cosine`，从大到小排序；预测分数同样取最高 98 个。CLS 始终保留。 |
| predictor 输入 | 公开入口调用 timm `_intermediate_layers(image, n=10)` 后取 `feature[-1]`；在 timm 0.9.16 中它实际是最后一个 block 的输出。 | 显式读取 block 6 的 197 个 token，与实际执行选择的位置一致。 |
| predictor 结构 | 通道平均池化至 64 → `MixerBlock(64,197)` → 去掉 CLS → 64→1 线性头。 | 同一结构。timm 1.0.20 与 0.9.16 的该 block 默认结构一致：token MLP 197→32→197、channel MLP 64→256→64、LayerNorm epsilon 1e-6、GELU，无 dropout。 |
| 训练监督 | 论文 Eq. 8 描述标准化分数的 sigmoid 目标；公开入口使用 KL。公开目标将重要性排名 0–145 的 logits 置 0，146–195 置 `−5 − 0.1 × rank`，再 softmax。 | 复用公开 KL；Adam，学习率 0.001，weight decay 0.0001，50 epochs，cosine 最低学习率 0.0001，batch 64。 |
| 排序指标 | 论文 Table 1 为保留前 100 个的 matching rate；公开 `top_n_match(...,-50)` 实际统计末尾 50 个的重合率。 | 分别报告保留前 98、前 100，以及删除末尾 50 个的重合率。 |
| 模型适配 | 论文另有 prompt tuning 阶段；Table 1 也单列不做后续 tuning 的结果。 | 本轮冻结主干与文本编码器，不加 prompt tuning。 |

公开 KL 对前 146 个位置赋予同一目标概率，因此没有训练它们之间的精细顺序。删除末尾 50 个的恢复程度与保留前 98 个的恢复程度回答不同问题；结合实际分类表现一起解读。

## 来源

- [论文 §3 与 Table 1](https://arxiv.org/html/2409.14607v2#S3)：教师、预测器、训练目标和评价定义。
- [官方删除实现](https://github.com/CEWu/PatchRanking/blob/main/clip/model.py#L388)、[分数回填](https://github.com/CEWu/PatchRanking/blob/main/trainers/zsclip.py#L93)、[运行参数](https://github.com/CEWu/PatchRanking/blob/main/scripts/zsclip/gpu0_INexp.sh)：位置、窗口和 cosine 分数。
- [官方训练入口](https://github.com/CEWu/PatchRanking/blob/main/train_predictor.py#L229)：排序分布、KL、predictor 与优化设置。
- [timm 0.9.16 MixerBlock](https://github.com/huggingface/pytorch-image-models/blob/v0.9.16/timm/models/mlp_mixer.py)、[timm 1.0.20 MixerBlock](https://github.com/huggingface/pytorch-image-models/blob/v1.0.20/timm/models/mlp_mixer.py)、[旧版 intermediate layers](https://github.com/huggingface/pytorch-image-models/blob/v0.9.16/timm/models/vision_transformer.py)：结构兼容性与特征位置。
- [官方 MIT License](https://github.com/CEWu/PatchRanking/blob/main/LICENSE)：使用或改写公开实现时随代码保留许可及原版权说明。

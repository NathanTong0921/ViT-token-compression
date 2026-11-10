# Direct classification training at 98 retained patches

## Question and fixed replication

Flowers49 showed that direct classification training increases the accuracy benefit of C over the strong shared U. This round applies the same training recipe at 98 patches and directly compares the two budgets on the same evaluation panel.

Use frozen CLIP ViT-B/16, block 6, the original Flowers train/development splits, seeds 2041–2043 and the existing 98-token C/U checkpoints. Each checkpoint receives either direct classification supervision or continued teacher KL for 1,200 updates. Keep batch size 32, AdamW learning rate 1e-5, weight decay 0.01, gradient norm limit 1, standardized-score gate temperature 0.5, teacher temperature 1 and FP32 execution unchanged from Flowers49. Retain 98 patches plus CLS in every hard forward pass.

The original 98-token teacher and warm-start weights are budget-specific. All other training settings remain fixed. The same seeded task/class-balanced sample streams are used for C/U and the two objectives. The 20-step smoke run checks hard-policy/native-gather agreement, gradient flow and throughput; its weights are discarded.

## Endpoints and comparisons

Save the six continuation checkpoints per arm for recovery and learning curves; the endpoint is always step 1,200. Evaluate all initial/KL/direct C/U weights on both the 1,020-image development panel and the existing 6,149-image/204-pair test panel. The larger panel has already been evaluated with earlier models. Development results do not select a checkpoint or alter this round's test settings.

Report accuracy, CE, raw cosine gap and geometric margin, with task/class-balanced weights and every seed. Primary within-budget contrasts are direct C−U and the change in C−U between direct and KL training. Per-arm direct−KL and direct−initial contrasts show how the route changes each selector.

Compare budgets using paired per-example outcomes from the same 6,149-image panel. The two main accuracy interactions are:

1. (direct C−U at 98) − (direct C−U at 49).
2. (direct C−U − KL C+KL U at 98) − the corresponding quantity at 49.

Average the fixed three seeds and bootstrap image clusters 2,000 times using seed 2717. Apply the two-primary adjustment within each analysis and metric. Retain raw-gap, geometric-margin and CE budget interactions as accompanying results. Comparing the actual budget differences determines whether the evidence supports budget dependence.

## Outputs and decision

Training/development outputs: `/data4/zhao.tong/experiments/direct-classification-98-v1`. Expanded-panel outputs: `/data4/zhao.tong/experiments/direct-classification-98-panel-v1`. Code and reused caches stay in `/data2/zhao.tong/experiments`. Curated results go to the correspondingly named local/Git directories; full checkpoints and raw outputs go to data4 archives and a GitHub Release. No new datasets or pretrained models are downloaded.

The three seeds run concurrently on available A6000 devices (2041 on GPU6, 2042 on GPU1, 2043 on GPU3), with separate shard directories and the same training recipe. Each seed's four continuations stay on the same device. Identical evaluation manifests are checked before the shard outputs are consolidated; per-shard execution records retain actual device placement. Expanded-panel inference runs on GPU6.

After both budgets are analyzed, reassess the route using the accuracy gain, its consistency across panels/seeds, and the paired budget interactions. This round contains no hyperparameter search, extra training steps, or further dataset expansion.

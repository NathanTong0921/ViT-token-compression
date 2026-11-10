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
2. [(direct C−direct U) − (KL C−KL U)] at 98, minus the same quantity at 49.

Average the fixed three seeds and bootstrap image clusters 2,000 times using seed 2717. Apply the two-primary adjustment within each analysis and metric. Retain raw-gap, geometric-margin and CE budget interactions as accompanying results. Comparing the actual budget differences determines whether the evidence supports budget dependence.

## Outputs and decision

Training/development outputs: `/data4/zhao.tong/experiments/direct-classification-98-v1`. Expanded-panel outputs: `/data4/zhao.tong/experiments/direct-classification-98-panel-v1`. Code and reused caches stay in `/data2/zhao.tong/experiments`. Curated results go to the correspondingly named local/Git directories; full checkpoints and raw outputs go to data4 archives and a GitHub Release. No new datasets or pretrained models are downloaded.

The three seeds run concurrently on available A6000 devices (2041 on GPU6, 2042 on GPU1, 2043 on GPU3), with separate shard directories and the same training recipe. Each seed's four continuations stay on the same device. Identical evaluation manifests are checked before the shard outputs are consolidated; per-shard execution records retain actual device placement. Expanded-panel inference runs on GPU6.

After both budgets are analyzed, reassess the route using the accuracy gain, its consistency across panels/seeds, and the paired budget interactions. This round contains no hyperparameter search, extra training steps, or further dataset expansion.


## Completed results, 2026-11-10

All twelve continuations reached 1,200 updates. All 18 development and 18 expanded-panel endpoints completed. Training used 834.87 GPU seconds (0.232 GPU hours); expanded cached-prefix evaluation used 308.15 GPU seconds (0.086 GPU hours). Native deletion and dense hard-policy smoke outputs agreed within 2.99e-7. These timings describe this continuation experiment, not total project training or deployment latency.

On the expanded panel, direct C/U accuracy was 97.5386%/97.4187%. C−U was +0.1200 pp [−0.0280,+0.2794]; its change relative to KL training was +0.0257 pp [−0.1395,+0.1948]. The corresponding Flowers49 effects remain +0.3056 and +0.2467 pp, with positive adjusted intervals. Directly paired 98−49 changes were −0.1856 pp [−0.4072,+0.0489] and −0.2210 pp [−0.4636,+0.0187]. This round did not establish another accuracy improvement or a reliable accuracy interaction between budgets.

After the primary results were known, a secondary accuracy/retention table was computed from the same outcomes. Direct C has nearly equal observed accuracy at 49 and 98 tokens; the C budget gap changes by −0.3545 pp [−0.5171,−0.1947] relative to KL. This exploratory contrast uses ordinary 95% image intervals. The fixed primary comparisons remain the basis for the budget-dependent conditioning conclusion.

The route contributes an effective training intervention at Flowers49 and a measured limit at Flowers98. Retain both in the research account. A focused possible follow-up is to exchange task descriptions in the final direct/KL C models and test whether the improved accuracy comes with more useful task dependence. No further training or follow-up evaluation was launched in this round.

[Development results](../../results/direct-classification-98-v1/README.md) · [Expanded panel and paired budget comparison](../../results/direct-classification-98-panel-v1/README.md)

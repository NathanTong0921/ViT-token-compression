# Offline task selection and exchange pilot

This development experiment follows the verified single-stage pruning pilot. It tests whether deletion-derived task rankings transfer to actual fixed-budget pruning, and compares them with shared choices that receive substantial extra information. It does not train a selector or evaluate official test data.

## Fixed panel and interventions

- Frozen OpenAI CLIP ViT-B/16, after block 6; 147/98/49 retained patches plus CLS. Original token values and positions remain unchanged.
- Use existing validation images and the 66 validation class pairs. The smoke panel takes the first one image per participating class in the existing balanced ordering (36 images); the larger pilot takes four (144 images). These choices precede pruning outcomes. Both sides of every single binary task have equal image counts.
- Exchange cases use two different validation tasks sharing an image's true class. The validation graph supports 31 such anchor classes and 217 task-pair combinations. Exchange aggregates therefore differ from balanced 66-task classification: average cases within each image first, then images. Record both quantities separately.
- Official test images and the 68 test class combinations remain unused.

## Offline rankings

Delete each stride-one 2-by-2 patch window from the intermediate sequence, run the native suffix, and record all 37 cosine scores. For task `(y,c)`, a window's utility is the full signed geometric margin minus its margin after deletion. A patch receives the mean utility of all windows containing it. Retain the top-k patches using stable index tie-breaking.

This is inspired by [Patch Ranking, section 3.1](https://arxiv.org/html/2409.14607v2#S3.SS1), which motivates block masking because individual token changes can be small. Its [official example](https://github.com/CEWu/PatchRanking/blob/main/scripts/zsclip/gpu0_INexp.sh) uses block size 2, but its current masking code uses non-overlapping windows. Our sliding, overlapping windows and signed geometric-margin objective are explicitly our implementation choices, not an exact reproduction. A separate one-patch-deletion smoke run checks sensitivity on the same images.

Two task rankings yield `task_a` and `task_b`. The mean of their utility vectors yields `shared_proxy`. A representation-preservation ranking averages `1 - cosine(full_embedding, deleted_embedding)` and does not use labels or candidate tasks. Attention and random remain comparisons. The ranking proxy uses near-full deletion interventions; final top-k accuracy/margins are measured separately, so proxy improvements need not survive strong compression.

## Stronger shared references and fixed search budget

For each image, task pair, and budget, evaluate the same 64 candidate subsets once. Initial sets are task A, task B, their mean proxy, attention, representation preservation, task-utility mixtures with A weights 0.25 and 0.75, and three random sets. The remaining candidates are deterministic seeded swap neighbors of these initial sets (10% of the smaller retained/deleted count, rounded with a minimum of one swap). All keep exactly k patches. Repeated subsets remain in the fixed compute budget.

- `shared_search_mean`: the subset maximizing the actual mean of the two geometric margins.
- `shared_search_maxmin`: the subset maximizing the smaller of the two margins.
- `task_search_a/b`: separately maximize each task's margin over this same bank.

The shared references see both tasks and the true label; they are privileged two-task shared references, not deployable image-only selectors. The representation-preservation ranking is the image-only reference. The bank is bounded search, not a guarantee of an optimal shared selection. Cached candidates also permit search-depth checks at 16/32/64 without extra model execution.

Primary evidence compares deletion-derived task choices against shared references and exchanged choices under actual pruning. The same-bank task-search margin advantage is mathematically nonnegative by construction: its existence alone is not evidence for H1. Its magnitude can diagnose whether an ineffective deletion proxy misses available choices.

## Exchange geometry and analysis

For each task evaluate its own set, the other task's set, and all shared references; retain correctness and signed margins. Both task directions point toward the common true class. The symmetric margin exchange loss is

`L = ((z_A - z_B) dot (d_A - d_B)) / 2`,

so `abs(L) <= norm(z_A-z_B) * sin(angle/2)` for unit task directions. Angle therefore supplies a mechanical scale even before any mechanism is established. Save `z_A,z_B`, verify this identity, and examine representation distance and normalized directional alignment alongside raw losses. Margin-controlled angle correlations alone do not demonstrate a causal mechanism.

Use fixed-seed image-cluster bootstrap intervals and image-grouped validation for exploratory angle prediction beyond full-model margin mean/difference and nonlinear controls. These remain development analyses; no final confirmatory claim follows from this pilot.

## Retention and cost

Save the sampled manifests, deletion scores/window positions, single-task outcomes and positions, full exchange rows, and candidate banks (all positions/scores plus the two proxy-task embeddings). Record model source, configuration, measured supervision-generation/evaluation cost, and peak GPU allocation. No persistent prefix cache is needed. Small scientific artifacts are copied locally and to GitHub; transfer-only artifacts are removed after verification.

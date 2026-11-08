# Direct classification supervision for token selection

## Question

Does optimizing the current selector's actual classification loss change the value of task conditioning, compared with continuing to imitate the offline utility ranking?

The existing teacher targets are derived from real CLIP mask responses and labels. This experiment removes the teacher-matching objective from one training branch: the current selected mask is evaluated by the frozen CLIP suffix, and its candidate classification loss updates the selector.

## Fixed comparison

- Flowers102, original training/development images and candidate pairs, CLIP B/16, block 6, 49 retained patches plus CLS.
- Start from the existing conditioned C and strong shared U checkpoints for seeds 2041–2043.
- Continue each checkpoint with either actual candidate cross-entropy or its original teacher-KL objective. This gives twelve continuations plus the six initial endpoints.
- Pair the sample stream, batch size 32, 1,200 additional updates, AdamW learning rate 1e-5, weight decay 0.01 and gradient norm limit 1 across objectives and arms.
- Freeze CLIP, including its existing logit scale. Use FP32 for the backbone, scoring and gradients. Re-evaluate the initial weights in the same execution path.
- Save step 200 and subsequent 200-step checkpoints for recovery and learning curves. The primary endpoint is step 1,200, with no choice based on development accuracy. A short training/throughput check precedes the full run.

## Differentiating a fixed-size selection

Hard top-k determines the actual 49-token policy in the forward pass. Training evaluates the suffix densely with dropped keys assigned zero attention weight in every remaining block. CLS is always present. The surviving tokens and final CLS therefore follow the same forward computation as gathering only those tokens; this equivalence is checked numerically on the real model.

For the backward pass, use a straight-through gate. Standardize each row of selector scores and form sigmoid gates with temperature 0.5. Solve the threshold so their sum is 49, and differentiate that threshold implicitly. This supplies a smooth, budget-preserving surrogate gradient while the forward mask remains hard. The classification objective is the actual hard-mask loss; its gradient with respect to discrete selection is approximated by this gate.

Attention masking for learnable token pruning follows [DynamicViT (NeurIPS 2021)](https://proceedings.neurips.cc/paper_files/paper/2021/hash/747d3443e319a22747fbb873e8b2f9f2-Abstract.html) and its [official implementation](https://github.com/raoyongming/DynamicViT/blob/master/models/dyvit.py). We adapt this approach to the frozen OpenAI CLIP suffix and fixed-cardinality gate, using zero dropped-key weight without additive softmax smoothing.

## Measurements

Use native hard token gathering for all evaluation. Report candidate cross-entropy, accuracy, raw score gap and geometric margin for initial, KL-continued and directly trained C/U. The two main comparisons are the final C−U difference and the change in that difference between direct and KL training. Also report each arm's improvement over its own initial and KL-continued endpoints.

Retain all three seeds and paired image-bootstrap intervals. Save per-example scores, masks and task/class weights, training curves, actual steps, model checkpoints and measured training cost. New final-test inference is outside this pilot.

## Report scope

The preceding content-benefit prediction experiment remains archived as a research record. Following the user's preference, it is not included in the current report evidence plan. This direct-training study addresses the original H3 learning question and the supervision limitation shared by the earlier KL/ranking/set-target experiments.

## Implementation and initial run

`run_direct_classification.py` trains and resumes the four continuations per seed, saving the Adam state at each checkpoint. Twelve local tests cover the gate, classifier loss, paired analysis, and identical interrupted/uninterrupted updates. On real CLIP, hard-policy and native-gather normalized features agree within 4.1e-7 across C/U, random masks, and all-token selection; gradients reach selector scores while the backbone remains frozen.

The 20-step A6000 smoke run measured 0.144 seconds per update for C and 0.137 for U, with about 4.1 GB of peak allocated GPU memory. All twelve continuations completed. Recorded optimizer-loop time was 1,105.73 seconds (0.307 GPU hours), including 1,054.44 seconds for direct training and 51.29 seconds for KL. Development evaluation, monitor evaluation and the timed smoke add 65.27 seconds. Reused checkpoint and teacher preparation are excluded from this round's added cost.

## Result

The three-seed development accuracy is 97.124% for direct C and 96.830% for direct U, versus 96.846% and 96.806% after KL continuation. Direct C improves over KL C by 0.278 percentage points (ordinary image-bootstrap 95% interval [0.025, 0.531]); the gain is positive in all three seeds. Direct C−U is +0.294pp with adjusted interval [−0.074, +0.662], and the change in that gap relative to KL is +0.253pp [−0.188, +0.719]. Both main raw-margin comparisons are positive. The actual classification objective improves C in this setting; its accuracy advantage over strong U remains unresolved. [Tables, learning curves and per-example reconstruction](../../results/direct-classification-v1/README.md).

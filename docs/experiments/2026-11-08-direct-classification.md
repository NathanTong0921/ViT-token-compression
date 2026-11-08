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

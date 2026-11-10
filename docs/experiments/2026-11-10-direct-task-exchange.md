# Task-description exchange after direct classification training

## Question

The fixed Flowers49 experiment found a +0.247 pp increase in the C−U accuracy gap after replacing teacher KL with direct classification training. The same recipe at Flowers98 yielded +0.026 pp. This analysis asks whether the change in accuracy is accompanied by a change in the value of the correct candidate-task description.

Use the existing 49/98-token expanded panels: frozen CLIP ViT-B/16, block 6, seeds 2041–2043, initial and fixed 1,200-step KL/direct C/U endpoints. The 6,149 images and 204 candidate pairs are identical across budgets. No model training or new image inference is needed.

## Intervention and aggregation

For each target image/task row, keep the image and the actual classification candidates fixed. The own outcome uses that task's token selection. Swapped outcomes use selections from every other task attached to the same image, evaluated against the original target candidates. Each donor changes the competing class while retaining the class shared by both tasks.

Saved outputs contain cosine scores against all 102 class texts. Because the remaining visual CLIP blocks do not receive the candidate description, scoring a saved donor output on the target candidates implements this exchange directly. Compute accuracy, raw gap, geometric margin and cross-entropy for each donor first, then average donor outcomes. Do not average donor logits before classifying.

Use the original target task/class weights. This preserves the interpretation of the existing accuracy tables. Average the same three trained seeds and bootstrap whole image clusters 2,000 times with seed 2717. Check the manifest's task degree and retain the donor-row mapping. U is independent of the description; its selections and exchange outcomes provide an implementation check.

## Comparisons

Let D be the own-minus-swapped outcome for C. The two primary accuracy comparisons are the change D_direct − D_KL at 49 and 98 tokens, with intervals adjusted across these two comparisons. Raw gap, geometric margin and CE accompany the accuracy results. CE is a loss, so lower values are better.

Show own and swapped outcomes for initial, KL and direct C, together with the separate direct−KL changes under own and swapped descriptions. This determines whether a larger exchange effect accompanies improved own-task classification or mainly reflects worse exchanged-task performance. Report all three seeds.

Connect this measurement to the original C−U training comparison with the exact decomposition:

    Change in C−U gap = Change in correct-task correspondence effect
                       + (Change in swapped C − Change in own U).

All changes compare direct training with KL continuation. The second term is called the remaining change after exchange; it is an arithmetic component, not a causal mediation estimate or a direct measurement of general image-processing ability. Token-set overlap before and after the objective change provides a secondary description of how much selection varies with the task.

## Deliverables

Write new results to `/data4/zhao.tong/experiments/direct-task-exchange-v1/`; retain curated outcomes, tables, figure sources and an interpretation in `results/direct-task-exchange-v1/`. Reuse full source predictions from the earlier 49/98 evaluation releases, and archive this analysis/code as a small separate release. Code and reused caches remain under `/data2/zhao.tong/experiments`.

This is a focused extension of H1 and H3: it connects useful task dependence to the accuracy improvement found by changing the training objective. A larger exchange effect alone is not the research finding; the result must be read with the own-task gain and the strong shared comparison.


## Completed results

The two manifests align exactly. All 6,149 images have four tasks, yielding three donors per target and 73,788 directed target/donor evaluations per endpoint. U's scores and masks are identical across tasks of the same image. Own outcomes reproduce the existing tables; independent direct calculations agree with the analysis.

At 49 tokens, C's direct−KL accuracy change is +0.4183 pp with the correct description and +0.3675 pp after exchange. The change in the own-minus-swapped effect is +0.0508 pp, adjusted interval [−0.0777,+0.1811]. At 98 tokens, the corresponding changes are +0.0638, +0.0449 and +0.0188 pp [−0.1000,+0.1441]. Neither primary comparison establishes an increased accuracy benefit from the correct task correspondence.

The 49-token C−U objective increment remains +0.2467 pp. Its exact arithmetic decomposition is +0.0508 pp in correspondence-effect change and +0.1959 pp in remaining change after exchange. Both correct and exchanged descriptions improve under direct training. The exchanges retain the class common to the original and donor tasks, so this result concerns competitor specificity within these candidate tasks.

Correct descriptions remain useful: direct C own-minus-swapped accuracy is +0.4885 pp at 49 and +0.4221 pp at 98. The change in geometric-margin exchange effect is negative at 49 (−0.000272) and positive at 98 (+0.000615), with adjusted intervals excluding zero. Selection overlap increases from 65.57% to 68.36% at 49 and from 78.56% to 81.37% at 98, with increases for each seed. Thus, the demonstrated Flowers49 accuracy improvement accompanies more similar selections across the candidate tasks, rather than stronger selection divergence.

This analysis uses saved outputs only: zero additional GPU time and no new training. CPU time was not separately recorded. [Tables, figure and checks](../../results/direct-task-exchange-v1/README.md).

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

# Hybrid selection: task value and learnability

Completed 2026-11-04. This round separates two questions: does combining a conditioned selector with a shared selector provide more useful evidence than combining two independent shared selectors, and can a small student learn the useful mixture? All comparisons use the existing development panels; final test images and combinations remain reserved.

**The strongest new result is on Flowers with 49 tokens:** the conditional/shared mixture improves classification margin over both original endpoints and over a two-shared mixture. The added benefit is therefore more specific than a generic gain from combining two selectors. Direct hybrid supervision, however, does not make the task-conditioned student outperform the image-only student.

## A: task/shared versus shared/shared mixing

Four settings, all six ordered distinct-seed pairs from seeds 2041–2043, fixed 50% exclusive-token mixing, and almost equal total selector capacity. Three same-seed pairs per setting remain historical references.

| Setting | CU−UU margin ×10³ | CU−UU accuracy, pp |
|---|---:|---:|
| Pets 49 | −0.508 | +0.047 |
| Pets 98 | −0.063 | +0.109 |
| Flowers 49 | +1.004 | +0.225 |
| Flowers 98 | +1.093 | +0.270 |

On Flowers49, the margin difference has a four-setting adjusted interval of [0.489, 1.515] ×10⁻³ and is positive in all six pairs. CU improves margin over C and U by 1.765 and 1.771 ×10⁻³; UU improves over U by 0.767 ×10⁻³. The additional mixing increment is 0.998 [0.507, 1.483] ×10⁻³ (ordinary paired 95% interval).

On Flowers98, final CU margin and accuracy exceed UU, while the extra mixing increment does not clearly exceed UU's: the conditional endpoint already starts stronger. Pets does not reproduce the margin advantage. Fixed d=2/4 interventions and actual replacement counts are retained in the detailed results; the main Flowers49 conclusion concerns the complete fixed mixing rule.

[Full A results and figure](../hybrid-specificity-v1/README.md)

## B: learning the hybrid selections

Eighteen new fits cover Pets49, Flowers49 and Flowers98, two residual branches and three fixed seeds. Both receive the same hybrid-mask target distribution. Comparison with the 18 existing original-target fits keeps the architecture, frozen shared base, sample order, optimizer and update count unchanged.

The task student has lower mean margin than the same-size image student in all three settings. No setting establishes a task-student accuracy advantage. Changing the target does not produce a clear target-by-conditioning interaction.

There is a limited positive transfer result at Flowers98: the image-only student improves margin over the frozen U by 0.389 [0.142, 0.643] ×10⁻³ and over the original-target image student by 0.512 ×10⁻³. It still falls below the hybrid teacher and does not improve accuracy.

A useful new diagnostic appears at Flowers49: both branches fit the **same hybrid-target distribution** much better, but their top-k mask overlap falls slightly on training and validation data. The obstacle is therefore not adequately described as a validation-only generalization gap. Probability-distribution fitting and useful token ranking need to be measured separately.

[Full B results and figures](../hybrid-distillation-v1/README.md)

## Contribution and next step

This extends proposal H1 and H3 from I20/I22 (shared/conditional complementarity) and I26 (learned residual correction). A strengthens the evidence that useful task-related complementarity exists in a controlled setting. B identifies where the current training route fails to recover it. This is positive progress for the research argument; a broadly better deployable selector has not emerged.

The next focused question is whether the hybrid-teacher advantage can be preserved by learning token membership more directly. Before another training sweep, separate changes in probability scale from changes in ranking using the saved logits. A subsequent ranking-target experiment should modify both task and image arms together, keep the strong shared endpoint, and measure mask recovery as well as classification. This is a proposed next step, not an experiment completed in this round.

## Verification, cost and recovery

All ten targeted tests and seven independent raw-result audits pass. The A audit reconstructs masks, endpoints, image-weighted means, fixed-count eligibility and primary intervals. The B audit checks paired training settings, target masks, raw validation logits, target fit, classification means and intervals. Complete original results, all fixed seeds and contrary settings are retained.

[Cost summary](cost_summary.json) separates target generation, cached training and suffix replay. [Run record](../../docs/experiments/2026-11-04-hybrid-followup.md) records the source version and commands. Complete raw arrays and weights are saved in server/local ZIP archives and GitHub Release attachments; [backup manifest](backup_manifest.json) records their scope and restoration check.

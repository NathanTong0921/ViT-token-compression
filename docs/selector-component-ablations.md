# Image-only capacity and task-descriptor components

Protocol fixed on 2026-10-29 before new training or evaluation. This follows the teaching-assistant feedback relayed by the user: strengthen explanation through ablations when accuracy improvements are small, and implement baselines faithfully. It extends proposal H3 and connects descriptor usage to H2.

## Two questions

1. Does candidate conditioning outperform an image-only network that allocates its parameter budget to image processing?
2. Which part of the unordered class-pair descriptor contributes to classification and task-specific selection: the mean text embedding, the absolute difference, or their combination?

Use CLIP ViT-B/16, block 6, Pets and Flowers102, retaining 49/196 or 98/196 patches. These four settings are fixed before outcomes. Original KL-trained conditioned and constant-context image-only models are reused unchanged. Previous ranking-objective, Pets147 and L/14 results remain available but are not selected as per-setting winners.

## Models

| Variant | Input and architecture | Status |
| --- | --- | --- |
| Conditioned | Patch + CLS + [text mean, absolute text difference], width 128 | Existing KL checkpoint |
| Image-only | Same architecture, full text descriptor replaced by its training-pair mean | Existing KL checkpoint |
| Image-only wide | Patch + CLS only, width 199, no text/context branch | New |
| Mean-only | Original width 128; absolute-difference half replaced by its training-pair mean | New |
| Difference-only | Original width 128; text-mean half replaced by its training-pair mean | New |

The original network has 348,033 trainable parameters. The wider image-only network has 347,399 (0.18% fewer). Its extra width uses the budget freed by removing the constant context branch. Keep the original image-only arm as the direct information-removal control, and report both baselines.

Each B/16 descriptor half has 512 dimensions. Replace the missing half before the existing 1024-dimensional context LayerNorm. Compute replacement constants from training class pairs only, using the same shared descriptor as the original model. The two component variants have the original architecture and initial tensors. The wider model uses the paired seed but has different tensor shapes; its initialization is not claimed to match weight by weight. All descriptors are invariant to candidate ordering. The models receive both candidate names, never an indication of which is correct.

Mean-only and difference-only name the retained inputs; their semantic roles are measured rather than assumed. Evaluate both variants after retraining. Existing correct-versus-exchanged-description results are complementary fixed-model interventions.

## Training and execution

- Reuse each budget's frozen prefix cache and existing 512-mask ridge teacher. No new image processing or teacher search is required.
- Use the original KL supervision, temperature 1, full image coverage, float32 selector training, AdamW learning rate 0.0003, weight decay 0.01, batch 128.
- Seeds 2041/2042/2043. Use the original balanced task/class/image stream and final update: Pets 3,990 updates on 2,944 images; Flowers 3,978 updates on 1,020 images.
- Three new variants × three seeds × four settings = 36 models. No validation checkpoint selection or replacement seeds. Record common train/validation KL, parameter counts, sampling records, initialization scope and costs.
- Run actual pruned CLIP classification with the existing stable token ordering and 64-row evaluation batches. Save all-class cosine scores, image embeddings, retained positions and selection scores. The wider model chooses once per image and reuses that selection for every task.
- Use only the existing training/development splits and unseen development class combinations. Reserved test images and combinations are not accessed in this round.

## Primary comparisons

Use all available unseen development combinations. Preserve the existing task-equal/class-within-task-equal weighting. First average the three seeds' per-example outcomes, then bootstrap whole images with all their task rows (2,000 draws, seed 2091). Report accuracy in percentage points and signed normalized geometric margin.

- **Capacity comparison:** conditioned minus image-only wide. Two-budget adjustment separately per dataset and metric (97.5% individual intervals).
- **Component comparisons:** conditioned minus mean-only and conditioned minus difference-only. Four-comparison adjustment across two removed components × two budgets, separately per dataset and metric (98.75% individual intervals).

Preserve ordinary 95% intervals, absolute performance, and all per-seed results. Wide minus original image-only and mean-only minus difference-only are secondary contrasts. Report the observed differences, including improvements by the stronger baseline.

## Task-exchange analysis

Within the same image, hold the evaluated binary task fixed and exchange the descriptor supplied to each selector. Use the saved all-class scores to reconstruct both directions for every eligible pair of unseen tasks sharing the image label. Evaluate full conditioning, mean-only and difference-only on the same image/task-pair panel; verify that both image-only variants are invariant.

For each variant, summarize the symmetric own-minus-exchanged margin and accuracy, averaging seeds and task pairs within images, then images equally. Compare full-conditioning exchange effects with each component variant using paired image-bootstrap ordinary 95% intervals (seed 2092). These explanatory contrasts are secondary, separate from the primary task/class-balanced classification estimand. Preserve token overlap and per-seed exchange effects. A new H2 regression matrix is not needed for these ablations.

## Deliverables

Produce a five-method performance table, paired primary contrasts, a component/exchange figure, training-fit summaries and an independent reconstruction from raw predictions. Retain all 36 checkpoints and raw scientific predictions locally, on the server and in a GitHub Release, with compact results and figures in Git. Summarize the contribution to H3 and its connection to H2, then use the complete development evidence to freeze final-test claims and procedures.

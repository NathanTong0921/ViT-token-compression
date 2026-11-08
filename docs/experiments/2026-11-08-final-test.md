# Reserved-test confirmation

## Question and fixed comparisons

Evaluate the current selectors on the original reserved images and candidate-class pairs before starting optional extensions. This confirms the main study: whether task conditioning improves selection beyond image-only capacity, whether exchanging the candidate task changes classification, and how these results depend on the token budget.

The configuration is `configs/final_test_v1.json`. It is fixed before final-test inference. Use all official test images and all original test pairs: Pets has 3,669 images and 68 pairs; Flowers has 6,149 images and 204 pairs. The Pets pairs cover 3,569 images (13,529 image-task rows); the Flowers pairs cover all 6,149 images (24,596 rows). The classes are known during selector training; the combinations are held out. All three existing training seeds (2041–2043) are averaged. No new fitting, teacher generation, or checkpoint selection is performed on test data.

| Setting | Retained patches | Trained comparisons |
|---|---|---|
| Pets, B/16, block 6 | 49, 98, 147 | Original conditioned C and image-only I; strong shared U at 49/98; PR at 98 |
| Flowers, B/16, block 6 | 49, 98 | C, I, U; PR at 98 |
| Pets and Flowers, L/14, block 12 | 128 | Existing C and I |

All settings also include full CLIP, attention selection, and three fixed random-selection seeds (2055–2057). C/I/U use their existing final trained weights; PR uses the minimum-development-KL checkpoints retained in the preceding comparison. The L/14 and Pets147 results compare C with the original I, not with the wider U.

## Measurements

- Classification: equal weight per candidate task and per class side, then average seeds. Report accuracy, geometric margin, and errors introduced or repaired relative to full CLIP. Bootstrap paired images, retaining their task rows together.
- Main contrasts: C−U across four B/16 settings; C−I across seven settings; U−PR and C−PR across the two 98-token settings. Report ordinary 95% intervals and separately labelled family-adjusted intervals for those fixed groups.
- Budget dependence: compare the C−U difference at 98 versus 49 patches directly on the same examples, separately for each dataset; retain Pets147 in the C−I budget comparison.
- Task exchange and H2: use the same image under two test tasks, exchange the resulting selected-feature vectors, and reuse the existing geometric-margin and difficulty-adjustment definitions. This requires no additional learned model or new target.

The seven settings and current weights are the scope of this test. Previous exploratory loss, identity, hybrid, and depth experiments remain development analyses. Deployment latency is taken from the completed measurement study; this run measures evaluation cost only.

## Execution

First check the new evaluation path against saved development predictions. Then extract test prefixes, check full-token reconstruction against native CLIP, and run the fixed methods. Cache prefixes only on the server; retain predictions, selections, image features, summaries and figures locally and in GitHub. Optional extensions continue on development data after this confirmation.

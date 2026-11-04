# Conditional/shared mixing versus two shared selectors

Fixed before reading new outcomes, 2026-11-04. This extends the equal-replacement
experiment. Its Flowers mixtures improved geometric margins over both endpoints,
but the contribution of candidate information was not separated from generic
complementarity between two selectors. This experiment tests that distinction
without new selector training.

## Primary comparison

Use all existing fixed seeds 2041, 2042, 2043, and all six ordered distinct-seed
pairs (s,t), at Pets49, Pets98, Flowers49, Flowers98. Compare:

- **C_s + U_t:** recipient conditioned selector from seed s; donor wide shared
  selector from seed t.
- **U_s + U_t:** recipient wide shared selector from seed s; the same donor U_t.

Both methods evaluate two selectors, retain k patches, and run one frozen CLIP
suffix. Selector parameter totals are 695,432 and 694,798 respectively. The
backbone, layer and token budget are unchanged. This counts resources; it is not
a measured full-inference latency claim.

In each pair, keep the intersection of recipient and donor top-k sets. If their
exclusive sets have size n, remove floor(n/2) lowest-ranked recipient-only tokens
and insert floor(n/2) highest-ranked donor-only tokens. Preserve spatial order.
The same rule applies to both arms. The actual number of replacements can differ
because their disagreement sizes differ, so both are reported per pair and row.
U_s + U_t must remain independent of the candidate task.

Using distinct seeds for both arms avoids comparing same-seed C/U coordination
against cross-seed U/U diversity. The three same-seed C_s+U_s mixtures are retained
as historical references only, with exact mask and score comparison against
`matched-replacement-v1`'s top_50 outputs. Neither same-seed nor a favorable seed
pair is substituted for the primary comparison.

## Secondary checks

1. Report mixture-minus-recipient and mixture-minus-donor. The incremental
   interaction is `(CU-C_s) - (UU-U_s)`, which distinguishes final performance
   from improvement beyond each recipient. These answer different questions.
2. Repeat the rule with exactly d=2 and d=4 replacements. For each d, include only
   pair/image/task rows for which both arms have at least d exclusive tokens.
   Report weighted coverage, image/row counts and all pair-level results. These
   are restricted-subset diagnostics, not replacements for the full-panel result.
   Stored masks on ineligible rows are recipient placeholders, excluded from this
   analysis. No task-dependent d is supplied to the shared baseline.
3. Exchange the task-conditioned selected sets while holding evaluated tasks
   fixed. Shared endpoints and U+U must have exactly zero exchange effect.

## Statistics and evidence

Primary outcomes: geometric margin, with balanced accuracy also reported. Use
four-setting adjusted intervals separately for the two metrics. Within each row,
average outcomes over all six fixed seed pairs before paired image bootstrap
(2,000 draws, seed2311). Pairs share models and are not independent replicates.
Retain ordinary intervals and every pair's point estimate. Classification uses
existing equal-task/equal-class weights; exchange uses equal images then task
pairs within image.

Fixed-d estimates retain original task/class row weights and equal pair weights,
restrict jointly feasible pair/rows, then renormalize the retained weights.
Collapse pair contributions before image resampling so repeated images across
pairs are not treated as new independent images. Eligibility and coverage can
vary between pairs; those diagnostics describe the retained population only.

Reuse exact cached image/mask embeddings and run only new masks through frozen
CLIP suffix. Independently check all original C/U endpoints and historical
same-seed mixtures. Save raw cosine scores, retained indices, eligibility,
displacements, per-image statistical contributions, all pair summaries, figures,
and execution costs. Formal test images and test candidate pairs remain unused.

## Execution

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_hybrid_specificity.py \
  --setting flowers_b16_49 --output runs/hybrid-specificity-v1/flowers_b16_49 --device cuda:2
PYTHONPATH=src .venv/bin/python scripts/analyze_hybrid_specificity.py \
  --source runs/hybrid-specificity-v1/flowers_b16_49 \
  --output runs/hybrid-specificity-v1/flowers_b16_49/analysis
PYTHONPATH=src:scripts:tests python3 -m unittest test_hybrid_specificity -q
```

Repeat all four declared settings. Curated local results put each analysis summary
at `results/hybrid-specificity-v1/{setting}/summary.json`; the plotting script reads
that layout. This experiment informs the method line alongside, rather than as a
prerequisite for, the separately authorized hybrid-supervision paired training.

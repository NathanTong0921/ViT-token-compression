"""Summarize the prespecified Flowers replication without reselecting outcomes."""
import argparse
import json
from pathlib import Path


def read_results(folder):
    return (json.loads((folder / 'analysis.json').read_text()),
            json.loads((folder / 'geometry' / 'analysis.json').read_text()))


def summarize(flowers):
    main, geometry = flowers
    unseen = main['task_splits']['unseen']
    effects = {k: unseen['budgets'][k]['conditioned_minus_image_only'] for k in ('49', '98')}
    contrast = unseen['secondary_change_in_conditioning_advantage']['98_minus_49']['margin']
    slopes = {k: geometry['budgets'][k]['primary_difficulty_adjusted'] for k in ('49', '98')}
    return {
        'scope': 'Targeted Flowers102 replication of the Pets 49/98 contrast and H2; retrained selectors, not direct model transfer.',
        'protocol': 'docs/flowers-targeted-validation.md',
        'selection_context': 'Budget choice used the previous Pets findings. Both Flowers budgets and all fixed seeds are retained; Pets 147 remains reported.',
        'unseen_image_count': unseen['image_count'],
        'unseen_task_count': unseen['task_count'],
        'budget_effects': effects,
        'prespecified_margin_budget_contrast': contrast,
        'positive_budget_contrast_supported': contrast['ci95'][0] > 0,
        'pets_margin_sign_pattern_replicated': (
            effects['49']['margin']['bonferroni_family_ci95'][1] < 0
            and effects['98']['margin']['bonferroni_family_ci95'][0] > 0),
        'positive_98_margin_benefit_supported': effects['98']['margin']['bonferroni_family_ci95'][0] > 0,
        'h2_positive_at_both_budgets': all(slopes[k]['positive_after_budget_adjustment'] for k in slopes),
        'h2': {k: {'angle_slope': slopes[k]['angle_slope'], 'ci95': slopes[k]['ci95'],
                   'ci_familywise_budgets': slopes[k]['ci_familywise_budgets'],
                   'image_fixed_effects': geometry['budgets'][k]['image_fixed_effects_sensitivity'],
                   'held_image_prediction': geometry['budgets'][k]['held_image_prediction']}
               for k in slopes},
        'between_dataset_limit': 'Training image counts, classes, task graphs and prompts differ; between-dataset effect differences do not isolate a causal dataset factor.',
        'uncertainty': 'Per-budget decisions use the predeclared two-budget adjusted intervals. The single direct margin contrast uses its 95% interval. All intervals condition on the fixed task graphs and training seeds.'}


def plot(datasets, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    specifications = [('margin', 1000, 'Conditioning: margin', 'Conditioned − image-only (×10⁻³)'),
                      ('accuracy', 100, 'Conditioning: accuracy', 'Conditioned − image-only (pp)'),
                      ('angle_slope', 1000, 'H2: adjusted exchange loss', 'Slope per 10° (×10⁻³)')]
    for index, (name, (main, geometry)) in enumerate(datasets.items()):
        budgets = [int(k) for k in main['task_splits']['unseen']['budgets']]
        positions = np.array([([49, 98, 147].index(k)) for k in budgets]) + (-.10 if index == 0 else .10)
        for ax, (metric, scale, title, ylabel) in zip(axes, specifications):
            entries = [(geometry['budgets'][str(k)]['primary_difficulty_adjusted'] if metric == 'angle_slope'
                        else main['task_splits']['unseen']['budgets'][str(k)]['conditioned_minus_image_only'][metric])
                       for k in budgets]
            values = np.array([entry['angle_slope' if metric == 'angle_slope' else 'mean'] for entry in entries]) * scale
            intervals = np.array([entry['ci95'] for entry in entries]) * scale
            ax.errorbar(positions, values, yerr=np.vstack([values - intervals[:, 0], intervals[:, 1] - values]),
                        fmt='o', capsize=4, linewidth=1.5, color=['#28609c', '#ce7829'][index], label=name)
            ax.axhline(0, color='#777777', linewidth=.8)
            ax.set(xticks=[0, 1, 2], xticklabels=['49', '98', '147'], xlabel='Retained patch tokens', ylabel=ylabel, title=title)
            ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle('Pets and Flowers102: frozen CLIP ViT-B/16, three fixed training seeds', fontsize=12)
    fig.text(.5, .015, 'Bars: paired image-bootstrap 95% intervals. Flowers 147 was not run; adjusted inference is reported separately.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .06, 1, .93])
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'cross_dataset_replication.{extension}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pets', type=Path, default=Path('results/pets-selector-budgets-v1'))
    parser.add_argument('--flowers', type=Path, default=Path('results/flowers-selector-budgets-v1'))
    args = parser.parse_args()
    pets, flowers = read_results(args.pets), read_results(args.flowers)
    result = summarize(flowers)
    (args.flowers / 'replication_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    plot({'Pets': pets, 'Flowers102': flowers}, args.flowers)
    print(json.dumps({k: v for k, v in result.items() if isinstance(v, bool)}, indent=2))

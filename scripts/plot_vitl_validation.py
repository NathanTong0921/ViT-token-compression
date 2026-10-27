"""Plot the targeted 50%-patch comparison across the two frozen CLIP backbones."""
import argparse
import csv
import json
from pathlib import Path


SOURCES = (
    ('Pets', 'ViT-B/16', 'pets-selector-budgets-v1', 98, 196),
    ('Pets', 'ViT-L/14', 'pets-vitl128-v1', 128, 256),
    ('Flowers102', 'ViT-B/16', 'flowers-selector-budgets-v1', 98, 196),
    ('Flowers102', 'ViT-L/14', 'flowers-vitl128-v1', 128, 256),
)
PANELS = (
    ('margin', 1000, 'Classification margin', 'Conditioned − image-only (×10⁻³)'),
    ('accuracy', 100, 'Classification accuracy', 'Conditioned − image-only (percentage points)'),
    ('angle_slope', 1000, 'H2: difficulty-adjusted exchange loss', 'Slope per 10° (×10⁻³)'),
)


def source_rows(results_root, dataset, backbone, folder_name, budget, patches):
    folder = results_root / folder_name
    main_path, geometry_path = folder / 'analysis.json', folder / 'geometry' / 'analysis.json'
    main = json.loads(main_path.read_text())
    geometry = json.loads(geometry_path.read_text())
    config = main['sources'][str(budget)]['config']
    assert config['model'] == backbone and config['keep_patch_tokens'] == budget
    main_effects = main['task_splits']['unseen']['budgets'][str(budget)]['conditioned_minus_image_only']
    h2 = geometry['budgets'][str(budget)]['primary_difficulty_adjusted']
    rows = []
    for metric, scale, _, _ in PANELS:
        entry = h2 if metric == 'angle_slope' else main_effects[metric]
        estimate = entry['angle_slope' if metric == 'angle_slope' else 'mean']
        if backbone == 'ViT-L/14':
            interval, level = entry['ci95'], .95
            adjustment = 'One predeclared budget within this dataset'
        elif metric == 'angle_slope':
            key = ('ci_familywise_budgets' if 'ci_familywise_budgets' in entry
                   else 'ci_familywise_three_budgets')
            interval, level = entry[key], entry['familywise_interval_level']
            adjustment = 'Adjusted across the original B/16 budget family within this dataset'
        else:
            interval, level = entry['bonferroni_family_ci95'], entry['individual_interval_level']
            adjustment = 'Adjusted across the original B/16 budget family within this dataset'
        rows.append({
            'dataset': dataset, 'backbone': backbone, 'retained_patches': budget,
            'original_patches': patches, 'prune_after_block': config['prune_after'],
            'metric': metric, 'estimate': estimate, 'ci_low': interval[0], 'ci_high': interval[1],
            'interval_level': level, 'adjustment': adjustment, 'plot_scale': scale,
            'estimate_in_plot_units': estimate * scale,
            'ci_low_in_plot_units': interval[0] * scale, 'ci_high_in_plot_units': interval[1] * scale,
            'source': str(geometry_path if metric == 'angle_slope' else main_path),
        })
    return rows


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    styles = {'ViT-B/16': ('#28609c', 'o', -.10), 'ViT-L/14': ('#ce7829', 's', .10)}
    datasets = ['Pets', 'Flowers102']
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.8))
    for ax, (metric, _, title, ylabel) in zip(axes, PANELS):
        for row in [r for r in rows if r['metric'] == metric]:
            color, marker, offset = styles[row['backbone']]
            x = datasets.index(row['dataset']) + offset
            low, high = row['ci_low_in_plot_units'], row['ci_high_in_plot_units']
            ax.vlines(x, low, high, color=color, linewidth=1.6)
            ax.hlines([low, high], x - .035, x + .035, color=color, linewidth=1.6)
            ax.plot(x, row['estimate_in_plot_units'], marker, color=color, markersize=6)
        ax.axhline(0, color='#777777', linewidth=.8)
        ax.set(xticks=[0, 1], xticklabels=datasets, xlim=(-.4, 1.4), ylabel=ylabel, title=title)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    handles = [Line2D([], [], marker=styles[b][1], color=styles[b][0], linestyle='none',
                      label=label) for b, label in (
                          ('ViT-B/16', 'ViT-B/16: 98/196 patches, after block 6'),
                          ('ViT-L/14', 'ViT-L/14: 128/256 patches, after block 12'))]
    fig.suptitle('Targeted cross-backbone validation at a 50% patch budget', fontsize=13, y=.99)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .93), ncol=2, frameon=False)
    b_levels = {r['dataset']: r['interval_level'] for r in rows if r['backbone'] == 'ViT-B/16'}
    fig.text(.5, .065,
             f"B/16: budget-adjusted intervals (Pets {100*b_levels['Pets']:.2f}%; Flowers {100*b_levels['Flowers102']:.2f}%). "
             'L/14: single-budget 95% intervals.', ha='center', fontsize=9)
    fig.text(.5, .025, 'Three fixed seeds; image-cluster bootstrap. Backbone changes include patch size and depth, not model size alone.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .12, 1, .85])
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'cross_backbone_validation.{extension}', dpi=200)
    plt.close(fig)


def main(results_root, output):
    rows = [row for source in SOURCES for row in source_rows(results_root, *source)]
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'cross_backbone_plot_table.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, output)
    print(json.dumps({'comparison_rows': len(rows), 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, default=Path('results/vitl-and-latency-v1'))
    args = parser.parse_args()
    main(args.results_root, args.output)

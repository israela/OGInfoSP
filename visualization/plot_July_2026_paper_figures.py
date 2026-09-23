"""Generate the planned July 2026 Gaussian and CIFAR paper figures."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


RESULTS_ROOT = Path('output/July_2026')
FIGURES_DIR = Path('output/July_2026_figures')

METHOD_LABELS = {
    'OGInfoSP': 'OGInfoSP',
    'OGInfoSPCal': 'OGInfoSP-Cal',
    'InfoSP': 'InfoSP',
    'InfoSCOP': 'InfoSCOP',
    'ClassicConformal': 'Classic Conformal',
    'AdaptiveClassicConformal': 'Classic Conformal\nAdaptive Score',
    'TopClass': 'Top Class',
}
METHOD_COLORS = {
    'OGInfoSP': {'none': '#1f4e8a', 'only_bias': '#1f4e8a', 'full': '#1f4e8a'},
    'OGInfoSPCal': {'none': '#1a7a3a', 'only_bias': '#1a7a3a', 'full': '#1a7a3a'},
    'InfoSP': {'none': '#b81d1d', 'only_bias': '#b81d1d', 'full': '#b81d1d'},
    'InfoSCOP': {'none': '#6a3d9a', 'only_bias': '#6a3d9a', 'full': '#6a3d9a'},
    'ClassicConformal': {'none': '#6b3a1f', 'only_bias': '#6b3a1f', 'full': '#6b3a1f'},
    'AdaptiveClassicConformal': {'none': '#0e7c86', 'only_bias': '#0e7c86', 'full': '#0e7c86'},
    'TopClass': {'none': '#c4197d', 'only_bias': '#c4197d', 'full': '#c4197d'},
}
METHOD_LINE_STYLES = {
    'OGInfoSP': '-',
    'OGInfoSPCal': '--',
    'InfoSP': '-.',
    'InfoSCOP': ':',
    'ClassicConformal': (0, (7, 3)),
    'AdaptiveClassicConformal': (0, (3, 1, 1, 1)),
    'TopClass': (0, (1, 2)),
}
VS_DOT_PHASES = {
    'OGInfoSP': 0,
    'OGInfoSPCal': 1.2,
    'InfoSP': 2.4,
    'InfoSCOP': 3.6,
    'ClassicConformal': 4.8,
    'AdaptiveClassicConformal': 0.8,
    'TopClass': 2.8,
}
METHOD_ZORDERS = {
    'OGInfoSP': 2,
    'ClassicConformal': 3,
    'OGInfoSPCal': 4,
    'InfoSP': 5,
    'InfoSCOP': 6,
}
VS_LABELS = {'none': 'No VS', 'only_bias': 'Bias VS', 'full': 'Full VS'}
VS_HATCHES = {'none': '', 'only_bias': '..', 'full': '//'}
VS_LINE_WIDTHS = {'none': 1.9, 'only_bias': 2.0, 'full': 2.0}

GAUSSIAN_METHODS = ('OGInfoSP', 'OGInfoSPCal', 'InfoSP', 'InfoSCOP', 'ClassicConformal')
SET_2_SELECTED_COMBINATIONS = tuple(
    (method, scaling)
    for method in GAUSSIAN_METHODS
    for scaling in ('only_bias', 'none')
    if scaling == 'only_bias' or method in ('OGInfoSP', 'OGInfoSPCal')
)
SET_2_INFO_COMBINATIONS = tuple(
    (method, scaling)
    for method in ('InfoSP', 'InfoSCOP', 'ClassicConformal')
    for scaling in ('none', 'only_bias')
)
SET_2_COMPARISONS = (
    ('all_vs-og_no_vs', SET_2_SELECTED_COMBINATIONS),
    ('infosp-infoscop-cc_with-without-vs', SET_2_INFO_COMBINATIONS),
)
CIFAR_ALL_METHODS = ('OGInfoSP', 'OGInfoSPCal', 'InfoSP', 'InfoSCOP',
                     'ClassicConformal', 'AdaptiveClassicConformal', 'TopClass')
CIFAR_SET_5_METHODS = tuple(
    method for method in CIFAR_ALL_METHODS if method != 'AdaptiveClassicConformal'
)
CIFAR_SELECTED_METHODS = ('OGInfoSP', 'OGInfoSPCal', 'InfoSP', 'InfoSCOP', 'TopClass')

GAUSSIAN_SCALINGS = {'none': (500, 0), 'only_bias': (400, 100)}
CIFAR_SCALINGS = {'none': (1000, 0), 'full': (500, 500)}

GAUSSIAN_SETS = (
    {
        'number': 1, 'K': 10,
        'informative_types': ('up_to_3', 'exclude_1'),
        'prob_types': ('trained_uneven_test_uneven',),
        'scaling_groups': (('none',),),
    },
    {
        'number': 2, 'K': 10,
        'informative_types': ('up_to_3', 'exclude_1'),
        'prob_types': ('trained_even_test_uneven', 'trained_uneven_test_even'),
        'scaling_groups': (('none', 'only_bias'), ('only_bias',)),
    },
    {
        'number': 3, 'K': 4,
        'informative_types': ('non_trivial', 'exclude_1'),
        'prob_types': ('trained_uneven_test_uneven',),
        'scaling_groups': (('none',),),
    },
    {
        'number': 4, 'K': 4,
        'informative_types': ('non_trivial', 'exclude_1'),
        'prob_types': ('trained_even_test_uneven', 'trained_uneven_test_even'),
        'scaling_groups': (('none', 'only_bias'), ('only_bias',)),
    },
)

CIFAR_SETS = (
    {
        'number': 5, 'K': 10, 'folder_prefix': 'CIFAR_all10',
        'informative_types': ('single_0_to_6', 'exclude_1'),
        'prob_types': ('even', 'uneven'),
        'scalings': ('full',), 'methods': CIFAR_SET_5_METHODS,
    },
    {
        'number': 6, 'K': 10, 'folder_prefix': 'CIFAR_all10',
        'informative_types': ('single_0_to_6', 'exclude_1'),
        'prob_types': ('even', 'uneven'),
        'scalings': ('none', 'full'), 'methods': CIFAR_SELECTED_METHODS,
    },
    {
        'number': 7, 'K': 3, 'folder_prefix': 'CIFAR_BCD',
        'informative_types': ('non_trivial', 'exclude_1'),
        'prob_types': ('even', 'uneven'),
        'scalings': ('full',), 'methods': CIFAR_ALL_METHODS,
    },
    {
        'number': 8, 'K': 3, 'folder_prefix': 'CIFAR_BCD',
        'informative_types': ('non_trivial', 'exclude_1'),
        'prob_types': ('even', 'uneven'),
        'scalings': ('none', 'full'), 'methods': CIFAR_SELECTED_METHODS,
    },
)


def result_filename(method, dataset, n_cal, n_vs, alpha, iterations, informative_type, scaling):
    scaling_label = scaling.replace('_', '')
    return (
        f'{method}_{dataset}_cal_{n_cal}_vs_{n_vs}_test_500_alpha_{alpha}'
        f'_iterations_{iterations}_informative_type_{informative_type}'
        f'_vectorscaling_{scaling_label}.txt'
    )


def load_result(folder, method, dataset, scaling_args, alpha, iterations, informative_type, scaling):
    n_cal, n_vs = scaling_args[scaling]
    path = folder / result_filename(method, dataset, n_cal, n_vs, alpha, iterations, informative_type, scaling)
    if not path.exists():
        raise FileNotFoundError(f'Missing result file: {path}')
    with path.open('r') as result_file:
        return json.load(result_file)


def decoded_value(result, key):
    value = result[key]
    return json.loads(value) if isinstance(value, str) else value


def load_results(folder, methods, dataset, scaling_args, alpha, iterations, informative_type, scalings):
    return {
        (method, scaling): load_result(
            folder, method, dataset, scaling_args, alpha, iterations,
            informative_type, scaling
        )
        for method in methods
        for scaling in scalings
    }


def save_figure(fig, output_path):
    fig.tight_layout()
    fig.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')


def gaussian_line(method, scaling):
    color = METHOD_COLORS[method][scaling]
    if scaling == 'none':
        linestyle = METHOD_LINE_STYLES[method]
        dash_capstyle = 'butt'
        zorder = METHOD_ZORDERS[method]
    else:
        linestyle = (VS_DOT_PHASES[method], (1, 5))
        dash_capstyle = 'round'
        zorder = 10 + METHOD_ZORDERS[method] / 10
    return {
        'color': color,
        'linestyle': linestyle,
        'linewidth': VS_LINE_WIDTHS[scaling],
        'dash_capstyle': dash_capstyle,
        'zorder': zorder,
    }


def save_gaussian_plot(results, scalings, value_key, ylabel, output_path, alpha=None, combinations=None):
    if combinations is None:
        combinations = [(method, scaling) for method in GAUSSIAN_METHODS
                        for scaling in scalings]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for method, scaling in combinations:
        result = results[method, scaling]
        training_args = decoded_value(result, 'training_args')
        distances = training_args['distances']
        if isinstance(distances, str):
            distances = json.loads(distances)
        values = decoded_value(result, value_key)
        ax.plot(distances, values, **gaussian_line(method, scaling))

    if alpha is not None:
        ax.axhline(alpha, color='#444444', linestyle=(0, (5, 3)), linewidth=1.0)
    ax.set_xlabel('SNR', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, color='#d0d0d0', alpha=0.6, linewidth=0.7)
    save_figure(fig, output_path)


def save_cifar_boxplot(results, methods, scalings, value_key, ylabel, output_path, alpha=None):
    combinations = [(method, scaling) for method in methods for scaling in scalings]
    values = [decoded_value(results[combination], value_key) for combination in combinations]
    multiple_scalings = len(scalings) > 1
    labels = [
        f'{METHOD_LABELS[method]}\n{VS_LABELS[scaling]}'
        if multiple_scalings else METHOD_LABELS[method]
        for method, scaling in combinations
    ]

    fig_width = max(8.5, len(combinations) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    boxplot = ax.boxplot(
        values, tick_labels=labels, patch_artist=True, widths=0.62,
        medianprops={'color': '#111111', 'linewidth': 1.2},
        whiskerprops={'color': '#444444'}, capprops={'color': '#444444'},
        flierprops={'marker': '.', 'markersize': 2.5, 'alpha': 0.35},
    )
    for (method, scaling), box in zip(combinations, boxplot['boxes']):
        box.set_facecolor(METHOD_COLORS[method][scaling])
        box.set_edgecolor('#333333')
        box.set_hatch(VS_HATCHES[scaling])
        box.set_alpha(0.78)

    if alpha is not None:
        ax.axhline(alpha, color='#444444', linestyle=(0, (5, 3)), linewidth=1.0)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis='x', labelrotation=25, labelsize=9)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, axis='y', color='#d0d0d0', alpha=0.6, linewidth=0.7)
    save_figure(fig, output_path)


def create_gaussian_legend(scalings, combinations=None, group_name=None):
    if combinations is None:
        combinations = [(method, scaling) for method in GAUSSIAN_METHODS for scaling in scalings]
    handles = []
    multiple_scalings = len({scaling for _, scaling in combinations}) > 1
    for method, scaling in combinations:
        line_args = gaussian_line(method, scaling)
        line_args.pop('zorder')
        label = METHOD_LABELS[method]
        if multiple_scalings:
            label += f' ({VS_LABELS[scaling]})'
        handles.append(Line2D([], [], label=label, **line_args))

    rows = len(scalings)
    method_count = len(dict.fromkeys(method for method, _ in combinations))
    fig = plt.figure(figsize=(12, max(0.8, rows * 0.55)))
    fig.legend(handles=handles, loc='center', ncol=method_count, frameon=False, fontsize=10, handlelength=5)
    if group_name is None:
        group_name = 'all_methods_no_vs' if scalings == ('none',) else '-'.join(scalings)
    output_path = FIGURES_DIR / 'legends' / f'gaussian_{group_name}_legend.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_path)


def generate_gaussian_figures():
    for plot_set in GAUSSIAN_SETS:
        set_dir = FIGURES_DIR / f'set_{plot_set["number"]:02d}_gaussian_K{plot_set["K"]}'
        set_dir.mkdir(parents=True, exist_ok=True)
        for informative_type in plot_set['informative_types']:
            for prob_type in plot_set['prob_types']:
                folder = RESULTS_ROOT / f'gaussian_K{plot_set["K"]}_{informative_type}_{prob_type}'
                for scalings in plot_set['scaling_groups']:
                    results = load_results(
                        folder, GAUSSIAN_METHODS, 'gaussian', GAUSSIAN_SCALINGS,
                        0.05, 10000, informative_type, scalings
                    )
                    group_name = '-'.join(scalings)
                    stem = f'gaussian_K{plot_set["K"]}_{informative_type}_{prob_type}_{group_name}'
                    save_gaussian_plot(
                        results, scalings, 'FCR_per_distance', 'FCR',
                        set_dir / f'{stem}_FCR.png', alpha=0.05
                    )
                    save_gaussian_plot(
                        results, scalings, 'mean_power_per_distance', 'Res. Adj. Power',
                        set_dir / f'{stem}_Power.png'
                    )
                    if plot_set['number'] in (1, 3):
                        save_gaussian_plot(
                            results, scalings, 'mean_selected_per_distance', 'Selected',
                            set_dir / f'{stem}_Selected.png'
                        )
                        save_gaussian_plot(
                            results, scalings, 'mean_correct_selected_per_distance',
                            'Correct Selected', set_dir / f'{stem}_CorrectSelected.png'
                        )
                if plot_set['number'] == 2:
                    scalings = ('none', 'only_bias')
                    results = load_results(
                        folder, GAUSSIAN_METHODS, 'gaussian', GAUSSIAN_SCALINGS,
                        0.05, 10000, informative_type, scalings
                    )
                    for comparison_name, combinations in SET_2_COMPARISONS:
                        stem = (f'gaussian_K{plot_set["K"]}_{informative_type}_{prob_type}_{comparison_name}')
                        save_gaussian_plot(
                            results, scalings, 'FCR_per_distance', 'FCR',
                            set_dir / f'{stem}_FCR.png', alpha=0.05,
                            combinations=combinations
                        )
                        save_gaussian_plot(
                            results, scalings, 'mean_power_per_distance', 'Res. Adj. Power',
                            set_dir / f'{stem}_Power.png', combinations=combinations
                        )
                        if comparison_name == 'infosp-infoscop-cc_with-without-vs':
                            save_gaussian_plot(
                                results, scalings, 'mean_correct_selected_per_distance',
                                'Correct Selected', set_dir / f'{stem}_CorrectSelected.png',
                                combinations=combinations
                            )


def generate_cifar_figures():
    for plot_set in CIFAR_SETS:
        set_dir = FIGURES_DIR / f'set_{plot_set["number"]:02d}_cifar_K{plot_set["K"]}'
        set_dir.mkdir(parents=True, exist_ok=True)
        for informative_type in plot_set['informative_types']:
            for prob_type in plot_set['prob_types']:
                folder = RESULTS_ROOT / f'{plot_set["folder_prefix"]}_{informative_type}_{prob_type}'
                results = load_results(
                    folder, plot_set['methods'], 'CIFAR10', CIFAR_SCALINGS,
                    0.1, 1000, informative_type, plot_set['scalings']
                )
                group_name = '-'.join(plot_set['scalings'])
                stem = f'cifar_K{plot_set["K"]}_{informative_type}_{prob_type}_{group_name}'
                save_cifar_boxplot(
                    results, plot_set['methods'], plot_set['scalings'], 'all_FCP', 'FCR',
                    set_dir / f'{stem}_FCR.png', alpha=0.1
                )
                save_cifar_boxplot(
                    results, plot_set['methods'], plot_set['scalings'], 'all_power',
                    'Res. Adj. Power', set_dir / f'{stem}_Power.png'
                )


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    gaussian_groups = {group for plot_set in GAUSSIAN_SETS
                       for group in plot_set['scaling_groups']}
    for scalings in sorted(gaussian_groups):
        create_gaussian_legend(scalings)
    for comparison_name, combinations in SET_2_COMPARISONS:
        create_gaussian_legend(
            ('none', 'only_bias'), combinations, comparison_name
        )
    generate_gaussian_figures()
    generate_cifar_figures()


if __name__ == '__main__':
    main()

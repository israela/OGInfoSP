import json
import numpy as np
import dataclasses
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cycler import cycler
from simulation.interfaces import ModelTrainingArgs, DataSamplingArgs, ProcedureArgs, ExecutionArgs, VectorScalingArgs


DEFAULT_STYLES = [('--', 1.2), (':', 1.4), ('--', 1.2), (':', 1.4)]

def plot_multiple_seq(seq_x, label_x, seq_label_pairs, file_name, alpha = None, title = '', label_y = '', include_legend = True):
    plt.figure(figsize=(4, 3))
    for i, (seq, label) in enumerate(seq_label_pairs):
        if seq is None:
            plt.plot([np.nan], [np.nan])
            continue
        style = DEFAULT_STYLES[i % len(DEFAULT_STYLES)]
        plt.plot(seq_x, seq, label=label, linewidth=style[1], linestyle=style[0], zorder=len(seq_label_pairs) - i)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    if alpha is not None:
        plt.axhline(y=alpha, color='red', linestyle='--', linewidth=1)
    if include_legend:
        plt.legend(fontsize='small')
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'{file_name}.jpg', dpi=300)
    plt.close()


def plot_two_seq(seq_x, label_x, seq_1, label_1, seq_2, label_2, file_name, alpha = None, title = '', label_y = '', include_legend = True):
    plot_multiple_seq(seq_x, label_x, [[seq_1, label_1], [seq_2, label_2]], file_name, alpha, title, label_y, include_legend)


def plot_one_seq(seq_x, label_x, seq_1, label_1, file_name, alpha = None, title = '', label_y = '', include_legend = True):
    plot_multiple_seq(seq_x, label_x, [[seq_1, label_1]], file_name, alpha, title, label_y, include_legend)


def plot_diff_seq_scatter(seq_x, label_x, seq_diff, label_diff_1, label_diff_2, file_name, title = ''):
    plt.figure(figsize=(4, 3))
    plt.scatter(seq_x[seq_diff > 0], seq_diff[seq_diff > 0], marker='.', color='green', label=f'{label_diff_1} > {label_diff_2}')
    plt.scatter(seq_x[seq_diff == 0], seq_diff[seq_diff == 0], marker='.', color='blue', label=f'{label_diff_1} == {label_diff_2}')
    plt.scatter(seq_x[seq_diff < 0], seq_diff[seq_diff < 0], marker='.', color='red', label=f'{label_diff_1} < {label_diff_2}')
    plt.xlabel(label_x)
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'{file_name}.jpg', dpi=300)
    plt.close()


def plot_boxplot(data_label_pairs, label_x, label_y, file_name, alpha = None, title = ''):
    plt.figure(figsize=(4, 3))
    data = [d for d, _ in data_label_pairs]
    labels = [label for _, label in data_label_pairs]
    plt.boxplot(data, tick_labels = labels)
    plt.xticks(fontsize=8) 
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    if alpha is not None:
        plt.axhline(y=alpha, color='red', linestyle='--', linewidth=0.6)
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'{file_name}.jpg', dpi=300)
    plt.close()


def create_legend_figure(seq_label_pairs, file_name):
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.axis('off')

    handles = []
    labels = []
    # Plot dummy lines in the same order as the real plots
    for i, (_, label) in enumerate(seq_label_pairs):
        style = DEFAULT_STYLES[i % len(DEFAULT_STYLES)]
        line, = ax.plot([], [], linewidth=style[1], linestyle=style[0])
        handles.append(line)
        labels.append(label)

    fig.legend(handles, labels, loc='center', ncol=len(labels), fontsize='small', frameon=False)
    fig.savefig(f'{file_name}.jpg', bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_data_generation_example(cal_samples, cal_classes, test_samples, file_name, include_test=True):
    plt.figure(figsize=(6, 5))
    unique_classes = np.unique(cal_classes)
    base_colors = ['blue', 'red', 'orange', 'purple', 'brown', 'cyan', 'magenta', 'navy', 'crimson', 'coral']

    if include_test:
        plt.scatter(test_samples[:, 0], test_samples[:, 1], s=20, color='green', alpha=0.6, edgecolor='black', linewidth=0.3, marker='o')

    for idx, cls in enumerate(unique_classes):
        mask = cal_classes == cls
        color = base_colors[idx % len(base_colors)]
        plt.scatter(cal_samples[mask, 0], cal_samples[mask, 1], s=25, color=color, alpha=0.6, edgecolor='black', linewidth=0.3, marker='^')

    plt.grid(True, linewidth=0.3, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{file_name}.jpg', dpi=300)
    plt.close()


def normalize(val):
    if dataclasses.is_dataclass(val):
        val = dataclasses.asdict(val)

    if isinstance(val, dict):
        return {k: normalize(v) for k, v in val.items()}
    if callable(val):
        val = val.method_name
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, list):
        val = json.dumps(val)

    return val


def write_results_generic(file_name, **kwargs):
    data = {k: normalize(v) for k, v in kwargs.items()}

    with open(f'{file_name}.txt', 'w') as f:
        json.dump(data, f, indent=2)


def create_filepath(execution_args: ExecutionArgs, training_args: ModelTrainingArgs, sampling_args: DataSamplingArgs, vector_scaling_args: VectorScalingArgs, procedure_args: ProcedureArgs):
    return (
        f"output/{execution_args.output_folder}/{execution_args.method.method_name}_{training_args.dataset if training_args.kind == 'load' else 'gaussian'}"
        f"_cal_{sampling_args.n_cal}_vs_{vector_scaling_args.n_vs}_test_{sampling_args.n_test}_alpha_{procedure_args.alpha}_iterations_{execution_args.iterations}_informative_type_{procedure_args.informative_type}"
        f"{('_denominator_' + ('cal' if procedure_args.use_cal_denominator else 'test')) if procedure_args.use_cal_denominator is not None else ''}"
        f"{('_vectorscaling_' + vector_scaling_args.type.replace('_', '')) if vector_scaling_args.type is not None else ''}"
        #f"{('_scores_' + ('predicted' if sampling_args.predict_scores else 'trueprob')) if sampling_args.predict_scores is not None else ''}"
        #f"{('_threshold_' + ('include' if procedure_args.include_threshold else 'strict')) if procedure_args.include_threshold is not None else ''}"
        f"{('_denom_' + ('increase' if procedure_args.increase_denominator else 'noincrease')) if procedure_args.increase_denominator is not None else ''}"
    )


def classifier_key(distance, probs):
    return f'{distance}_{json.dumps(probs)}'
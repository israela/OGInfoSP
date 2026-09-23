from simulation.sample_and_train import sample_and_train
from simulation.utils import plot_data_generation_example as plot_data_generation_example_impl
from simulation.interfaces import DefaultGaussiansModelArgs, DataSamplingArgs, VectorScalingArgs


def plot_data_generation_example(probs, SNR, filename):
    training_args = DefaultGaussiansModelArgs('default_gaussians', SNR, probs, 100)  # the classifier is not important for sampling
    sampling_args = DataSamplingArgs(500, 500, probs)
    vector_scaling_args = VectorScalingArgs(type='none', n_vs=0)
    samples = sample_and_train(training_args, sampling_args, vector_scaling_args)
    output_path = f'output/figures/data_generation/{filename}'
    for include_test in (True, False):
        suffix = '' if include_test else '_no_test'
        plot_data_generation_example_impl(samples.cal_data_samples, samples.cal_data_classes, samples.test_data_samples, output_path + suffix, include_test)


if __name__ == "__main__":
    plot_data_generation_example([0.33, 0.33, 0.34], 2, 'data_generation_example_K3_even_snr_2')
    plot_data_generation_example([0.25, 0.25, 0.25, 0.25], 2, 'data_generation_example_K4_even_snr_2')
    plot_data_generation_example([0.25, 0.25, 0.25, 0.25], 2.5, 'data_generation_example_K4_even_snr_25')
    plot_data_generation_example([0.1, 0.7, 0.1, 0.1], 2, 'data_generation_example_K4_uneven_snr_2')
    plot_data_generation_example([0.1, 0.7, 0.1, 0.1], 2.5, 'data_generation_example_K4_uneven_snr_25')

    for SNR in [2, 2.5, 3, 5]:
        SNR_label = str(SNR).replace('.', '')
        plot_data_generation_example([0.15, 0.45] + [0.05] * 8, SNR, f'data_generation_example_K10_uneven_snr_{SNR_label}')

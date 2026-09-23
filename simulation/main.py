from itertools import combinations
import time

from simulation.algorithms.OGInfoSP_algo import OGInfoSP, OGInfoSPCal
from simulation.algorithms.InfoSP_algo import InfoSP_algo
from simulation.algorithms.InfoSCOP_algo import InfoSCOP_algo
from simulation.algorithms.classic_conformal_algo import classic_conformal_algo
from simulation.algorithms.adaptive_classic_conformal_algo import adaptive_classic_conformal_algo
from simulation.algorithms.top_class_algo import top_class_algo
from simulation.interfaces import DataSamplingArgs, DefaultGaussiansSeriesModelArgs, ExecutionArgs, LoadModelArgs, ProcedureArgs, VectorScalingArgs
from simulation.run_and_plot import run_and_plot_distribution, run_and_plot_per_distance
from simulation.sample_and_train import train_default_gaussian
from simulation.utils import classifier_key


def get_cal_split(n_calibration, n_vector_scaling, vector_scaling):
    if vector_scaling == 'none':
        return n_calibration, 0
    if n_vector_scaling >= n_calibration:
        raise ValueError('n_vector_scaling must be smaller than n_calibration')
    n_vs = 0 if vector_scaling == 'none_reduced_cal' else n_vector_scaling
    return n_calibration - n_vector_scaling, n_vs


def get_informative_sets(K, informative_type):
    if K < 2:
        raise ValueError('K must be at least 2')

    if informative_type == 'non_trivial':
        classes = range(K)
        max_size = K - 1
    elif informative_type == 'exclude_1':
        classes = [i for i in range(K) if i != 1]
        max_size = len(classes)
    elif informative_type == 'up_to_3':
        classes = range(K)
        max_size = min(3, K)
    elif informative_type == 'single_0_to_6':
        if K < 7:
            raise ValueError("informative_type 'single_0_to_6' requires K >= 7")
        return [[i] for i in range(7)]
    else:
        raise ValueError(f'Unsupported informative_type: {informative_type}')

    return [
        list(informative_set)
        for size in range(1, max_size + 1)
        for informative_set in combinations(classes, size)
    ]


def _validate_probs(K, probs, prob_type):
    if len(probs) != K:
        raise ValueError(f"Probability type '{prob_type}' has {len(probs)} classes; expected K={K}")


def train_gaussian_cache(distances, probability_vectors, n_train):
    return {classifier_key(distance, probs): train_default_gaussian(distance, probs, n_train)
            for distance in distances for probs in probability_vectors}


def run_methods_for_params(*, dataset_name, K, methods, training_args, sampling_probs, n_cal, n_test, n_vs, vector_scaling,
                           informative_type, alpha, output_folder, iterations, max_workers, trained_gaussian_model_args_cache):
    informative_sets = get_informative_sets(K, informative_type)
    sampling_args = DataSamplingArgs(n_cal=n_cal, n_test=n_test, probs=sampling_probs)
    vector_scaling_args = VectorScalingArgs(type=vector_scaling, n_vs=n_vs)

    for method in methods:
        print(dataset_name, f'K={K}', informative_type, method.method_name, vector_scaling, output_folder)
        execution_args = ExecutionArgs(method=method, output_folder=output_folder, iterations=iterations, max_workers=max_workers)
        procedure_args = ProcedureArgs(informative_sets=informative_sets, informative_type=informative_type, alpha=alpha)

        if training_args.kind == 'default_gaussians_series':
            run_and_plot_per_distance(execution_args, training_args, sampling_args, vector_scaling_args, procedure_args, trained_gaussian_model_args_cache)
        else:
            run_and_plot_distribution(execution_args, training_args, sampling_args, vector_scaling_args, procedure_args)


def run_gaussian_experiment(*, K, informative_types, sizes, alphas, prob_types, methods, iterations, vector_scalings, n_vector_scaling,
                            distances, n_train, max_workers, output_folder, trained_gaussian_model_args_cache):
    if not trained_gaussian_model_args_cache:
        raise ValueError('A nonempty trained_gaussian_model_args_cache is required for Gaussian experiments')

    for prob_type, (training_probs, test_probs) in prob_types.items():
        _validate_probs(K, training_probs, prob_type)
        _validate_probs(K, test_probs, prob_type)

    for informative_type in informative_types:
        get_informative_sets(K, informative_type)
        for size in sizes:
            for alpha in alphas:
                for vector_scaling in vector_scalings:
                    n_cal, n_vs = get_cal_split(size, n_vector_scaling, vector_scaling)
                    for prob_type, (training_probs, test_probs) in prob_types.items():
                        training_args = DefaultGaussiansSeriesModelArgs('default_gaussians_series', distances, training_probs, n_train)
                        run_methods_for_params(
                            dataset_name='gaussian', K=K, methods=methods, training_args=training_args, sampling_probs=test_probs,
                            n_cal=n_cal, n_test=size, n_vs=n_vs, vector_scaling=vector_scaling,
                            informative_type=informative_type, alpha=alpha,
                            output_folder=output_folder.format(K=K, informative_type=informative_type, prob_type=prob_type),
                            iterations=iterations, max_workers=max_workers,
                            trained_gaussian_model_args_cache=trained_gaussian_model_args_cache)


def run_cifar_experiment(*, K, model_dir, informative_types, sizes, alphas, prob_types, methods,
                         iterations, vector_scalings, n_calibration, n_vector_scaling, max_workers, output_folder):
    for prob_type, probs in prob_types.items():
        _validate_probs(K, probs, prob_type)

    training_args = LoadModelArgs(kind='load', dataset='CIFAR10', model_dir=model_dir)
    for informative_type in informative_types:
        get_informative_sets(K, informative_type)
        for size in sizes:
            for alpha in alphas:
                for vector_scaling in vector_scalings:
                    n_cal, n_vs = get_cal_split(n_calibration, n_vector_scaling, vector_scaling)
                    for prob_type, probs in prob_types.items():
                        run_methods_for_params(
                            dataset_name='cifar', K=K, methods=methods, training_args=training_args, sampling_probs=probs,
                            n_cal=n_cal, n_test=size, n_vs=n_vs, vector_scaling=vector_scaling,
                            informative_type=informative_type, alpha=alpha,
                            output_folder=output_folder.format(K=K, informative_type=informative_type, prob_type=prob_type),
                            iterations=iterations, max_workers=max_workers, trained_gaussian_model_args_cache={})


if __name__ == '__main__':
    probs_K3 = [0.33, 0.33, 0.34]
    probs_K3_uneven = [0.2, 0.6, 0.2]
    probs_K4 = [0.25, 0.25, 0.25, 0.25]
    probs_K4_uneven = [0.1, 0.7, 0.1, 0.1]
    probs_K10 = [0.1] * 10
    probs_K10_uneven = [0.15, 0.45] + [0.05] * 8
    distances_full = [distance / 10 for distance in range(1, 51)]
    distances_partial = [distance / 10 for distance in range(1, 51, 5)]
    max_workers = 14
    n_train = 10000

    folder_id = 'July_2026'
    run_gaussian_K3 = True
    run_gaussian_K4 = True
    run_gaussian_K10 = True
    run_cifar_BCD = True
    run_cifar_all10 = True
    overall_start_time = time.time()

    trained_gaussian_model_args_cache = {}
    if run_gaussian_K3 or run_gaussian_K4 or run_gaussian_K10:
        gaussian_probs = []
        if run_gaussian_K3:
            gaussian_probs += [probs_K3, probs_K3_uneven]
        if run_gaussian_K4:
            gaussian_probs += [probs_K4, probs_K4_uneven]
        if run_gaussian_K10:
            gaussian_probs += [probs_K10, probs_K10_uneven]
        trained_gaussian_model_args_cache = train_gaussian_cache(distances_full, gaussian_probs, n_train=n_train)

    if run_gaussian_K3:
        run_gaussian_experiment(
            K=3, informative_types=('non_trivial', 'exclude_1'), sizes=(500,), alphas=(0.05,),
            prob_types={
                'trained_even_test_even': (probs_K3, probs_K3),
                'trained_uneven_test_even': (probs_K3_uneven, probs_K3),
                'trained_even_test_uneven': (probs_K3, probs_K3_uneven),
                'trained_uneven_test_uneven': (probs_K3_uneven, probs_K3_uneven),
            },
            methods=(OGInfoSP, OGInfoSPCal, InfoSP_algo, InfoSCOP_algo, classic_conformal_algo),
            iterations=10000, vector_scalings=('none', 'only_bias', 'none_reduced_cal'), n_vector_scaling=100,
            distances=distances_full, n_train=n_train, max_workers=max_workers,
            output_folder=f'{folder_id}/gaussian_K{{K}}_{{informative_type}}_{{prob_type}}',
            trained_gaussian_model_args_cache=trained_gaussian_model_args_cache)

    if run_gaussian_K4:
        run_gaussian_experiment(
            K=4, informative_types=('non_trivial', 'exclude_1'), sizes=(500,), alphas=(0.05,),
            prob_types={
                'trained_even_test_even': (probs_K4, probs_K4),
                'trained_uneven_test_even': (probs_K4_uneven, probs_K4),
                'trained_even_test_uneven': (probs_K4, probs_K4_uneven),
                'trained_uneven_test_uneven': (probs_K4_uneven, probs_K4_uneven),
            },
            methods=(OGInfoSP, OGInfoSPCal, InfoSP_algo, InfoSCOP_algo, classic_conformal_algo),
            iterations=10000, vector_scalings=('none', 'only_bias', 'none_reduced_cal'), n_vector_scaling=100,
            distances=distances_full, n_train=n_train, max_workers=max_workers,
            output_folder=f'{folder_id}/gaussian_K{{K}}_{{informative_type}}_{{prob_type}}',
            trained_gaussian_model_args_cache=trained_gaussian_model_args_cache)

    if run_gaussian_K10:
        run_gaussian_experiment(
            K=10, informative_types=('non_trivial', 'exclude_1', 'up_to_3'), sizes=(500,), alphas=(0.05,),
            prob_types={
                'trained_even_test_even': (probs_K10, probs_K10),
                'trained_uneven_test_even': (probs_K10_uneven, probs_K10),
                'trained_even_test_uneven': (probs_K10, probs_K10_uneven),
                'trained_uneven_test_uneven': (probs_K10_uneven, probs_K10_uneven),
            },
            methods=(OGInfoSP, OGInfoSPCal, InfoSP_algo, InfoSCOP_algo, classic_conformal_algo, adaptive_classic_conformal_algo),
            iterations=10000, vector_scalings=('none', 'only_bias', 'none_reduced_cal'), n_vector_scaling=100,
            distances=distances_full, n_train=n_train, max_workers=max_workers,
            output_folder=f'{folder_id}/gaussian_K{{K}}_{{informative_type}}_{{prob_type}}',
            trained_gaussian_model_args_cache=trained_gaussian_model_args_cache)

    if run_cifar_BCD:
        run_cifar_experiment(
            K=3, model_dir='./cifar/models/bcd', informative_types=('non_trivial', 'exclude_1'),
            sizes=(500,), alphas=(0.1,),
            prob_types={'even': probs_K3, 'uneven': probs_K3_uneven},
            methods=(OGInfoSP, OGInfoSPCal, InfoSP_algo, InfoSCOP_algo, classic_conformal_algo, adaptive_classic_conformal_algo, top_class_algo),
            iterations=1000, vector_scalings=('none', 'none_reduced_cal', 'full'),
            n_calibration=1000, n_vector_scaling=500, max_workers=max_workers,
            output_folder=f'{folder_id}/CIFAR_BCD_{{informative_type}}_{{prob_type}}')

    if run_cifar_all10:
        run_cifar_experiment(
            K=10, model_dir='./cifar/models/all10', informative_types=('single_0_to_6', 'exclude_1'),
            sizes=(500,), alphas=(0.1,),
            prob_types={'even': probs_K10, 'uneven': probs_K10_uneven},
            methods=(OGInfoSP, OGInfoSPCal, InfoSP_algo, InfoSCOP_algo, classic_conformal_algo, adaptive_classic_conformal_algo, top_class_algo),
            iterations=1000, vector_scalings=('none', 'none_reduced_cal', 'full'),
            n_calibration=1000, n_vector_scaling=500, max_workers=max_workers,
            output_folder=f'{folder_id}/CIFAR_all10_{{informative_type}}_{{prob_type}}')
        
    overall_end_time = time.time()
    print('Overall execution time:', overall_end_time - overall_start_time)
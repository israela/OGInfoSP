import os
import numpy as np
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from simulation.utils import plot_two_seq, plot_one_seq, plot_boxplot, write_results_generic, create_filepath, classifier_key
from simulation.interfaces import ModelTrainingArgs, DefaultGaussiansModelArgs, DataSamplingArgs, ProcedureArgs, ExecutionArgs, TrainedGaussiansModelArgs, VectorScalingArgs

multiprocessing.set_start_method("spawn", force=True)


def run_algo(method_and_args):
    method, args = method_and_args
    return method(*args)


def run_and_plot_per_distance(execution_args: ExecutionArgs, training_args: ModelTrainingArgs, sampling_args: DataSamplingArgs, vector_scaling_args: VectorScalingArgs, procedure_args: ProcedureArgs, trained_gaussian_model_args_cache):
    assert(training_args.kind == 'default_gaussians_series')
    start_time = time.time()
    FCR_per_distance = []
    mFCR_per_distance = []
    mean_power_per_distance = []
    FCR_standard_error_per_distance = []
    mean_selected_per_distance = []
    mean_correct_selected_per_distance = []
    
    for distance in training_args.distances:
        if trained_gaussian_model_args_cache is None:
            trained_args = DefaultGaussiansModelArgs('default_gaussians', distance, training_args.probs, training_args.n_train)
        else:
            trained_args = trained_gaussian_model_args_cache[classifier_key(distance, training_args.probs)]
        method_and_args = (execution_args.method, (trained_args, sampling_args, vector_scaling_args, procedure_args))
        with ProcessPoolExecutor(max_workers = execution_args.max_workers) as executor:
            all_FCP_parts = list(executor.map(run_algo, [method_and_args] * execution_args.iterations))

        V, R, power = np.array(all_FCP_parts).T
        all_FCP = V / np.maximum(R, 1)
        FCR = np.mean(V / np.maximum(R, 1))
        mFCR = np.mean(V) / np.maximum(np.mean(R), 1)
        FCR_standard_error = np.std(all_FCP, ddof=1) / np.sqrt(len(all_FCP))
        FCR_per_distance.append(FCR)
        mFCR_per_distance.append(mFCR)
        mean_power_per_distance.append(np.mean(power))
        FCR_standard_error_per_distance.append(FCR_standard_error)
        mean_selected_per_distance.append(np.mean(R))
        mean_correct_selected_per_distance.append(np.mean(R - V))

    end_time = time.time()
    print('done:', 'execution time:', end_time - start_time)
    file_path = create_filepath(execution_args, training_args, sampling_args, vector_scaling_args, procedure_args)
    os.makedirs(f'output/{execution_args.output_folder}', exist_ok = True)
    plot_two_seq(training_args.distances, 'Distance', FCR_per_distance, 'FCR', mFCR_per_distance, 'mFCR', file_path + '_FCR', procedure_args.alpha)
    plot_one_seq(training_args.distances, 'Distance', mean_power_per_distance, 'Power', file_path + '_Power')
    plot_one_seq(training_args.distances, 'Distance', mean_selected_per_distance, 'Selected', file_path + '_Selected')
    plot_one_seq(training_args.distances, 'Distance', mean_correct_selected_per_distance, 'Correct Selected', file_path + '_CorrectSelected')
    write_results_generic(file_path, execution_time = end_time - start_time, execution_args = execution_args, training_args = training_args, sampling_args = sampling_args, vector_scaling_args = vector_scaling_args, procedure_args = procedure_args, FCR_per_distance = FCR_per_distance, mFCR_per_distance = mFCR_per_distance, mean_power_per_distance = mean_power_per_distance, FCR_standard_error_per_distance = FCR_standard_error_per_distance, mean_selected_per_distance = mean_selected_per_distance, mean_correct_selected_per_distance = mean_correct_selected_per_distance)


def run_and_plot_distribution(execution_args: ExecutionArgs, training_args: ModelTrainingArgs, sampling_args: DataSamplingArgs, vector_scaling_args: VectorScalingArgs, procedure_args: ProcedureArgs):
    start_time = time.time()
    
    method_and_args = (execution_args.method, (training_args, sampling_args, vector_scaling_args, procedure_args))
    with ProcessPoolExecutor(max_workers=execution_args.max_workers) as executor:
        all_FCP_parts = list(executor.map(run_algo, [method_and_args] * execution_args.iterations))
    V, R, power = np.array(all_FCP_parts).T
    all_FCP = V / np.maximum(R, 1)
    all_power = power
    all_selected = R
    all_correct_selected = R - V
    FCR = np.mean(all_FCP)
    mean_power = np.mean(all_power)

    end_time = time.time()
    print('done:', 'execution time:', end_time - start_time, 'FCR:', FCR, 'power:', mean_power)
    file_path = create_filepath(execution_args, training_args, sampling_args, vector_scaling_args, procedure_args)
    os.makedirs(f'output/{execution_args.output_folder}', exist_ok = True)
    plot_boxplot([[all_FCP, 'FCR']], None, None, file_path + '_FCR', procedure_args.alpha)
    plot_boxplot([[all_power, 'Power']], None, None, file_path + '_Power')
    plot_boxplot([[all_selected, 'Selected']], None, None, file_path + '_Selected')
    plot_boxplot([[all_correct_selected, 'Correct Selected']], None, None, file_path + '_CorrectSelected')
    write_results_generic(file_path, execution_time = end_time - start_time, execution_args = execution_args, training_args = training_args, sampling_args = sampling_args, vector_scaling_args = vector_scaling_args, procedure_args = procedure_args, FCR = FCR, mean_power = mean_power, all_FCP = all_FCP, all_power = all_power, all_selected = all_selected, all_correct_selected = all_correct_selected)
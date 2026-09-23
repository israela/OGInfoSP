import numpy as np
from simulation.sample_and_train import sample_and_train
from simulation.interfaces import CalAndTestSamples, DataSamplingArgs, ModelTrainingArgs, ProcedureArgs, VectorScalingArgs


def classic_conformal_algo(training_args: ModelTrainingArgs, sampling_args: DataSamplingArgs, vector_scaling_args: VectorScalingArgs, procedure_args: ProcedureArgs):
    samples: CalAndTestSamples = sample_and_train(training_args, sampling_args, vector_scaling_args)

    S_cal = 1 - samples.cal_class_scores
    S_test = 1 - samples.test_class_scores
    S_cal_y = S_cal[np.arange(len(S_cal)), samples.cal_data_classes]
    q_1_alpha_cal_rank = int(np.ceil((sampling_args.n_cal + 1) * (1 - procedure_args.alpha)))
    q_1_alpha_cal = np.partition(S_cal_y, q_1_alpha_cal_rank - 1)[q_1_alpha_cal_rank - 1]
    prediction_sets = S_test <= q_1_alpha_cal
    empty_idx = np.where(prediction_sets.sum(axis=1) == 0)[0]
    prediction_sets[empty_idx, S_test[empty_idx].argmin(axis=1)] = True

    if procedure_args.informative_type == 'non_trivial':
        prediction_sets[np.all(prediction_sets, axis=1)] = False
    elif procedure_args.informative_type == 'exclude_1':
        prediction_sets[prediction_sets[:, 1]] = False
    elif procedure_args.informative_type == 'up_to_3':
        prediction_sets[prediction_sets.sum(axis=1) > 3] = False
    elif procedure_args.informative_type == 'single_0_to_6':
        prediction_sets[(prediction_sets.sum(axis=1) != 1) | (prediction_sets[:, 7:].any(axis=1))] = False
    else:
        raise ValueError('unsupported informative_type')
    
    selected_test_mask = np.any(prediction_sets, axis=1)
    correct_test_mask = prediction_sets[np.arange(sampling_args.n_test), samples.test_data_classes]

    selected_test = np.sum(selected_test_mask)
    incorrect_selected_test = np.sum(selected_test_mask & ~correct_test_mask)
    power = np.sum(1 / np.sum(prediction_sets, axis=1)[correct_test_mask]) / sampling_args.n_test
    return incorrect_selected_test, selected_test, power


classic_conformal_algo.method_name = 'ClassicConformal'
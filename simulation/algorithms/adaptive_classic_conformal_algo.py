import numpy as np
from simulation.sample_and_train import sample_and_train
from simulation.interfaces import CalAndTestSamples, DataSamplingArgs, ModelTrainingArgs, ProcedureArgs, VectorScalingArgs


def adaptive_classic_conformal_algo(training_args: ModelTrainingArgs, sampling_args: DataSamplingArgs, vector_scaling_args: VectorScalingArgs, procedure_args: ProcedureArgs):
    samples: CalAndTestSamples = sample_and_train(training_args, sampling_args, vector_scaling_args)

    cal_probs = samples.cal_class_scores
    test_probs = samples.test_class_scores
    n_cal = sampling_args.n_cal
    n_test = sampling_args.n_test
    K = cal_probs.shape[1]

    # Calibration: compute E_i = cumsum[rank(y_i)] - U_i * π̂_(rank(y_i))
    cal_sorted_idx = np.argsort(-cal_probs, axis=1)
    cal_sorted_probs = np.take_along_axis(cal_probs, cal_sorted_idx, axis=1)
    cal_cumsum = np.round(np.cumsum(cal_sorted_probs, axis=1), 9)
    cal_ranks = np.argsort(cal_sorted_idx, axis=1)
    true_ranks = cal_ranks[np.arange(n_cal), samples.cal_data_classes]

    epsilon_cal = np.random.uniform(0, 1, size=n_cal)
    E_cal = cal_cumsum[np.arange(n_cal), true_ranks] - epsilon_cal * cal_sorted_probs[np.arange(n_cal), true_ranks]

    # Threshold
    quantile_rank = int(np.ceil((n_cal + 1) * (1 - procedure_args.alpha)))
    tau_hat = np.partition(E_cal, quantile_rank - 1)[quantile_rank - 1]

    # Test prediction sets: S(x, U; π̂, τ̂)
    test_sorted_idx = np.argsort(-test_probs, axis=1)
    test_sorted_probs = np.take_along_axis(test_probs, test_sorted_idx, axis=1)
    test_cumsum = np.round(np.cumsum(test_sorted_probs, axis=1), 9)

    above_threshold = test_cumsum >= tau_hat
    L = np.where(above_threshold.any(axis=1), np.argmax(above_threshold, axis=1), K - 1)

    # Randomize boundary class
    epsilon_test = np.random.uniform(0, 1, size=n_test)
    excess = test_cumsum[np.arange(n_test), L] - tau_hat
    p_remove = excess / test_sorted_probs[np.arange(n_test), L]
    L[epsilon_test <= p_remove] -= 1
    L = np.maximum(L, 0)  # prevent empty prediction sets

    # Build prediction sets
    mask_sorted = np.arange(K)[None, :] <= L[:, None]
    prediction_sets = np.zeros((n_test, K), dtype=bool)
    np.put_along_axis(prediction_sets, test_sorted_idx, mask_sorted, axis=1)

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


adaptive_classic_conformal_algo.method_name = 'AdaptiveClassicConformal'
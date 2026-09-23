import numpy as np
from simulation.sample_and_train import sample_and_train
from simulation.interfaces import CalAndTestSamples, DataSamplingArgs, ModelTrainingArgs, ProcedureArgs, VectorScalingArgs


def compute_p_values(cal_data_classes, cal_class_scores, test_class_scores):
    S_cal = 1 - cal_class_scores
    S_test = 1 - test_class_scores
    S_cal_y = S_cal[np.arange(len(S_cal)), cal_data_classes]
    S_cal_y_expanded = S_cal_y[:, np.newaxis, np.newaxis]
    S_test_expanded = S_test[np.newaxis, :, :]
    comparisons = S_cal_y_expanded >= S_test_expanded
    counts = comparisons.sum(axis=0)
    p_values = (counts + 1) / (len(cal_class_scores) + 1)
    return p_values


def compute_q_values(p_values, informative_type):
    if informative_type == 'non_trivial':
        q_values = p_values.min(axis=1)
    elif informative_type == 'exclude_1':
        q_values = p_values[:, 1]
    elif informative_type == 'up_to_3':
        q_values = np.sort(p_values, axis=1)[:, -4] # 4'th largest = (K - 3)'th smallest
    elif informative_type == 'single_0_to_6':
        if p_values.shape[1] == 7:
            # the 2nd largest (for singleton)
            q_values = np.sort(p_values, axis=1)[:, -2]
        else:
            # the maximum between the 2nd largest (for singleton) and the largest among the excluded classes (>= 7)
            q_values = np.maximum(np.sort(p_values, axis=1)[:, -2], p_values[:, 7:].max(axis=1))
    else:
        raise ValueError('unsupported informative_type')
    return q_values


def BH(q_values, alpha):
    sorted_indices = np.argsort(q_values)
    sorted_q_values = q_values[sorted_indices]
    m = len(q_values)
    passed = sorted_q_values <= (np.arange(1, m + 1) / m) * alpha

    if not np.any(passed):
        return np.array([], dtype=int)
    
    max_i = np.max(np.where(passed)[0])
    return sorted_indices[:max_i + 1]


def InfoSP_algo(training_args: ModelTrainingArgs, sampling_args: DataSamplingArgs, vector_scaling_args: VectorScalingArgs, procedure_args: ProcedureArgs):
    samples: CalAndTestSamples = sample_and_train(training_args, sampling_args, vector_scaling_args)
    p_values = compute_p_values(samples.cal_data_classes, samples.cal_class_scores, samples.test_class_scores)
    q_values = compute_q_values(p_values, procedure_args.informative_type)
    selected_indices = BH(q_values, procedure_args.alpha)
    include_in_prediction_set = p_values[selected_indices] > procedure_args.alpha * len(selected_indices) / len(p_values)
    empty = include_in_prediction_set.sum(axis=1) == 0
    include_in_prediction_set[empty, p_values[selected_indices][empty].argmax(axis=1)] = True
    #prediction_sets = [np.where(row)[0] for row in include_in_prediction_set]

    selected_test = len(selected_indices)
    test_correctness = include_in_prediction_set[np.arange(selected_test), samples.test_data_classes[selected_indices]]
    incorrect_selected_test = np.sum(~test_correctness)
    correct_prediction_set_sizes = include_in_prediction_set[test_correctness].sum(axis=1)
    power = np.sum(1 / correct_prediction_set_sizes) / sampling_args.n_test
    return incorrect_selected_test, selected_test, power


InfoSP_algo.method_name = 'InfoSP'
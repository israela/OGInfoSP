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


def compute_initial_exclude_1_p_values(cal_class_1_scores, test_class_1_scores):
    S_cal_1 = 1 - cal_class_1_scores
    S_test_1 = 1 - test_class_1_scores
    comparisons = S_cal_1[:, np.newaxis] >= S_test_1
    p_values = (1 + np.sum(comparisons, axis=0)) / (1 + len(cal_class_1_scores))
    return p_values


def compute_initial_single_0_to_6_p_values(cal_true_class_7_9_scores, test_class_scores):
    S_cal_7_9 = 1 - cal_true_class_7_9_scores
    S_test_7_9 = 1 - test_class_scores[:, 7:]
    comparisons = S_test_7_9[:, :, np.newaxis] < S_cal_7_9
    p_values = (1 + np.sum(comparisons, axis=2)) / (1 + len(cal_true_class_7_9_scores))
    return np.max(p_values, axis=1)


def compute_initial_non_trivial_p_values(cal_true_class_scores, test_class_scores):
    S_cal = 1 - cal_true_class_scores
    S_test = 1 - test_class_scores
    comparisons = S_test[:, :, np.newaxis] < S_cal
    p_values = (1 + np.sum(comparisons, axis=2)) / (1 + len(cal_true_class_scores))
    min_p_values = np.min(p_values, axis=1) # for size up to j, K-j smallest
    return min_p_values


def compute_initial_up_to_size_p_values(cal_true_class_scores, test_class_scores, up_to_size):
    S_cal = 1 - cal_true_class_scores
    S_test = 1 - test_class_scores
    comparisons = S_test[:, :, np.newaxis] < S_cal
    p_values = (1 + np.sum(comparisons, axis=2)) / (1 + len(cal_true_class_scores))
    return np.sort(p_values, axis=1)[:, -(up_to_size + 1)] # for size up to j, 4'th largest = (K - 3)'th smallest


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


def sample_and_train_with_split(training_args: ModelTrainingArgs, sampling_args: DataSamplingArgs, n_cal_split, vector_scaling_args: VectorScalingArgs):
    samples: CalAndTestSamples = sample_and_train(training_args, sampling_args, vector_scaling_args)
    cal1_classes = samples.cal_data_classes[:n_cal_split]
    cal1_scores = samples.cal_class_scores[:n_cal_split]
    cal2_classes = samples.cal_data_classes[n_cal_split:]
    cal2_scores = samples.cal_class_scores[n_cal_split:]
    return cal1_classes, cal1_scores, cal2_classes, cal2_scores, samples.test_data_classes, samples.test_class_scores


def InfoSCOP_algo(training_args: ModelTrainingArgs, sampling_args: DataSamplingArgs, vector_scaling_args: VectorScalingArgs, procedure_args: ProcedureArgs):
    alpha = procedure_args.alpha
    n_cal_split = sampling_args.n_cal // 2
    cal1_classes, cal1_scores, cal2_classes, cal2_scores, test_data_classes, test_class_scores = sample_and_train_with_split(training_args, sampling_args, n_cal_split, vector_scaling_args)
    cal2_and_test_scores = np.concatenate([cal2_scores, test_class_scores])

    if procedure_args.informative_type == 'exclude_1':
        cal1_true_class_1_scores = cal1_scores[cal1_classes == 1][:, 1]
        phase_1_p_values = compute_initial_exclude_1_p_values(cal1_true_class_1_scores, cal2_and_test_scores[:, 1])
        phase_1_selected = BH(phase_1_p_values, alpha / ((len(cal1_true_class_1_scores) + 1) / (n_cal_split + 1)))
    elif procedure_args.informative_type == 'non_trivial':
        cal1_true_class_scores = cal1_scores[np.arange(n_cal_split), cal1_classes]
        phase_1_p_values = compute_initial_non_trivial_p_values(cal1_true_class_scores, cal2_and_test_scores)
        phase_1_selected = BH(phase_1_p_values, alpha)
    elif procedure_args.informative_type == 'up_to_3':
        cal1_true_class_scores = cal1_scores[np.arange(n_cal_split), cal1_classes]
        phase_1_p_values = compute_initial_up_to_size_p_values(cal1_true_class_scores, cal2_and_test_scores, 3)
        phase_1_selected = BH(phase_1_p_values, alpha)
    elif procedure_args.informative_type == 'single_0_to_6':
        if cal1_scores.shape[1] == 7:
            phase_1_selected = np.arange(len(cal2_and_test_scores))
        else:
            excluded_mask = cal1_classes >= 7
            cal1_excluded_true_class_scores = cal1_scores[np.arange(n_cal_split)[excluded_mask], cal1_classes[excluded_mask]]
            phase_1_p_values = compute_initial_single_0_to_6_p_values(cal1_excluded_true_class_scores, cal2_and_test_scores)
            phase_1_selected = BH(phase_1_p_values, alpha / ((len(cal1_excluded_true_class_scores) + 1) / (n_cal_split + 1)))
    else:
        raise ValueError('unsupported informative_type')

    n_cal2 = len(cal2_scores)
    selected_cal2_indices = phase_1_selected[phase_1_selected < n_cal2]
    selected_test_indices = phase_1_selected[phase_1_selected >= n_cal2] - n_cal2
    cal2_selected_classes = cal2_classes[selected_cal2_indices]
    cal2_selected_scores = cal2_scores[selected_cal2_indices]
    test_selected_classes = test_data_classes[selected_test_indices]
    test_selected_scores = test_class_scores[selected_test_indices]

    if len(test_selected_classes) == 0 or len(cal2_selected_classes) == 0:
        return 0, 0, 0

    p_values = compute_p_values(cal2_selected_classes, cal2_selected_scores, test_selected_scores)
    q_values = compute_q_values(p_values, procedure_args.informative_type)
    selected_indices = BH(q_values, alpha)

    if len(selected_indices) == 0:
        return 0, 0, 0

    include_in_prediction_set = p_values[selected_indices] > alpha * len(selected_indices) / len(p_values)
    empty = include_in_prediction_set.sum(axis=1) == 0
    include_in_prediction_set[empty, p_values[selected_indices][empty].argmax(axis=1)] = True
    #prediction_sets = [np.where(row)[0] for row in include_in_prediction_set]

    selected_test = len(selected_indices)
    test_correctness = include_in_prediction_set[np.arange(selected_test), test_selected_classes[selected_indices]]
    incorrect_selected_test = np.sum(~test_correctness)
    correct_prediction_set_sizes = include_in_prediction_set[test_correctness].sum(axis=1)
    power = np.sum(1 / correct_prediction_set_sizes) / sampling_args.n_test
    return incorrect_selected_test, selected_test, power


InfoSCOP_algo.method_name = 'InfoSCOP'
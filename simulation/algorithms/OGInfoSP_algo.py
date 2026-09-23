import numpy as np
import math
from dataclasses import replace
from simulation.sample_and_train import sample_and_train
from simulation.interfaces import ModelTrainingArgs, DataSamplingArgs, ProcedureArgs, CalAndTestSamples, VectorScalingArgs


def compute_P_hat_matrix(class_scores, informative_sets):
    """Returns matrix of shape (n_samples, n_sets) with P_hat values."""
    return np.column_stack([np.sum(np.sort(class_scores[:, C], axis=1)[:, ::-1], axis=1) for C in informative_sets])


def compute_class_in_set_matrix(data_classes, informative_sets):
    """Returns boolean array of shape (n_samples, n_sets) where [i,j] = (data_classes[i] in informative_sets[j])."""
    n = len(data_classes)
    n_sets = len(informative_sets)
    class_in_set = np.zeros((n, n_sets), dtype=bool)
    for j, C in enumerate(informative_sets):
        class_in_set[:, j] = np.isin(data_classes, C)
    return class_in_set


def compute_FCP_parts(class_in_set, P_hat_matrix, set_weights, mu, alpha, compute_power = False, include_threshold = False):
    if math.isinf(mu):
        return 0, 0, 0

    P_hat_scaled_matrix = (set_weights + mu) * P_hat_matrix
    max_val = P_hat_scaled_matrix.max(axis=1, keepdims=True)
    C_i_idx = np.argmin(np.where(P_hat_scaled_matrix == max_val, set_weights, np.inf), axis=1) # break ties in favor of smaller set_weight
    C_i_val = P_hat_scaled_matrix[np.arange(P_hat_matrix.shape[0]), C_i_idx]

    if include_threshold:
        D_i = (C_i_val - mu * (1 - alpha) >= 0).astype(int)
    else:
        D_i = (C_i_val - mu * (1 - alpha) > 0).astype(int)

    selected = np.sum(D_i)
    incorrect_selected = None
    weighted_correct_selected = None
    if class_in_set is not None:
        correct_if_selected = class_in_set[np.arange(class_in_set.shape[0]), C_i_idx]
        incorrect_selected = np.sum(~correct_if_selected & (D_i == 1))
        if compute_power:
            weighted_correct_selected = np.sum(set_weights[C_i_idx[correct_if_selected & (D_i == 1)]])

    return incorrect_selected, selected, weighted_correct_selected


def estimate_FCP(cal_class_in_set, cal_P_hat_matrix, test_P_hat_matrix, set_weights, mu, alpha, use_cal_denominator = False, include_threshold = False, increase_denominator = False):
    if use_cal_denominator:
        incorrect_selected_cal, selected_cal, _ = compute_FCP_parts(cal_class_in_set, cal_P_hat_matrix, set_weights, mu, alpha, include_threshold = include_threshold)
        if increase_denominator:
            return (1 + incorrect_selected_cal) / (1 + selected_cal)
        else:
            return ((1 + incorrect_selected_cal) / (1 + cal_P_hat_matrix.shape[0])) / (np.maximum(selected_cal, 1) / cal_P_hat_matrix.shape[0])
    else:
        incorrect_selected_cal, _, _ = compute_FCP_parts(cal_class_in_set, cal_P_hat_matrix, set_weights, mu, alpha, include_threshold = include_threshold)
        _, selected_test, _ = compute_FCP_parts(None, test_P_hat_matrix, set_weights, mu, alpha, include_threshold = include_threshold)
        return ((1 + incorrect_selected_cal) / (1 + cal_P_hat_matrix.shape[0])) / (np.maximum(selected_test, 1) / test_P_hat_matrix.shape[0])


def get_mu_candidates(cal_P_hat_matrix, set_weights, alpha):
    n_sets = cal_P_hat_matrix.shape[1]
    all_mu_candidates = []
    for j in range(n_sets):
        P_j = cal_P_hat_matrix[:, j]
        w_j = set_weights[j]
        mu_candidates = w_j * P_j / (1 - alpha - P_j)  # values of mu where D can change
        mu_candidates = mu_candidates[mu_candidates >= 0]
        all_mu_candidates.append(mu_candidates * (1 + 10 * np.finfo(float).eps))  # adding 10 eps for float precision issues
        for j2 in range(j):
            P_j2 = cal_P_hat_matrix[:, j2]
            w_j2 = set_weights[j2]
            denom = P_j - P_j2
            mu_candidates = np.divide(w_j2 * P_j2 - w_j * P_j, denom, out=np.full_like(denom, -1.0), where=(denom != 0))  # values of mu where C can change
            mu_candidates = mu_candidates[mu_candidates >= 0]
            all_mu_candidates.append(mu_candidates * (1 + 10 * np.finfo(float).eps))  # adding 10 eps for float precision issues
    unique_candidates = np.unique(np.concatenate(all_mu_candidates))  # already sorted
    if len(unique_candidates) == 0:
        return unique_candidates
    # remove near duplicates, keep largest per 0.01-wide bin
    last_in_bucket = np.append(np.diff((unique_candidates / 1e-2).astype(int)) > 0, True) # append True to keep last candidate
    filtered_candidates = unique_candidates[last_in_bucket]
    return filtered_candidates


def mu_alpha(set_weights, cal_P_hat_matrix, cal_class_in_set, test_P_hat_matrix, alpha, use_cal_denominator = False, include_threshold = False, increase_denominator = False):
    mu_candidates = get_mu_candidates(cal_P_hat_matrix, set_weights, alpha)
    for mu in mu_candidates:
        FCP_hat = estimate_FCP(cal_class_in_set, cal_P_hat_matrix, test_P_hat_matrix, set_weights, mu, alpha, use_cal_denominator, include_threshold, increase_denominator)
        if FCP_hat <= alpha:
            return mu, FCP_hat
    return float('inf'), 0


def compute_efficient_matrices(class_scores, data_classes, allowed_classes, max_set_size):
    scores = class_scores[:, allowed_classes]

    # P_hat_matrix: column k-1 = sum of top-k scores among allowed classes
    sorted_scores = np.sort(scores, axis=1)[:, ::-1]
    P_hat_matrix = np.cumsum(sorted_scores, axis=1)[:, :max_set_size]

    # class_in_set: true class is in best-set-of-size-k iff it's allowed and ranked < k
    is_allowed = np.isin(data_classes, allowed_classes)
    # Map data_classes to column index within allowed_classes
    class_to_col = np.full(class_scores.shape[1], -1, dtype=int)
    class_to_col[allowed_classes] = np.arange(len(allowed_classes))
    mapped_classes = class_to_col[data_classes]  # -1 for disallowed classes

    ranks = np.argsort(np.argsort(-scores, axis=1), axis=1)
    # For disallowed classes, use 0 as placeholder (will be zeroed out)
    safe_mapped = np.where(is_allowed, mapped_classes, 0)
    true_class_ranks = ranks[np.arange(len(data_classes)), safe_mapped]
    class_in_set = true_class_ranks[:, np.newaxis] < np.arange(1, max_set_size + 1)
    class_in_set[~is_allowed] = False

    return P_hat_matrix, class_in_set


def OGInfoSP_algo(training_args: ModelTrainingArgs, sampling_args: DataSamplingArgs, vector_scaling_args: VectorScalingArgs, procedure_args: ProcedureArgs):
    samples: CalAndTestSamples = sample_and_train(training_args, sampling_args, vector_scaling_args)

    if procedure_args.informative_type in ('non_trivial', 'exclude_1', 'up_to_3', 'single_0_to_6'):
        # Efficient path: one column per set cardinality k=1..max_set_size.
        # P_hat_matrix[i, k-1] = sum of top-k scores for sample i (= best P_hat over all sets of size k).
        # class_in_set[i, k-1] = whether sample i's true class is among its top-k scored classes.
        K = samples.cal_class_scores.shape[1]
        if procedure_args.informative_type == 'non_trivial':
            allowed_classes, max_set_size = np.arange(K), K - 1
        elif procedure_args.informative_type == 'exclude_1':
            allowed_classes, max_set_size = np.array([i for i in range(K) if i != 1]), K - 1
        elif procedure_args.informative_type == 'up_to_3':
            allowed_classes, max_set_size = np.arange(K), 3
        elif procedure_args.informative_type == 'single_0_to_6':
            allowed_classes, max_set_size = np.arange(7), 1
        set_weights = np.array([1.0 / k for k in range(1, max_set_size + 1)])
        cal_P_hat_matrix, cal_class_in_set = compute_efficient_matrices(samples.cal_class_scores, samples.cal_data_classes, allowed_classes, max_set_size)
        test_P_hat_matrix, test_class_in_set = compute_efficient_matrices(samples.test_class_scores, samples.test_data_classes, allowed_classes, max_set_size)
    else:
        # General path: one column per explicit informative set.
        # P_hat_matrix[i, j] = sum of scores for the classes in informative_sets[j] for sample i.
        # class_in_set[i, j] = whether sample i's true class is in informative_sets[j].
        set_weights = np.array([1.0 / len(C) for C in procedure_args.informative_sets])
        cal_P_hat_matrix = compute_P_hat_matrix(samples.cal_class_scores, procedure_args.informative_sets)
        cal_class_in_set = compute_class_in_set_matrix(samples.cal_data_classes, procedure_args.informative_sets)
        test_P_hat_matrix = compute_P_hat_matrix(samples.test_class_scores, procedure_args.informative_sets)
        test_class_in_set = compute_class_in_set_matrix(samples.test_data_classes, procedure_args.informative_sets)

    mu, FCP_hat = mu_alpha(set_weights, cal_P_hat_matrix, cal_class_in_set, test_P_hat_matrix, procedure_args.alpha, procedure_args.use_cal_denominator, procedure_args.include_threshold, procedure_args.increase_denominator)

    incorrect_selected_test, selected_test, weighted_correct_selected = compute_FCP_parts(test_class_in_set, test_P_hat_matrix, set_weights, mu, procedure_args.alpha, compute_power = True, include_threshold = procedure_args.include_threshold)
    power = weighted_correct_selected / sampling_args.n_test
    return incorrect_selected_test, selected_test, power


OGInfoSP_algo.method_name = 'OGInfoSPCommon'


def OGInfoSP(training_args: ModelTrainingArgs, sampling_args: DataSamplingArgs, vector_scaling_args: VectorScalingArgs, procedure_args: ProcedureArgs):
    procedure_args = replace(
        procedure_args,
        use_cal_denominator=False,
        include_threshold=False,
        increase_denominator=False,
    )
    return OGInfoSP_algo(training_args, sampling_args, vector_scaling_args, procedure_args)


OGInfoSP.method_name = 'OGInfoSP'


def OGInfoSPCal(training_args: ModelTrainingArgs, sampling_args: DataSamplingArgs, vector_scaling_args: VectorScalingArgs, procedure_args: ProcedureArgs):
    procedure_args = replace(
        procedure_args,
        use_cal_denominator=True,
        include_threshold=False,
        increase_denominator=True,
    )
    return OGInfoSP_algo(training_args, sampling_args, vector_scaling_args, procedure_args)


OGInfoSPCal.method_name = 'OGInfoSPCal'
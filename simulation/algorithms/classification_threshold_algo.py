import numpy as np
from simulation.interfaces import CalAndTestSamples, DataSamplingArgs, ModelTrainingArgs, ProcedureArgs, VectorScalingArgs
from simulation.sample_and_train import sample_and_train


def estimate_FSR(threshold, cal_data_classes, cal_class_scores, denominator_class_scores):
    cal_top_scores = np.max(cal_class_scores, axis=1)
    cal_top_classes = np.argmax(cal_class_scores, axis=1)
    incorrect_rate = (np.sum((cal_top_scores > threshold) & (cal_top_classes != cal_data_classes)) + 1) / (len(cal_class_scores) + 1)

    denominator_top_scores = np.max(denominator_class_scores, axis=1)
    denominator_selected = denominator_top_scores > threshold
    selected_rate = np.maximum(np.sum(denominator_selected), 1) / len(denominator_class_scores)
    return incorrect_rate / selected_rate


def classification_threshold_algo(training_args: ModelTrainingArgs, sampling_args: DataSamplingArgs, vector_scaling_args: VectorScalingArgs, procedure_args: ProcedureArgs):
    samples: CalAndTestSamples = sample_and_train(training_args, sampling_args, vector_scaling_args)

    denominator_class_scores = samples.cal_class_scores if procedure_args.use_cal_denominator else samples.test_class_scores
    threshold_candidates = np.concatenate((samples.test_class_scores.ravel(), samples.cal_class_scores.ravel()))
    threshold_candidates = threshold_candidates[threshold_candidates >= 0.5]
    FSR_estimates = np.array([estimate_FSR(threshold, samples.cal_data_classes, samples.cal_class_scores, denominator_class_scores) for threshold in threshold_candidates])
    threshold = np.min(threshold_candidates[FSR_estimates <= procedure_args.alpha], initial=1.0)

    top_classes = np.argmax(samples.test_class_scores, axis=1)
    test_top_scores = np.max(samples.test_class_scores, axis=1)
    selected_mask = test_top_scores > threshold
    if procedure_args.informative_type == 'exclude_1':
        selected_mask &= top_classes != 1
    elif procedure_args.informative_type == 'single_0_to_6':
        selected_mask &= top_classes <= 6
    elif procedure_args.informative_type in ('non_trivial', 'up_to_3'):
        pass
    else:
        raise ValueError('unsupported informative_type')

    correct_mask = top_classes == samples.test_data_classes
    selected_test = np.sum(selected_mask)
    incorrect_selected_test = np.sum(selected_mask & ~correct_mask)
    power = np.sum(selected_mask & correct_mask) / sampling_args.n_test
    return incorrect_selected_test, selected_test, power


classification_threshold_algo.method_name = 'ClassificationThreshold'

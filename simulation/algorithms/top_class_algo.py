import numpy as np
from simulation.sample_and_train import sample_and_train
from simulation.interfaces import CalAndTestSamples, DataSamplingArgs, ModelTrainingArgs, ProcedureArgs, VectorScalingArgs


def top_class_algo(training_args: ModelTrainingArgs, sampling_args: DataSamplingArgs, vector_scaling_args: VectorScalingArgs, procedure_args: ProcedureArgs):
    samples: CalAndTestSamples = sample_and_train(training_args, sampling_args, vector_scaling_args)

    top_class = np.argmax(samples.test_class_scores, axis=1)
    correct_mask = top_class == samples.test_data_classes

    if procedure_args.informative_type == 'non_trivial':
        selected_mask = np.ones(sampling_args.n_test, dtype=bool)
    elif procedure_args.informative_type == 'exclude_1':
        selected_mask = top_class != 1
    elif procedure_args.informative_type == 'up_to_3':
        selected_mask = np.ones(sampling_args.n_test, dtype=bool)
    elif procedure_args.informative_type == 'single_0_to_6':
        selected_mask = top_class <= 6
    else:
        raise ValueError('unsupported informative_type')

    selected_test = np.sum(selected_mask)
    incorrect_selected_test = np.sum(selected_mask & ~correct_mask)
    power = np.sum(selected_mask & correct_mask) / sampling_args.n_test
    return incorrect_selected_test, selected_test, power


top_class_algo.method_name = 'TopClass'
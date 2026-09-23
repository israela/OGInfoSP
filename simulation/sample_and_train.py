import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import multivariate_normal
import warnings
from sklearn.exceptions import ConvergenceWarning
from simulation.interfaces import CalAndTestSamples, ModelTrainingArgs, DataSamplingArgs, TrainedGaussiansModelArgs, VectorScalingArgs
from simulation.vector_scaling_full import vector_scale_samples as vector_scale_samples_full
from simulation.vector_scaling_no_bias import vector_scale_samples as vector_scale_samples_no_bias
from simulation.vector_scaling_only_bias import vector_scale_samples as vector_scale_samples_only_bias

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def sample_gaussian_data(n, centers, probs):
    classes = np.random.choice(len(centers), size=n, p=probs)
    samples = np.random.randn(n, len(centers[0])) + np.array(centers)[classes]
    return classes, samples


def sample_and_train_gaussian(centers, train_probs, sample_probs, n_train, n_cal, n_test, predict_scores):
    classifier = train_gaussian(centers, train_probs, n_train) if predict_scores else None
    return sample_gaussian(classifier, centers, sample_probs, n_cal, n_test, predict_scores)


def train_gaussian(centers, probs, n_train):
    train_data_classes, train_data_samples = sample_gaussian_data(n_train, centers, probs)
    classifier = LogisticRegression()
    classifier.fit(train_data_samples.reshape(-1, len(centers[0])), train_data_classes)
    return classifier


def train_default_gaussian(distance, probs, n_train):
    centers = default_centers(distance, len(probs))
    classifier = train_gaussian(centers, probs, n_train)
    return TrainedGaussiansModelArgs(kind='trained_gaussians', classifier=classifier, centers=centers, probs=probs)


def sample_gaussian(classifier, centers, probs, n_cal, n_test, predict_scores):
    cal_data_classes, cal_data_samples = sample_gaussian_data(n_cal, centers, probs)
    test_data_classes, test_data_samples = sample_gaussian_data(n_test, centers, probs)

    if predict_scores:
        cal_class_scores = np.zeros((len(cal_data_samples), len(centers)))
        cal_class_scores[:, classifier.classes_] = classifier.predict_proba(cal_data_samples)
        test_class_scores = np.zeros((len(test_data_samples), len(centers)))
        test_class_scores[:, classifier.classes_] = classifier.predict_proba(test_data_samples)
    else:
        def true_probabilities(samples):
            likelihoods = np.empty((samples.shape[0], len(centers)))

            for class_index, center in enumerate(centers):
                distribution = multivariate_normal(mean=center)
                likelihoods[:, class_index] = distribution.pdf(samples)

            weighted_likelihoods = likelihoods * np.asarray(probs)
            return weighted_likelihoods / weighted_likelihoods.sum(axis=1, keepdims=True)

        cal_class_scores = true_probabilities(cal_data_samples)
        test_class_scores = true_probabilities(test_data_samples)
      
    return CalAndTestSamples(cal_data_classes, cal_data_samples, cal_class_scores, test_data_classes, test_data_samples, test_class_scores)


def default_centers(d, dim):
    if dim == 2:
        return np.array([[0, 0], [d, 0]])
    if dim == 3:
        return np.array([[0, 0], [d, 0], [d, d]])
    if dim == 4:
        return np.array([[0, 0], [d, 0], [0, d], [d, d]])
    if dim == 5:
        return np.array([[0, 0], [d, 0], [0, d], [d, d], [-d, 0]])
    if dim == 10:
        # 9 points on a circle of radius d around the origin and one point at the origin at index 1
        circle_points = d * np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 9, endpoint=False)])
        return np.vstack((circle_points[:1], [0, 0], circle_points[1:]))
    if dim == 11:
        return np.vstack(([0,0], d + 0.1 * np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 10, endpoint=False)])))
    raise NotImplementedError()


def sample_and_train(training_args: ModelTrainingArgs, sampling_args: DataSamplingArgs, vector_scaling_args: VectorScalingArgs):
    if training_args.kind == 'load':
        if training_args.dataset == 'CIFAR10':
            from cifar.CIFAR10_model import sample_cifar10  # loading here to avoid loading pytorch in other processes
            samples: CalAndTestSamples = sample_cifar10(sampling_args.probs, sampling_args.n_cal + vector_scaling_args.n_vs, sampling_args.n_test, training_args.model_dir)
        else:
            raise NotImplementedError('unknown dataset')
        
    elif training_args.kind == 'trained_gaussians':
        samples: CalAndTestSamples = sample_gaussian(training_args.classifier, training_args.centers, sampling_args.probs, sampling_args.n_cal + vector_scaling_args.n_vs, sampling_args.n_test, sampling_args.predict_scores)
    
    elif training_args.kind in ('default_gaussians', 'gaussians'):
        centers = default_centers(training_args.distance, len(training_args.probs)) if training_args.kind == 'default_gaussians' else training_args.centers
        samples: CalAndTestSamples = sample_and_train_gaussian(centers, training_args.probs, sampling_args.probs, training_args.n_train, sampling_args.n_cal + vector_scaling_args.n_vs, sampling_args.n_test, sampling_args.predict_scores)
    elif training_args.kind == 'default_gaussians_series':
        raise ValueError('For default_gaussians_series, handle the series first')
    else:
        raise ValueError('unknown training_args.kind')

    if vector_scaling_args.type == 'full':
        samples = vector_scale_samples_full(samples, vector_scaling_args.n_vs)
    elif vector_scaling_args.type == 'no_bias':
        samples = vector_scale_samples_no_bias(samples, vector_scaling_args.n_vs)
    elif vector_scaling_args.type == 'only_bias':
        samples = vector_scale_samples_only_bias(samples, vector_scaling_args.n_vs)
    else:
        assert(vector_scaling_args.n_vs == 0)
        pass

    return samples

import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize

from simulation.interfaces import CalAndTestSamples

def probs_to_logits(p):
    return np.log(np.clip(p, 1e-12, 1.0))


def nll_constrained(b_free, logits, labels):
    b = np.append(b_free, -np.sum(b_free))
    return nll(b, logits, labels)


def nll(b, logits, labels):
    scaled = logits + b
    log_probs = scaled - logsumexp(scaled, axis=1, keepdims=True)
    return -np.mean(log_probs[np.arange(len(labels)), labels])


def fit_vector_scaling(cal_probs, cal_labels):
    cal_logits = probs_to_logits(cal_probs)
    params0 = np.zeros(cal_logits.shape[1] - 1)
    res = minimize(nll_constrained, params0, args=(cal_logits, cal_labels), method='L-BFGS-B')
    return np.append(res.x, -np.sum(res.x))


def apply_vector_scaling(probs, b):
    logits = probs_to_logits(probs)
    scaled = logits + b
    scaled_probs = np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))
    return scaled_probs


def vector_scale_samples(samples, num_fit=100):
    b = fit_vector_scaling(samples.cal_class_scores[:num_fit], samples.cal_data_classes[:num_fit])
    cal_scaled = apply_vector_scaling(samples.cal_class_scores[num_fit:], b)
    test_scaled = apply_vector_scaling(samples.test_class_scores, b)
    return CalAndTestSamples(samples.cal_data_classes[num_fit:], samples.cal_data_samples[num_fit:], cal_scaled, samples.test_data_classes, samples.test_data_samples, test_scaled)
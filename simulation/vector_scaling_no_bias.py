import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize

from simulation.interfaces import CalAndTestSamples

def probs_to_logits(p):
    return np.log(np.clip(p, 1e-12, 1.0))


def nll(T_log, logits, labels):
    T = np.exp(T_log)
    T = np.clip(T, 1e-6, None)
    scaled = logits / T  # [N,K]
    log_probs = scaled - logsumexp(scaled, axis=1, keepdims=True)
    return -np.mean(log_probs[np.arange(len(labels)), labels])


def fit_vector_scaling(cal_probs, cal_labels):
    cal_logits = probs_to_logits(cal_probs)
    T_log0 = np.zeros(cal_logits.shape[1])
    res = minimize(nll, T_log0, args=(cal_logits, cal_labels), method='L-BFGS-B')
    return np.clip(np.exp(res.x), 1e-6, None)


def apply_vector_scaling(probs, T):
    logits = probs_to_logits(probs)
    scaled = logits / T
    scaled_probs = np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))
    return scaled_probs


def vector_scale_samples(samples, num_fit=100):
    T = fit_vector_scaling(samples.cal_class_scores[:num_fit], samples.cal_data_classes[:num_fit])
    cal_scaled = apply_vector_scaling(samples.cal_class_scores[num_fit:], T)
    test_scaled = apply_vector_scaling(samples.test_class_scores, T)
    return CalAndTestSamples(samples.cal_data_classes[num_fit:], samples.cal_data_samples[num_fit:], cal_scaled, samples.test_data_classes, samples.test_data_samples, test_scaled)
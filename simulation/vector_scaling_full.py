import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize

from simulation.interfaces import CalAndTestSamples


def probs_to_logits(p):
    return np.log(np.clip(p, 1e-12, 1.0))


def nll(params, logits, labels):
    K = logits.shape[1]
    a = params[:K]
    b = params[K:]
    scaled = a * logits + b
    log_probs = scaled - logsumexp(scaled, axis=1, keepdims=True)
    return -np.mean(log_probs[np.arange(len(labels)), labels])


def nll_constrained(params, logits, labels):
    K = logits.shape[1]
    a = params[:K]
    b_free = params[K:]
    b = np.append(b_free, -np.sum(b_free))
    return nll(np.concatenate([a, b]), logits, labels)


def fit_vector_scaling(cal_probs, cal_labels):
    cal_logits = probs_to_logits(cal_probs)
    K = cal_logits.shape[1]
    params0 = np.concatenate([np.ones(K), np.zeros(K - 1)])
    res = minimize(nll_constrained, params0, args=(cal_logits, cal_labels), method='L-BFGS-B')
    a = res.x[:K]
    b_free = res.x[K:]
    b = np.append(b_free, -np.sum(b_free))
    return a, b


def apply_vector_scaling(probs, a, b):
    logits = probs_to_logits(probs)
    scaled = a * logits + b
    scaled_probs = np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))
    return scaled_probs


def vector_scale_samples(samples, num_fit=100):
    a, b = fit_vector_scaling(samples.cal_class_scores[:num_fit], samples.cal_data_classes[:num_fit])
    cal_scaled = apply_vector_scaling(samples.cal_class_scores[num_fit:], a, b)
    test_scaled = apply_vector_scaling(samples.test_class_scores, a, b)
    return CalAndTestSamples(samples.cal_data_classes[num_fit:], samples.cal_data_samples[num_fit:], cal_scaled, samples.test_data_classes, samples.test_data_samples, test_scaled)
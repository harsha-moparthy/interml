import numpy as np
from scipy.special import logsumexp


def q370(student_logits, teacher_logits, temperature=2):
    """Compute temperature-scaled forward KL distillation loss from teacher probabilities
    to student probabilities, multiplying by temperature squared.
    """
    s = np.asarray(student_logits, float) / temperature
    t = np.asarray(teacher_logits, float) / temperature
    log_s = s - logsumexp(s, axis=-1, keepdims=True)
    log_t = t - logsumexp(t, axis=-1, keepdims=True)
    # Temperature-squared scaling compensates for the smaller softened-logit gradients.
    return float(
        temperature**2 * np.mean(np.sum(np.exp(log_t) * (log_t - log_s), axis=-1))
    )

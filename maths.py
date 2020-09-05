"""

"""
import numpy as np


def entropy(x, return_probs=False):
    """Computes the entropy of a given vector

    Args:
        x: a vector

    Returns:
        the entropy of vector, x, using log base e
    """
    N = len(x)

    # Handling the edge case where only one value is given
    if N <= 1:
        return 0

    probs = {
        val: np.count_nonzero(x == val) / N
        for val in np.array(x)
    }

    ent = -np.sum([v * np.log(v) for k, v in probs.items()])

    if return_probs:
        return ent, probs

    else:
        return ent


def kl_divergence(p, q):
    """Computes the Kullbeck-Liebler divergence between two given probability distribution

    Args:
        p:
        q:

    Returns:

    """
    ent_p, probs_p = entropy(x=p, return_probs=True)

    N = len(q)

    probs_q = {
        val: np.count_nonzero(q == val) / N
        for val in np.array(q)
    }

    cross_pq = 0
    pass

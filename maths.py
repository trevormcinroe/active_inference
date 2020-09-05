"""

"""
import numpy as np


def entropy(x):
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
    return ent

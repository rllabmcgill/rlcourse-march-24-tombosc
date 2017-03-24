import numpy as np    

def oh_encode(i, n):
    """ one-hot encode i in a vector of size n """
    if isinstance(i, np.ndarray):
        return i
    res = np.zeros(n)
    res[i] = 1.
    return res

def epsilon_greedify(p, epsilon):
    """
    epsilon-greedify vector
    """
    amax = np.argmax(p)
    n = p.shape[0]
    _p = np.ones(p.shape) * epsilon / n
    _p[amax] = 1 - ((n-1) * epsilon) / n
    return _p

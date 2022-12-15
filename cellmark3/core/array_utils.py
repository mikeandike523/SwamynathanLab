import functools

import numpy as np

def multiply_arrays(*args):
    return functools.reduce(np.multiply, args)

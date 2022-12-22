import functools

import numpy as np

def multiply_arrays(*args):
    return functools.reduce(np.multiply, args)

def bin_edges_to_midpoints(edges):
    return np.sum([edges[1:],edges[:-1]],axis=0)

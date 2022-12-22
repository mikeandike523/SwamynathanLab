import numpy as np

def find_threshold_minimizing_intra_class_variance(arr, steps=4096):
    min_value = np.min(arr)
    max_value = np.max(arr)
    test_values = np.linspace(min_value, max_value, steps)
    variances = []
    for test_value in test_values:
        left_values = arr[arr < test_value]
        right_values = arr[arr >= test_value]
        variances.append(
            ((np.max(left_values)-np.min(left_values))*np.var(left_values) if len(left_values) > 0 else 0)+
            ((np.max(right_values)-np.min(right_values))*np.var(right_values) if len(right_values) > 0 else 0)
        )
    return test_values[np.argmin(variances)]
import warnings

from .typing import is_iterable_but_not_string


def handle_varargs_or_iterable(args_array):
    if is_iterable_but_not_string(args_array[0]):
        if len(args_array) > 1:
            raise TypeError("If a list, tuple, or other iterable is supplied, no additional positional arguments can be supplied.")
        args_array = args_array[0]
    return args_array


def null_factory_with_warnings(key):
    warnings.warn(f"Provided default value for key `{key}`.")
    return None

def is_iterable(item):

    try:
        iter(item)
    except TypeError:
        return False

    return True

def is_iterable_but_not_string(item):

    return is_iterable(item) and type(item) is not str

        
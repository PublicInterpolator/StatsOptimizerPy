import operator


def indexed_sort(arr, reverse=False):
    """
    Sorts the the input iterable object.
    Returns sorted list and the indices of it's elements in initial iterable.

    Parameters
    ----------
    arr : list, ndarray
        The iterable object to be sorted.
    reverse : bool, optional
        If false(default) the array will be sorted in ascending order
        and in descending otherwise.

    Returns
    -------
    tuple[list, list]
        Sorted list, list of indices.
    """
    arr_sorted = sorted(enumerate(arr), key=operator.itemgetter(1), reverse=reverse)
    arr = [item[1] for item in arr_sorted]
    indices = [item[0] for item in arr_sorted]
    return arr, indices

import operator
import random
import functools


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
    indices = [item[0] for item in arr_sorted]
    arr = [item[1] for item in arr_sorted]
    return arr, indices


def random_event_generate(probability):
    """
    Returns True with given probability and False otherwise.

    Parameters
    ----------
    probability : float
        Probability of event (from [0, 1]).

    Returns
    -------
    bool
    """
    rand_max = int(1e10)
    if random.randrange(0, rand_max) < probability * rand_max:
        return True
    return False


def float2str(number, digits=2):
    """
    Converts float into string with fixed number of
    digits in fractional part.

    Parameters
    ----------
    number : float
        Number to be converted into string.
    digits : int, optional
        Number of digits in fractional part.

    Returns
    -------
    str
    """
    return f'{number:.{digits}f}'


def documentation_inheritance(base):
    """
    Decorator for copying documentation.

    Parameters
    ----------
    base : Any
        Original object.

    Returns
    -------
    Any
        Decorated object.
    """

    def wrapper(inheritor):
        inheritor.__doc__ = base.__doc__
        return inheritor

    return wrapper


def getter_by_id(max_id_getter):
    """
    Decorator for class methods with semantics of getting something by it's ID.
    Provides the following behavior: if no ID is passed, then method returns list of results
    for all correct IDs. If incorrect ID (less than 0 or greater than maximal ID) is passed,
    then returns None. Otherwise (if correct ID is passed) calls decorating method.

    Parameters
    ----------
    max_id_getter : callable
        Method to get maximal ID.

    Returns
    -------
    callable
        Decorator for getter method.
    """

    def decorator(getter_method):

        @functools.wraps(getter_method)
        def wrapper(self, id=None):
            max_id = max_id_getter(self)
            if id is None:
                return [getter_method(self, ID) for ID in range(max_id)]
            if not 0 <= id < max_id:
                return None
            return getter_method(self, id)

        return wrapper

    return decorator

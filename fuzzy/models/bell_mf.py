from simpful.fuzzy_sets import MF_object
import numpy as np


def _bell(x, a, b, c):
    return 1 / (1 + np.power( np.abs((x-c)/b), 2*a))


class Bell_MF(MF_object):
    """
        Creates a GeneralizedBell membership function.

        Args:
            a: related to the slope of the edges.
            b: width of the distribution.
            c: center of the distribution.
    """

    def __init__(self, a, b, c):  # noqa
        self.a = a
        self.b = b
        self.c = c

    def _execute(self, x):
        return _bell(x, self.a, self.b, self.c)

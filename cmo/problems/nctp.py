import numpy as np
import os
from cmo.problems.utils import CMOP, load_pareto_front_from_file


__all__ = ['NCTP1', 'NCTP2', 'NCTP3', 'NCTP4', 'NCTP5', 'NCTP6', 'NCTP7', 'NCTP8', 'NCTP9',
           'NCTP10', 'NCTP11', 'NCTP12', 'NCTP13', 'NCTP14', 'NCTP15', 'NCTP16', 'NCTP17', 'NCTP18']


# DEAP convention #


def nctp(X, theta, a, b, c, d, e, z1, z2=None):
    """
    Generator for constrained multiobjective problems NCTP.
    """

    X = np.atleast_2d(X)

    # Distance values
    t1 = X[:, 2:] - X[:, 1:-1] ** 2
    t2 = 1 - X[:, 1:-1]
    t3 = 100 * t1 ** 2 + t2 ** 2
    g = np.sum(t3, axis=1, keepdims=True)

    # Objective values
    f1 = X[:, [0]]
    f2 = g * (1 - np.sqrt(f1) / g) + z1

    # Constraint values
    h1 = np.cos(theta) * (f2 - e) - np.sin(theta) * f1
    h2 = np.sin(theta) * (f2 - e) + np.cos(theta) * f1
    h3 = a * np.abs(np.sin(b * np.pi * h2 ** c)) ** d
    c1 = h3 - h1

    if z2 is None:
        return np.column_stack([f1, f2, c1])
    else:
        c2 = f2 + 0.73 * f1 - z2
        return np.column_stack([f1, f2, c1, c2])


def nctp1(X):
    """
    NCTP1 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e, z1, z2 = -0.2 * np.pi, 0.2, 10, 1, 0.5, 1, -0.5, 4
    return nctp(X, theta, a, b, c, d, e, z1, z2)


def nctp2(X):
    """
    NCTP2 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e, z1, z2 = -0.2 * np.pi, 0.75, 10, 1, 0.5, 1, -0.5, 4
    return nctp(X, theta, a, b, c, d, e, z1, z2)


def nctp3(X):
    """
    NCTP3 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e, z1, z2 = -0.2 * np.pi, 2, 10, 1, 6, 1, -0.5, 6
    return nctp(X, theta, a, b, c, d, e, z1, z2)


def nctp4(X):
    """
    NCTP4 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e, z1 = -0.2 * np.pi, 0.2, 10, 1, 0.5, 1, -0.5
    return nctp(X, theta, a, b, c, d, e, z1)


def nctp5(X):
    """
    NCTP5 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e, z1 = -0.2 * np.pi, 0.75, 10, 1, 0.5, 1, -0.5
    return nctp(X, theta, a, b, c, d, e, z1)


def nctp6(X):
    """
    NCTP6 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e, z1 = -0.2 * np.pi, 2, 10, 1, 6, 1, -0.5
    return nctp(X, theta, a, b, c, d, e, z1)


def nctp7(X):
    """
    NCTP7 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e, z1, z2 = -0.2 * np.pi, 0.2, 10, 1, 0.5, 1, 1, 4
    return nctp(X, theta, a, b, c, d, e, z1, z2)


def nctp8(X):
    """
    NCTP8 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e, z1, z2 = -0.2 * np.pi, 0.75, 10, 1, 0.5, 1, 1, 4
    return nctp(X, theta, a, b, c, d, e, z1, z2)


def nctp9(X):
    """
    NCTP9 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e, z1, z2 = -0.2 * np.pi, 2, 10, 1, 6, 1, 1, 6
    return nctp(X, theta, a, b, c, d, e, z1, z2)


def nctp10(X):
    """
    NCTP10 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e, z1 = -0.2 * np.pi, 0.2, 10, 1, 0.5, 1, 1
    return nctp(X, theta, a, b, c, d, e, z1)


def nctp11(X):
    """
    NCTP11 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e, z1 = -0.2 * np.pi, 0.75, 10, 1, 0.5, 1, 1
    return nctp(X, theta, a, b, c, d, e, z1)


def nctp12(X):
    """
    NCTP12 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e, z1 = -0.2 * np.pi, 2, 10, 1, 6, 1, 1
    return nctp(X, theta, a, b, c, d, e, z1)


def nctp13(X):
    """
    NCTP13 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e, z1, z2 = -0.2 * np.pi, 0.2, 10, 1, 0.5, 1, 2, 4
    return nctp(X, theta, a, b, c, d, e, z1, z2)


def nctp14(X):
    """
    NCTP14 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e, z1, z2 = -0.2 * np.pi, 0.75, 10, 1, 0.5, 1, 2.5, 4
    return nctp(X, theta, a, b, c, d, e, z1, z2)


def nctp15(X):
    """
    NCTP15 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e, z1, z2 = -0.2 * np.pi, 2, 10, 1, 6, 1, 4, 6
    return nctp(X, theta, a, b, c, d, e, z1, z2)


def nctp16(X):
    """
    NCTP16 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e, z1 = -0.2 * np.pi, 0.2, 10, 1, 0.5, 1, 2
    return nctp(X, theta, a, b, c, d, e, z1)


def nctp17(X):
    """
    NCTP17 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e, z1 = -0.2 * np.pi, 0.75, 10, 1, 0.5, 1, 2.5
    return nctp(X, theta, a, b, c, d, e, z1)


def nctp18(X):
    """
    NCTP18 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e, z1 = -0.2 * np.pi, 2, 10, 1, 6, 1, 4
    return nctp(X, theta, a, b, c, d, e, z1)


# Generator #


class NCTP(CMOP):
    """
    The NCPT test suite generator of constrained multibjecitve problems NCTP1-NCTP18.

    Parameters
    ----------
    :param prob_id (int): NCTP problem id.
    :param n_var (int): Dimension of the decision space.
    :param scale_var (bool, optional): Whether the decision variables are scaled to [0, 1]. Default is False.
    :param scale_obj (bool, optional): Whether the objective values are scaled to [0, 1]. Default is False.

    Raise
    -----
    :raise ValueError: If prob_id is not in {1, ..., 18} or n_var is smaller than 3.

    References
    ----------
    [Li2016] J. Li, Y. Wang, S. Yang and Z. Cai, "A comparative study of constraint-handling
    techniques in evolutionary constrained multiobjective optimization," 2016 IEEE Congress on
    Evolutionary Computation (CEC), 2016, pp. 4175-4182, doi: 10.1109/CEC.2016.7744320.
    """

    _prob = {
        1: nctp1,
        2: nctp2,
        3: nctp3,
        4: nctp4,
        5: nctp5,
        6: nctp6,
        7: nctp7,
        8: nctp8,
        9: nctp9,
        10: nctp10,
        11: nctp11,
        12: nctp12,
        13: nctp13,
        14: nctp14,
        15: nctp15,
        16: nctp16,
        17: nctp17,
        18: nctp18
    }

    _params = {
        1: {'theta': -0.2 * np.pi, 'a': 0.2, 'b': 10, 'c': 1, 'd': 0.5, 'e': 1, 'z1': -0.5, 'z2': 4},
        2: {'theta': -0.2 * np.pi, 'a': 0.75, 'b': 10, 'c': 1, 'd': 0.5, 'e': 1, 'z1': -0.5, 'z2': 4},
        3: {'theta': -0.2 * np.pi, 'a': 2, 'b': 10, 'c': 1, 'd': 6, 'e': 1, 'z1': -0.5, 'z2': 6},
        4: {'theta': -0.2 * np.pi, 'a': 0.2, 'b': 10, 'c': 1, 'd': 0.5, 'e': 1, 'z1': -0.5, 'z2': None},
        5: {'theta': -0.2 * np.pi, 'a': 0.75, 'b': 10, 'c': 1, 'd': 0.5, 'e': 1, 'z1': -0.5, 'z2': None},
        6: {'theta': -0.2 * np.pi, 'a': 2, 'b': 10, 'c': 1, 'd': 6, 'e': 1, 'z1': -0.5, 'z2': None},
        7: {'theta': -0.2 * np.pi, 'a': 0.2, 'b': 10, 'c': 1, 'd': 0.5, 'e': 1, 'z1': 1, 'z2': 4},
        8: {'theta': -0.2 * np.pi, 'a': 0.75, 'b': 10, 'c': 1, 'd': 0.5, 'e': 1, 'z1': 1, 'z2': 4},
        9: {'theta': -0.2 * np.pi, 'a': 2, 'b': 10, 'c': 1, 'd': 6, 'e': 1, 'z1': 1, 'z2': 6},
        10: {'theta': -0.2 * np.pi, 'a': 0.2, 'b': 10, 'c': 1, 'd': 0.5, 'e': 1, 'z1': 1, 'z2': None},
        11: {'theta': -0.2 * np.pi, 'a': 0.75, 'b': 10, 'c': 1, 'd': 0.5, 'e': 1, 'z1': 1, 'z2': None},
        12: {'theta': -0.2 * np.pi, 'a': 2, 'b': 10, 'c': 1, 'd': 6, 'e': 1, 'z1': 1, 'z2': None},
        13: {'theta': -0.2 * np.pi, 'a': 0.2, 'b': 10, 'c': 1, 'd': 0.5, 'e': 1, 'z1': 2, 'z2': 4},
        14: {'theta': -0.2 * np.pi, 'a': 0.75, 'b': 10, 'c': 1, 'd': 0.5, 'e': 1, 'z1': 2.5, 'z2': 4},
        15: {'theta': -0.2 * np.pi, 'a': 2, 'b': 10, 'c': 1, 'd': 6, 'e': 1, 'z1': 4, 'z2': 6},
        16: {'theta': -0.2 * np.pi, 'a': 0.2, 'b': 10, 'c': 1, 'd': 0.5, 'e': 1, 'z1': 2, 'z2': None},
        17: {'theta': -0.2 * np.pi, 'a': 0.75, 'b': 10, 'c': 1, 'd': 0.5, 'e': 1, 'z1': 2.5, 'z2': None},
        18: {'theta': -0.2 * np.pi, 'a': 2, 'b': 10, 'c': 1, 'd': 6, 'e': 1, 'z1': 4, 'z2': None}
    }

    def __init__(self, prob_id, n_var=10, scale_var=False, scale_obj=False):

        if prob_id not in set(range(1, 19)):
            raise ValueError("Please select a valid prob id.")

        # Set fn
        self.fn = self._prob[prob_id]

        # Set dim
        if n_var < 3:
            raise ValueError("Please select a larger value for n_var (>= 3).")

        n_iq_constr = 2 if prob_id in [1, 2, 3, 7, 8, 9, 13, 14, 15] else 1

        # Set xl, xu
        xl = np.array([0] * n_var)
        xu = np.array([5] * n_var)

        # Set nadir and ideal points
        params = self._params[prob_id]

        self.theta = params['theta']
        self.a = params['a']
        self.b = params['b']
        self.c = params['c']
        self.d = params['d']
        self.e = params['e']
        self.z1 = params['z1']
        self.z2 = params['z2']

        ideal_point = None
        nadir_point = None

        # Set name
        name = "NCTP{}".format(str(prob_id))

        super(NCTP, self).__init__(n_var=n_var,
                                   n_obj=2,
                                   n_iq_constr=n_iq_constr,
                                   n_eq_constr=0,
                                   xl=xl,
                                   xu=xu,
                                   scale_var=scale_var,
                                   scale_obj=scale_obj,
                                   nadir_point=nadir_point,
                                   ideal_point=ideal_point,
                                   name=name)

    def _fn(self, X):
        return self.fn(X)

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("NCTP", fname))


# Pymoo convention #


class NCTP1(NCTP):
    """
    NCTP1 constrained multiobjective optimization problem.
    """
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('No. of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('No. of variables must be greater than two.')

        super(NCTP1, self).__init__(prob_id=1, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)


class NCTP2(NCTP):
    """
    NCTP2 constrained multiobjective optimization problem.
    """
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('No. of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('No. of variables must be greater than two.')

        super(NCTP2, self).__init__(prob_id=2, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)


class NCTP3(NCTP):
    """
    NCTP3 constrained multiobjective optimization problem.
    """
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('No. of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('No. of variables must be greater than two.')

        super(NCTP3, self).__init__(prob_id=3, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)


class NCTP4(NCTP):
    """
    NCTP4 constrained multiobjective optimization problem.
    """
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('No. of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('No. of variables must be greater than two.')

        super(NCTP4, self).__init__(prob_id=4, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)


class NCTP5(NCTP):
    """
    NCTP5 constrained multiobjective optimization problem.
    """
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('No. of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('No. of variables must be greater than two.')

        super(NCTP5, self).__init__(prob_id=5, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)


class NCTP6(NCTP):
    """
    NCTP6 constrained multiobjective optimization problem.
    """
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('No. of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('No. of variables must be greater than two.')

        super(NCTP6, self).__init__(prob_id=6, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)


class NCTP7(NCTP):
    """
    NCTP7 constrained multiobjective optimization problem.
    """
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('No. of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('No. of variables must be greater than two.')

        super(NCTP7, self).__init__(prob_id=7, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)


class NCTP8(NCTP):
    """
    NCTP8 constrained multiobjective optimization problem.
    """
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('No. of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('No. of variables must be greater than two.')

        super(NCTP8, self).__init__(prob_id=8, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)


class NCTP9(NCTP):
    """
    NCTP9 constrained multiobjective optimization problem.
    """
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('No. of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('No. of variables must be greater than two.')

        super(NCTP9, self).__init__(prob_id=9, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)


class NCTP10(NCTP):
    """
    NCTP10 constrained multiobjective optimization problem.
    """
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('No. of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('No. of variables must be greater than two.')

        super(NCTP10, self).__init__(prob_id=10, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)


class NCTP11(NCTP):
    """
    NCTP11 constrained multiobjective optimization problem.
    """
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('No. of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('No. of variables must be greater than two.')

        super(NCTP11, self).__init__(prob_id=11, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)


class NCTP12(NCTP):
    """
    NCTP12 constrained multiobjective optimization problem.
    """
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('No. of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('No. of variables must be greater than two.')

        super(NCTP12, self).__init__(prob_id=12, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)


class NCTP13(NCTP):
    """
    NCTP13 constrained multiobjective optimization problem.
    """
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('No. of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('No. of variables must be greater than two.')

        super(NCTP13, self).__init__(prob_id=13, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)


class NCTP14(NCTP):
    """
    NCTP14 constrained multiobjective optimization problem.
    """
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('No. of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('No. of variables must be greater than two.')

        super(NCTP14, self).__init__(prob_id=14, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)


class NCTP15(NCTP):
    """
    NCTP15 constrained multiobjective optimization problem.
    """
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('No. of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('No. of variables must be greater than two.')

        super(NCTP15, self).__init__(prob_id=15, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)


class NCTP16(NCTP):
    """
    NCTP16 constrained multiobjective optimization problem.
    """
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('No. of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('No. of variables must be greater than two.')

        super(NCTP16, self).__init__(prob_id=16, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)


class NCTP17(NCTP):
    """
    NCTP17 constrained multiobjective optimization problem.
    """
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('No. of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('No. of variables must be greater than two.')

        super(NCTP17, self).__init__(prob_id=17, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)


class NCTP18(NCTP):
    """
    NCTP18 constrained multiobjective optimization problem.
    """
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('No. of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('No. of variables must be greater than two.')

        super(NCTP18, self).__init__(prob_id=18, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)

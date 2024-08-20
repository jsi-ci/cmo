import os
import numpy as np
from cmo.problems.utils import CMOP, load_pareto_front_from_file

__all__ = ['CF1', 'CF2', 'CF3', 'CF4', 'CF5', 'CF6', 'CF7', 'CF8', 'CF9', 'CF10']


# DEAP convention #


def to_div_2(X):
    n = X.shape[1]
    i = np.arange(1, n + 1, 1)
    i1 = np.arange(2, n, 2)
    i2 = np.arange(1, n, 2)
    return i, i1, i2


def to_div_3(X):
    n = X.shape[1]
    i = np.arange(1, n + 1, 1)
    i1 = np.arange(3, n, 3)
    i2 = np.arange(4, n, 3)
    i3 = np.arange(2, n, 3)
    return i, i1, i2, i3


def to_sin(X, i):
    return X - np.sin(6 * np.pi * X[:, [0]] + i * np.pi / X.shape[1])


def to_cos(X, i):
    return X - np.cos(6 * np.pi * X[:, [0]] + i * np.pi / X.shape[1])


def to_sin2(X, i):
    return X - 0.8 * X[:, [0]] * np.sin(6 * np.pi * X[:, [0]] + i * np.pi / X.shape[1])


def to_cos2(X, i):
    return X - 0.8 * X[:, [0]] * np.cos(6 * np.pi * X[:, [0]] + i * np.pi / X.shape[1])


def to_sin3(X, i):
    return X - 2 * X[:, [1]] * np.sin(2 * np.pi * X[:, [0]] + i * np.pi / X.shape[1])


def to_cos3(X, i):
    Y = to_sin3(X, i)
    return 4 * Y ** 2 - np.cos(8 * np.pi * Y) + 1


def cf1(X, N=10, a=1):
    """
    CF1 constrained multiobjective optimization problem. It returns a tuple of 3 = (2 objective + 1 constraint) values.

    From: Q. Zhang, A. Zhou, S. Zhao, P.N. Suganthan, W. Lium and S. Tiwari, "Multiobjective optimization Test Instances
    for the CEC 2009 Special Session and Competition," Technical Report CES-487
    """

    # Objective values
    i, i1, i2 = to_div_2(X)

    h = (X - X[:, [0]] ** (0.5 * (1.0 + 3.0 * (i - 2) / (X.shape[1] - 2)))) ** 2
    f1 = X[:, [0]] + 2 * np.mean(h[:, i1], axis=1, keepdims=True)
    f2 = 1 - X[:, [0]] + 2 * np.mean(h[:, i2], axis=1, keepdims=True)

    # Constraint values
    c = f1 + f2 - a * np.abs(np.sin(N * np.pi * (f1 - f2 + 1))) - 1

    return f1, f2, -c


def cf2(X, N=2, a=1):
    """
    CF2 constrained multiobjective optimization problem. It returns a tuple of 3 = (2 objective + 1 constraint) values.

    From: Q. Zhang, A. Zhou, S. Zhao, P.N. Suganthan, W. Lium and S. Tiwari, "Multiobjective optimization Test Instances
    for the CEC 2009 Special Session and Competition," Technical Report CES-487
    """

    # Objective values
    i, i1, i2 = to_div_2(X)
    y = to_sin(X, i)
    z = to_cos(X, i)

    f1 = X[:, [0]] + 2 * np.mean(y[:, i1] ** 2, axis=1, keepdims=True)
    f2 = 1 - np.sqrt(X[:, [0]]) + 2 * np.mean(z[:, i2] ** 2, axis=1, keepdims=True)

    # Constraint values
    t = f2 + np.sqrt(f1) - a * np.sin(N * np.pi * (np.sqrt(f1) - f2 + 1)) - 1
    c = t / (1 + np.exp(4 * np.abs(t)))

    return f1, f2, -c


def cf3(X, N=2, a=1):
    """
    CF3 constrained multiobjective optimization problem. It returns a tuple of 3 = (2 objective + 1 constraint) values.

    From: Q. Zhang, A. Zhou, S. Zhao, P.N. Suganthan, W. Lium and S. Tiwari, "Multiobjective optimization Test Instances
    for the CEC 2009 Special Session and Competition," Technical Report CES-487
    """

    # Objective values
    i, i1, i2 = to_div_2(X)
    y = to_sin(X, i)

    h1 = np.sum(y[:, i1] ** 2, axis=1, keepdims=True)
    h2 = np.prod(np.cos(20 * y[:, i1] * np.pi / np.sqrt(i1 + 1)), axis=1, keepdims=True)
    h3 = np.sum(y[:, i2] ** 2, axis=1, keepdims=True)
    h4 = np.prod(np.cos(20 * y[:, i2] * np.pi / np.sqrt(i2 + 1)), axis=1, keepdims=True)

    f1 = X[:, [0]] + 2 / len(i1) * (4 * h1 - 2 * h2 + 2)
    f2 = 1 - X[:, [0]] ** 2 + 2 / len(i2) * (4 * h3 - 2 * h4 + 2)

    # Constraint values
    c = f1 ** 2 + f2 - a * np.sin(N * np.pi * (f1 ** 2 - f2 + 1)) - 1

    return f1, f2, -c


def cf4(X):
    """
    CF4 constrained multiobjective optimization problem. It returns a tuple of 3 = (2 objective + 1 constraint) values.

    From: Q. Zhang, A. Zhou, S. Zhao, P.N. Suganthan, W. Lium and S. Tiwari, "Multiobjective optimization Test Instances
    for the CEC 2009 Special Session and Competition," Technical Report CES-487
    """

    # Objective values
    i, i1, i2 = to_div_2(X)
    y = to_sin(X, i)

    h = y ** 2
    h[:, 1] = np.abs(y[:, 1])
    change = y[:, 1] >= 3 / 2 * (1 - np.sqrt(2) / 2)
    h[change, 1] = 0.125 + (y[change, 1] - 1) ** 2

    f1 = X[:, [0]] + np.sum(h[:, i1], axis=1, keepdims=True)
    f2 = 1 - X[:, [0]] + np.sum(h[:, i2], axis=1, keepdims=True)

    # Constraint values
    t = X[:, [1]] - np.sin(6 * np.pi * X[:, [0]] + 2 * np.pi / X.shape[1]) - 0.5 * X[:, [0]] + 0.25
    c = t / (1 + np.exp(4 * np.abs(t)))

    return f1, f2, -c


def cf5(X):
    """
    CF5 constrained multiobjective optimization problem. It returns a tuple of 3 = (2 objective + 1 constraint) values.

    From: Q. Zhang, A. Zhou, S. Zhao, P.N. Suganthan, W. Lium and S. Tiwari, "Multiobjective optimization Test Instances
    for the CEC 2009 Special Session and Competition," Technical Report CES-487
    """

    # Objective values
    i, i1, i2 = to_div_2(X)
    y = to_sin2(X, i)
    z = to_cos2(X, i)

    h1 = 2 * z[:, i1] ** 2 - np.cos(4 * np.pi * z[:, i1]) + 1
    h2 = 2 * y[:, i2] ** 2 - np.cos(4 * np.pi * y[:, i2]) + 1
    h2[:, 0] = np.abs(y[:, 1])
    change = y[:, 1] >= 3 / 2 * (1 - np.sqrt(2) / 2)
    h2[change, 0] = 0.125 + (y[change, 1] - 1) ** 2

    f1 = X[:, [0]] + np.sum(h1, axis=1, keepdims=True)
    f2 = 1 - X[:, [0]] + np.sum(h2, axis=1, keepdims=True)

    # Constraint values
    c = X[:, [1]] - 0.8 * X[:, [0]] * np.sin(6 * np.pi * X[:, [0]] + 2 * np.pi / X.shape[1]) - 0.5 * X[:, [0]] + 0.25

    return f1, f2, -c


def cf6(X):
    """
    CF6 constrained multiobjective optimization problem. It returns a tuple of 4 = (2 objective + 2 constraint) values.

    From: Q. Zhang, A. Zhou, S. Zhao, P.N. Suganthan, W. Lium and S. Tiwari, "Multiobjective optimization Test Instances
    for the CEC 2009 Special Session and Competition," Technical Report CES-487
    """

    # Objective values
    i, i1, i2 = to_div_2(X)
    y = to_sin2(X, i)
    z = to_cos2(X, i)

    f1 = X[:, [0]] + np.sum(z[:, i1] ** 2, axis=1, keepdims=True)
    f2 = (1 - X[:, [0]]) ** 2 + np.sum(y[:, i2] ** 2, axis=1, keepdims=True)

    # Constraint values
    h3 = 0.5 * (1 - X[:, [0]]) - (1 - X[:, [0]]) ** 2
    h4 = 0.25 * np.sqrt(1 - X[:, [0]]) - 0.5 * (1 - X[:, [0]])
    c1 = X[:, [1]] - 0.8 * X[:, [0]] * np.sin(6 * np.pi * X[:, [0]] + 2 * np.pi / X.shape[1]) - np.sign(h3) * np.sqrt(
        np.abs(h3))
    c2 = X[:, [3]] - 0.8 * X[:, [0]] * np.sin(6 * np.pi * X[:, [0]] + 4 * np.pi / X.shape[1]) - np.sign(h4) * np.sqrt(
        np.abs(h4))

    return f1, f2, -c1, -c2


def cf7(X):
    """
    CF7 constrained multiobjective optimization problem. It returns a tuple of 4 = (2 objective + 2 constraint) values.

    From: Q. Zhang, A. Zhou, S. Zhao, P.N. Suganthan, W. Lium and S. Tiwari, "Multiobjective optimization Test Instances
    for the CEC 2009 Special Session and Competition," Technical Report CES-487
    """

    # Objective values
    i, i1, i2 = to_div_2(X)
    y = to_sin(X, i)
    z = to_cos(X, i)

    h1 = 2 * z[:, i1] ** 2 - np.cos(4 * np.pi * z[:, i1]) + 1
    h2 = 2 * y[:, i2] ** 2 - np.cos(4 * np.pi * y[:, i2]) + 1
    h2[:, :2] = y[:, [1, 3]] ** 2

    f1 = X[:, [0]] + np.sum(h1, axis=1, keepdims=True)
    f2 = (1 - X[:, [0]]) ** 2 + np.sum(h2, axis=1, keepdims=True)

    # Constraint values
    h3 = 0.5 * (1 - X[:, [0]]) - (1 - X[:, [0]]) ** 2
    h4 = 0.25 * np.sqrt((1 - X[:, [0]])) - 0.5 * (1 - X[:, [0]])
    c1 = X[:, [1]] - np.sin(6 * np.pi * X[:, [0]] + 2 * np.pi / X.shape[1]) - np.sign(h3) * np.sqrt(np.abs(h3))
    c2 = X[:, [3]] - np.sin(6 * np.pi * X[:, [0]] + 4 * np.pi / X.shape[1]) - np.sign(h4) * np.sqrt(np.abs(h4))

    return f1, f2, -c1, -c2


def cf8(X, N=2, a=4):
    """
    CF8 constrained multiobjective optimization problem. It returns a tuple of 4 = (3 objective + 1 constraint) values.

    From: Q. Zhang, A. Zhou, S. Zhao, P.N. Suganthan, W. Lium and S. Tiwari, "Multiobjective optimization Test Instances
    for the CEC 2009 Special Session and Competition," Technical Report CES-487
    """

    # Objective values
    i, i1, i2, i3 = to_div_3(X)
    y = to_sin3(X, i)

    h1 = 0.5 * X[:, [0]] * np.pi
    h2 = 0.5 * X[:, [1]] * np.pi

    f1 = np.cos(h1) * np.cos(h2) + 2 * np.mean(y[:, i1] ** 2, axis=1, keepdims=True)
    f2 = np.cos(h1) * np.sin(h2) + 2 * np.mean(y[:, i2] ** 2, axis=1, keepdims=True)
    f3 = np.sin(h1) + 2 * np.mean(y[:, i3] ** 2, axis=1, keepdims=True)

    # Constraint values
    h3 = (f1 ** 2 + f2 ** 2) / (1 - f3 ** 2)
    h4 = (f1 ** 2 - f2 ** 2) / (1 - f3 ** 2)
    c = h3 - a * np.abs(np.sin(N * np.pi * (h4 + 1))) - 1

    return f1, f2, f3, -c


def cf9(X, N=2, a=3):
    """
    CF9 constrained multiobjective optimization problem. It returns a tuple of 4 = (3 objective + 1 constraint) values.

    From: Q. Zhang, A. Zhou, S. Zhao, P.N. Suganthan, W. Lium and S. Tiwari, "Multiobjective optimization Test Instances
    for the CEC 2009 Special Session and Competition," Technical Report CES-487
    """

    # Objective values
    i, i1, i2, i3 = to_div_3(X)
    y = to_sin3(X, i)

    h1 = 0.5 * X[:, [0]] * np.pi
    h2 = 0.5 * X[:, [1]] * np.pi

    f1 = np.cos(h1) * np.cos(h2) + 2 * np.mean(y[:, i1] ** 2, axis=1, keepdims=True)
    f2 = np.cos(h1) * np.sin(h2) + 2 * np.mean(y[:, i2] ** 2, axis=1, keepdims=True)
    f3 = np.sin(h1) + 2 * np.mean(y[:, i3] ** 2, axis=1, keepdims=True)

    # Constraint values
    h3 = (f1 ** 2 + f2 ** 2) / (1 - f3 ** 2)
    h4 = (f1 ** 2 - f2 ** 2) / (1 - f3 ** 2)
    c = h3 - a * np.sin(N * np.pi * (h4 + 1)) - 1

    return f1, f2, f3, -c


def cf10(X, N=2, a=1):
    """
    CF10 constrained multiobjective optimization problem. It returns a tuple of 4 = (3 objective + 1 constraint) values.

    From: Q. Zhang, A. Zhou, S. Zhao, P.N. Suganthan, W. Lium and S. Tiwari, "Multiobjective optimization Test Instances
    for the CEC 2009 Special Session and Competition," Technical Report CES-487
    """

    # Objective values
    i, i1, i2, i3 = to_div_3(X)
    z = to_cos3(X, i)

    h1 = 0.5 * X[:, [0]] * np.pi
    h2 = 0.5 * X[:, [1]] * np.pi

    f1 = np.cos(h1) * np.cos(h2) + 2 * np.mean(z[:, i1], axis=1, keepdims=True)
    f2 = np.cos(h1) * np.sin(h2) + 2 * np.mean(z[:, i2], axis=1, keepdims=True)
    f3 = np.sin(h1) + 2 * np.mean(z[:, i3], axis=1, keepdims=True)

    # Constraint values
    h3 = (f1 ** 2 + f2 ** 2) / (1 - f3 ** 2)
    h4 = (f1 ** 2 - f2 ** 2) / (1 - f3 ** 2)
    c = h3 - a * np.sin(N * np.pi * (h4 + 1)) - 1

    return f1, f2, f3, -c


# Pymoo convention #


class CF1(CMOP):
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False, N=10, a=1):
        if n_obj != 2:
            raise ValueError("CF problems have fixed numbers of objectives.")

        if n_var < 2:
            raise ValueError('Incorrect number of variables:', n_var)

        super(CF1, self).__init__(n_var=n_var, n_obj=n_obj, n_iq_constr=1, n_eq_constr=0, xl=0., xu=1.,
                                  scale_var=scale_var, scale_obj=scale_obj, nadir_point=None,
                                  ideal_point=None, name="CF1")
        self.N = N
        self.a = a

    def _fn(self, X):
        X = np.atleast_2d(X)
        return np.column_stack(cf1(X, N=self.N, a=self.a))

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("CF", fname))


class CF2(CMOP):
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False, N=2, a=1):
        xl = np.array([0.] + [-1.] * (n_var - 1))
        xu = np.array([1.] * n_var)

        if n_obj != 2:
            raise ValueError("CF problems have fixed numbers of objectives.")

        if n_var <= 2:
            raise ValueError('Incorrect number of variables:', n_var)

        super(CF2, self).__init__(n_var=n_var, n_obj=n_obj, n_iq_constr=1, n_eq_constr=0, xl=xl, xu=xu,
                                  scale_var=scale_var, scale_obj=scale_obj, nadir_point=None,
                                  ideal_point=None, name="CF2")
        self.N = N
        self.a = a

    def _fn(self, X):
        X = np.atleast_2d(X)
        return np.column_stack(cf2(X, N=self.N, a=self.a))

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("CF", fname))


class CF3(CMOP):
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False, N=2, a=1):
        xl = np.array([0.] + [-2.] * (n_var - 1))
        xu = np.array([1.] + [2.] * (n_var - 1))

        if n_obj != 2:
            raise ValueError("CF problems have fixed numbers of objectives.")

        if n_var <= 2:
            raise ValueError('Incorrect number of variables:', n_var)

        super(CF3, self).__init__(n_var=n_var, n_obj=n_obj, n_iq_constr=1, n_eq_constr=0, xl=xl, xu=xu,
                                  scale_var=scale_var, scale_obj=scale_obj, nadir_point=None,
                                  ideal_point=None, name="CF3")
        self.N = N
        self.a = a

    def _fn(self, X):
        X = np.atleast_2d(X)
        return np.column_stack(cf3(X, N=self.N, a=self.a))

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("CF", fname))


class CF4(CMOP):
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        xl = np.array([0.] + [-2.] * (n_var - 1))
        xu = np.array([1.] + [2.] * (n_var - 1))

        if n_obj != 2:
            raise ValueError("CF problems have fixed numbers of objectives.")

        if n_var <= 2:
            raise ValueError('Incorrect number of variables:', n_var)

        super(CF4, self).__init__(n_var=n_var, n_obj=n_obj, n_iq_constr=1, n_eq_constr=0, xl=xl, xu=xu,
                                  scale_var=scale_var, scale_obj=scale_obj, nadir_point=None,
                                  ideal_point=None, name="CF4")

    def _fn(self, X):
        X = np.atleast_2d(X)
        return np.column_stack(cf4(X))

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("CF", fname))


class CF5(CMOP):
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        xl = np.array([0.] + [-2.] * (n_var - 1))
        xu = np.array([1.] + [2.] * (n_var - 1))

        if n_obj != 2:
            raise ValueError("CF problems have fixed numbers of objectives.")

        if n_var <= 2:
            raise ValueError('Incorrect number of variables:', n_var)

        super(CF5, self).__init__(n_var=n_var, n_obj=n_obj, n_iq_constr=1, n_eq_constr=0, xl=xl, xu=xu,
                                  scale_var=scale_var, scale_obj=scale_obj, nadir_point=None,
                                  ideal_point=None, name="CF5")

    def _fn(self, X):
        X = np.atleast_2d(X)
        return np.column_stack(cf5(X))

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("CF", fname))


class CF6(CMOP):
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        xl = np.array([0.] + [-2.] * (n_var - 1))
        xu = np.array([1.] + [2.] * (n_var - 1))

        if n_obj != 2:
            raise ValueError("CF problems have fixed numbers of objectives.")

        if n_var < 4:
            raise ValueError('Incorrect number of variables:', n_var)

        super(CF6, self).__init__(n_var=n_var, n_obj=n_obj, n_iq_constr=2, n_eq_constr=0, xl=xl, xu=xu,
                                  scale_var=scale_var, scale_obj=scale_obj, nadir_point=None,
                                  ideal_point=None, name="CF6")

    def _fn(self, X):
        X = np.atleast_2d(X)
        return np.column_stack(cf6(X))

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("CF", fname))


class CF7(CMOP):
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        xl = np.array([0.] + [-2.] * (n_var - 1))
        xu = np.array([1.] + [2.] * (n_var - 1))

        if n_obj != 2:
            raise ValueError("CF problems have fixed numbers of objectives.")

        if n_var < 4:
            raise ValueError('Incorrect number of variables:', n_var)

        super(CF7, self).__init__(n_var=n_var, n_obj=n_obj, n_iq_constr=2, n_eq_constr=0, xl=xl, xu=xu,
                                  scale_var=scale_var, scale_obj=scale_obj, nadir_point=None,
                                  ideal_point=None, name="CF7")

    def _fn(self, X):
        X = np.atleast_2d(X)
        return np.column_stack(cf7(X))

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("CF", fname))


class CF8(CMOP):
    def __init__(self, n_obj=3, n_var=10, scale_var=False, scale_obj=False, N=2, a=4):
        xl = np.array([0.] * 2 + [-4.] * (n_var - 2))
        xu = np.array([1.] * 2 + [4.] * (n_var - 2))

        if n_obj != 3:
            raise ValueError("CF problems have fixed numbers of objectives.")

        if n_var < 5:
            raise ValueError('Incorrect number of variables:', n_var)

        super(CF8, self).__init__(n_var=n_var, n_obj=n_obj, n_iq_constr=1, n_eq_constr=0, xl=xl, xu=xu,
                                  scale_var=scale_var, scale_obj=scale_obj, nadir_point=None,
                                  ideal_point=None, name="CF8")
        self.N = N
        self.a = a

    def _fn(self, X):
        X = np.atleast_2d(X)
        return np.column_stack(cf8(X, N=self.N, a=self.a))

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("CF", fname))


class CF9(CMOP):
    def __init__(self, n_obj=3, n_var=10, scale_var=False, scale_obj=False, N=2, a=3):
        xl = np.array([0.] * 2 + [-2.] * (n_var - 2))
        xu = np.array([1.] * 2 + [2.] * (n_var - 2))

        if n_obj != 3:
            raise ValueError("CF problems have fixed numbers of objectives.")

        if n_var < 5:
            raise ValueError('Incorrect number of variables:', n_var)

        super(CF9, self).__init__(n_var=n_var, n_obj=n_obj, n_iq_constr=1, n_eq_constr=0, xl=xl, xu=xu,
                                  scale_var=scale_var, scale_obj=scale_obj, nadir_point=None,
                                  ideal_point=None, name="CF9")
        self.N = N
        self.a = a

    def _fn(self, X):
        X = np.atleast_2d(X)
        return np.column_stack(cf9(X, N=self.N, a=self.a))

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("CF", fname))


class CF10(CMOP):
    def __init__(self, n_obj=3, n_var=10, scale_var=False, scale_obj=False, N=2, a=1):
        xl = np.array([0.] * 2 + [-2.] * (n_var - 2))
        xu = np.array([1.] * 2 + [2.] * (n_var - 2))

        if n_obj != 3:
            raise ValueError("CF problems have fixed numbers of objectives.")

        if n_var < 5:
            raise ValueError('Incorrect number of variables:', n_var)

        super(CF10, self).__init__(n_var=n_var, n_obj=n_obj, n_iq_constr=1, n_eq_constr=0, xl=xl, xu=xu,
                                   scale_var=scale_var, scale_obj=scale_obj, nadir_point=None,
                                   ideal_point=None, name="CF10")
        self.N = N
        self.a = a

    def _fn(self, X):
        X = np.atleast_2d(X)
        return np.column_stack(cf10(X, N=self.N, a=self.a))

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("CF", fname))
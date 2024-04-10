import os
import numpy as np
from cmop.utils import CMOP, load_pareto_front_from_file

from pymoo.util.ref_dirs import get_reference_directions


__all__ = ['MW1', 'MW2', 'MW3', 'MW4', 'MW5', 'MW6', 'MW7', 'MW8', 'MW9', 'MW10', 'MW11',
           'MW12', 'MW13', 'MW14']


class MW(CMOP):

    _n_iq_constrs = {
        **dict.fromkeys([1, 2, 4, 6, 8, 9, 14], 1),
        **dict.fromkeys([3, 7, 12, 13], 2),
        **dict.fromkeys([5, 10], 3),
        **dict.fromkeys([11], 4)
    }

    _xus = {
        **dict.fromkeys([1, 2, 3, 4, 5, 7, 8, 9, 10, 12], 1),
        **dict.fromkeys([6], 1.1),
        **dict.fromkeys([13, 14], 1.5),
        **dict.fromkeys([11], np.sqrt(2))
    }

    def __init__(self, prob_id, n_var=10, n_obj=3, scale_var=False, scale_obj=False):

        if prob_id not in range(1, 15):
            raise ValueError("Please select a valid prob id.")

        # Set dim
        n_iq_constr = self._n_iq_constrs[prob_id]

        # Set xl, xu
        xl = 0
        xu = self._xus[prob_id]

        # Set nadir and ideal points
        ideal_point = None
        nadir_point = None

        # Set name
        name = 'MW{}'.format(prob_id)

        super(MW, self).__init__(n_var=n_var,
                                 n_obj=n_obj,
                                 n_iq_constr=n_iq_constr,
                                 n_eq_constr=0,
                                 xl=xl,
                                 xu=xu,
                                 scale_var=scale_var,
                                 scale_obj=scale_obj,
                                 nadir_point=nadir_point,
                                 ideal_point=ideal_point,
                                 name=name)

    @staticmethod
    def LA1(A, B, C, D, theta):
        return A * np.power(np.sin(B * np.pi * np.power(theta, C)), D)

    @staticmethod
    def LA2(A, B, C, D, theta):
        return A * np.power(np.sin(B * np.power(theta, C)), D)

    @staticmethod
    def LA3(A, B, C, D, theta):
        return A * np.power(np.cos(B * np.power(theta, C)), D)

    def g1(self, X):
        d = self.n_var
        n = d - self.n_obj

        z = np.power(X[:, self.n_obj - 1:], n)
        i = np.arange(self.n_obj - 1, d)

        exp = 1 - np.exp(-10.0 * (z - 0.5 - i / (2 * d)) * (z - 0.5 - i / (2 * d)))
        distance = 1 + exp.sum(axis=1, keepdims=True)
        return distance

    def g2(self, X):
        d = self.n_var
        n = d

        i = np.arange(self.n_obj - 1, d)
        z = 1 - np.exp(-10.0 * (X[:, self.n_obj - 1:] - i / n) * (X[:, self.n_obj - 1:] - i / n))
        contrib = (0.1 / n) * z * z + 1.5 - 1.5 * np.cos(2 * np.pi * z)
        distance = 1 + contrib.sum(axis=1, keepdims=True)
        return distance

    def g3(self, X):
        contrib = 2.0 * np.power(
            X[:, self.n_obj - 1:] + (X[:, self.n_obj - 2:-1] - 0.5) * (X[:, self.n_obj - 2:-1] - 0.5) - 1.0, 2.0)
        distance = 1 + contrib.sum(axis=1, keepdims=True)
        return distance


class MW1(MW):
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('Number of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('Number of variables must be greater than two.')

        super(MW1, self).__init__(prob_id=1, n_var=n_var, n_obj=n_obj, scale_var=scale_var, scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        g = self.g1(X)
        f0 = X[:, [0]]
        f1 = g * (1 - 0.85 * f0 / g)
        F = np.column_stack([f0, f1])
        C = f0 + f1 - 1 - self.LA1(0.5, 2.0, 1.0, 8.0, np.sqrt(2.0) * f1 - np.sqrt(2.0) * f0)
        return np.column_stack([F, C])

    def _calc_pareto_front(self):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("MW", fname))


class MW2(MW):
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('Number of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('Number of variables must be greater than two.')

        super(MW2, self).__init__(prob_id=2, n_var=n_var, n_obj=n_obj, scale_var=scale_var, scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        g = self.g2(X)
        f0 = X[:, [0]]
        f1 = g * (1 - f0 / g)
        F = np.column_stack([f0, f1])
        C = f0 + f1 - 1 - self.LA1(0.5, 3.0, 1.0, 8.0, np.sqrt(2.0) * f1 - np.sqrt(2.0) * f0)
        return np.column_stack([F, C])

    def _calc_pareto_front(self):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("MW", fname))


class MW3(MW):
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('Number of objectives must equal two.')

        super(MW3, self).__init__(prob_id=3, n_var=n_var, n_obj=n_obj, scale_var=scale_var, scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        g = self.g3(X)
        f0 = X[:, [0]]
        f1 = g * (1 - f0 / g)
        g0 = f0 + f1 - 1.05 - self.LA1(0.45, 0.75, 1.0, 6.0, np.sqrt(2.0) * f1 - np.sqrt(2.0) * f0)
        g1 = 0.85 - f0 - f1 + self.LA1(0.3, 0.75, 1.0, 2.0, np.sqrt(2.0) * f1 - np.sqrt(2.0) * f0)
        F = np.column_stack([f0, f1])
        C = np.column_stack([g0, g1])
        return np.column_stack([F, C])

    def _calc_pareto_front(self):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("MW", fname))


class MW4(MW):
    def __init__(self, n_var=10, n_obj=3, scale_var=False, scale_obj=False):
        if not n_var > 2:
            raise ValueError('Number of variables must be greater than two.')

        super(MW4, self).__init__(prob_id=4, n_var=n_var, n_obj=n_obj, scale_var=scale_var, scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        g = self.g1(X)
        F = g * np.ones((X.shape[0], self.n_obj))
        F[:, 1:] *= X[:, (self.n_obj - 2)::-1]
        F[:, 0:-1] *= np.flip(np.cumprod(1 - X[:, :(self.n_obj - 1)], axis=1), axis=1)

        C = F.sum(axis=1) - 1 - self.LA1(0.4, 2.5, 1.0, 8.0, F[:, -1] - F[:, :-1].sum(axis=1))
        C.reshape((-1, 1))
        return np.column_stack([F, C])

    def _calc_pareto_front(self):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("MW", fname))


class MW5(MW):
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('Number of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('Number of variables must be greater than two.')

        super(MW5, self).__init__(prob_id=5, n_var=n_var, n_obj=n_obj, scale_var=scale_var, scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        g = self.g1(X)
        f0 = g * X[:, [0]]
        f1 = g * np.sqrt(1.0 - np.power(f0 / g, 2.0) + 1e-6)

        with np.errstate(divide='ignore'):
            atan = np.arctan(f1 / f0)

        g0 = f0 ** 2 + f1 ** 2 - np.power(1.7 - self.LA2(0.2, 2.0, 1.0, 1.0, atan), 2.0)
        t = 0.5 * np.pi - 2 * np.abs(atan - 0.25 * np.pi)
        g1 = np.power(1 + self.LA2(0.5, 6.0, 3.0, 1.0, t), 2.0) - f0 ** 2 - f1 ** 2
        g2 = np.power(1 - self.LA2(0.45, 6.0, 3.0, 1.0, t), 2.0) - f0 ** 2 - f1 ** 2
        F = np.column_stack([f0, f1])
        C = np.column_stack([g0, g1, g2])
        return np.column_stack([F, C])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("MW", fname))


class MW6(MW):
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('Number of objectives must equal two.')

        super(MW6, self).__init__(prob_id=6, n_var=n_var, n_obj=n_obj, scale_var=scale_var, scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        g = self.g2(X)
        f0 = g * X[:, [0]]
        f1 = g * np.sqrt(np.power(1.1, 2.0) - np.power(X[:, [0]], 2.0) + 1e-6)

        with np.errstate(divide='ignore'):
            atan = np.arctan(f1 / f0)

        C = f0 ** 2 / np.power(1.0 + self.LA3(0.15, 6.0, 4.0, 10.0, atan), 2.0) + f1 ** 2 / np.power(
            1.0 + self.LA3(0.75, 6.0, 4.0, 10.0, atan), 2.0) - 1
        F = np.column_stack([f0, f1])
        return np.column_stack([F, C])

    def _calc_pareto_front(self):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("MW", fname))


class MW7(MW):
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('Number of objectives must equal two.')

        super(MW7, self).__init__(prob_id=7, n_var=n_var, n_obj=n_obj, scale_var=scale_var, scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        g = self.g3(X)
        f0 = g * X[:, [0]]
        f1 = g * np.sqrt(1 - np.power(f0 / g, 2) + 1e-6)

        with np.errstate(divide='ignore'):
            atan = np.arctan(f1 / f0)

        g0 = f0 ** 2 + f1 ** 2 - np.power(1.2 + np.abs(self.LA2(0.4, 4.0, 1.0, 16.0, atan)), 2.0)
        g1 = np.power(1.15 - self.LA2(0.2, 4.0, 1.0, 8.0, atan), 2.0) - f0 ** 2 - f1 ** 2
        F = np.column_stack([f0, f1])
        C = np.column_stack([g0, g1])
        return np.column_stack([F, C])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("MW", fname))


class MW8(MW):
    def __init__(self, n_var=10, n_obj=3, scale_var=False, scale_obj=False):
        super(MW8, self).__init__(prob_id=8, n_var=n_var, n_obj=n_obj, scale_var=scale_var, scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        g = self.g2(X)
        f = g.reshape((-1, 1)) * np.ones((X.shape[0], self.n_obj))
        f[:, 1:] *= np.sin(0.5 * np.pi * X[:, (self.n_obj - 2)::-1])
        cos = np.cos(0.5 * np.pi * X[:, :(self.n_obj - 1)])
        f[:, 0:-1] *= np.flip(np.cumprod(cos, axis=1), axis=1)

        f_squared = (f ** 2).sum(axis=1)
        g0 = f_squared - (1.25 - self.LA2(0.5, 6.0, 1.0, 2.0, np.arcsin(f[:, -1] / np.sqrt(f_squared)))) * (
                1.25 - self.LA2(0.5, 6.0, 1.0, 2.0, np.arcsin(f[:, -1] / np.sqrt(f_squared))))
        C = g0.reshape((-1, 1))
        return np.column_stack([f, C])

    def _calc_pareto_front(self):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("MW", fname))


class MW9(MW):
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('Number of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('Number of variables must be greater than two.')

        super(MW9, self).__init__(prob_id=9, n_var=n_var, n_obj=n_obj, scale_var=scale_var, scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        g = self.g1(X)
        f0 = g * X[:, [0]]
        f1 = g * (1.0 - np.power(f0 / g, 0.6))
        t1 = (1 - 0.64 * f0 * f0 - f1) * (1 - 0.36 * f0 * f0 - f1)
        t2 = (1.35 * 1.35 - (f0 + 0.35) * (f0 + 0.35) - f1) * (1.15 * 1.15 - (f0 + 0.15) * (f0 + 0.15) - f1)
        C = np.minimum(t1, t2)
        F = np.column_stack([f0, f1])
        return np.column_stack([F, C])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("MW", fname))


class MW10(MW):
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('Number of objectives must equal two.')

        super(MW10, self).__init__(prob_id=10, n_var=n_var, n_obj=n_obj, scale_var=scale_var, scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        g = self.g2(X)
        f0 = g * np.power(X[:, [0]], self.n_var)
        f1 = g * (1.0 - np.power(f0 / g, 2.0))

        g0 = -1.0 * (2.0 - 4.0 * f0 * f0 - f1) * (2.0 - 8.0 * f0 * f0 - f1)
        g1 = (2.0 - 2.0 * f0 * f0 - f1) * (2.0 - 16.0 * f0 * f0 - f1)
        g2 = (1.0 - f0 * f0 - f1) * (1.2 - 1.2 * f0 * f0 - f1)
        F = np.column_stack([f0, f1])
        G = np.column_stack([g0, g1, g2])
        return np.column_stack([F, G])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("MW", fname))


class MW11(MW):
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('Number of objectives must equal two.')

        super(MW11, self).__init__(prob_id=11, n_var=n_var, n_obj=n_obj, scale_var=scale_var, scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        g = self.g3(X)
        f0 = g * X[:, [0]]
        f1 = g * np.sqrt(np.power(np.sqrt(2.0), 2.0) - np.power(X[:, [0]], 2.0) + 1e-6)

        g0 = -1.0 * (3.0 - f0 * f0 - f1) * (3.0 - 2.0 * f0 * f0 - f1)
        g1 = (3.0 - 0.625 * f0 * f0 - f1) * (3.0 - 7.0 * f0 * f0 - f1)
        g2 = -1.0 * (1.62 - 0.18 * f0 * f0 - f1) * (1.125 - 0.125 * f0 * f0 - f1)
        g3 = (2.07 - 0.23 * f0 * f0 - f1) * (0.63 - 0.07 * f0 * f0 - f1)
        F = np.column_stack([f0, f1])
        C = np.column_stack([g0, g1, g2, g3])
        return np.column_stack([F, C])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("MW", fname))


class MW12(MW):
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('Number of objectives must equal two.')
        if not n_var > 2:
            raise ValueError('Number of variables must be greater than two.')

        super(MW12, self).__init__(prob_id=12, n_var=n_var, n_obj=n_obj, scale_var=scale_var, scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        g = self.g1(X)
        f0 = g * X[:, [0]]
        f1 = g * (0.85 - 0.8 * (f0 / g) - 0.08 * np.abs(np.sin(3.2 * np.pi * (f0 / g))))

        g0 = -1.0 * (1 - 0.625 * f0 - f1 + 0.08 * np.sin(2 * np.pi * (f1 - f0 / 1.6))) * (
                1.4 - 0.875 * f0 - f1 + 0.08 * np.sin(2 * np.pi * (f1 / 1.4 - f0 / 1.6)))
        g1 = (1 - 0.8 * f0 - f1 + 0.08 * np.sin(2 * np.pi * (f1 - f0 / 1.5))) * (
                1.8 - 1.125 * f0 - f1 + 0.08 * np.sin(2 * np.pi * (f1 / 1.8 - f0 / 1.6)))
        F = np.column_stack([f0, f1])
        C = np.column_stack([g0, g1])
        return np.column_stack([F, C])

    def _calc_pareto_front(self):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("MW", fname))


class MW13(MW):
    def __init__(self, n_obj=2, n_var=10, scale_var=False, scale_obj=False):
        if n_obj != 2:
            raise ValueError('Number of objectives must equal two.')

        super(MW13, self).__init__(prob_id=13, n_var=n_var, n_obj=n_obj, scale_var=scale_var, scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        g = self.g2(X)
        f0 = g * X[:, [0]]
        f1 = g * (5.0 - np.exp(f0 / g) - np.abs(0.5 * np.sin(3 * np.pi * f0 / g)))

        g0 = -1.0 * (5.0 - (1 + f0 + 0.5 * f0 * f0) - 0.5 * np.sin(3 * np.pi * f0) - f1) * (
                5.0 - (1 + 0.7 * f0) - 0.5 * np.sin(3 * np.pi * f0) - f1)
        g1 = (5.0 - np.exp(f0) - 0.5 * np.sin(3 * np.pi * f0) - f1) * (
                5.0 - (1 + 0.4 * f0) - 0.5 * np.sin(3 * np.pi * f0) - f1)
        F = np.column_stack([f0, f1])
        C = np.column_stack([g0, g1])
        return np.column_stack([F, C])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("MW", fname))


class MW14(MW):
    def __init__(self, n_var=10, n_obj=3, scale_var=False, scale_obj=False):
        super(MW14, self).__init__(prob_id=14, n_var=n_var, n_obj=n_obj, scale_var=scale_var, scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        g = self.g3(X)
        f = np.zeros((X.shape[0], self.n_obj))
        f[:, :-1] = X[:, :(self.n_obj - 1)]
        LA1 = self.LA1(1.5, 1.1, 2.0, 1.0, f[:, :-1])
        inter = (6 - np.exp(f[:, :-1]) - LA1).sum(axis=1, keepdims=True)
        f[:, [-1]] = g / (self.n_obj - 1) * inter

        alpha = 6.1 - 1 - f[:, :-1] - 0.5 * f[:, :-1] * f[:, :-1] - LA1
        C = f[:, [-1]] - 1 / (self.n_obj - 1) * alpha.sum(axis=1, keepdims=True)
        return np.column_stack([f, C])

    def _calc_pareto_front(self):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("MW", fname))

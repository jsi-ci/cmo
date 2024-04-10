import numpy as np
import os
from cmo.problems.utils import CMOP, load_pareto_front_from_file


__all__ = ['C1DTLZ1', 'C1DTLZ3', 'C2DTLZ2', 'ConvexC2DTLZ2', 'C3DTLZ1', 'C3DTLZ4', 'CDTLZ']


class CDTLZ(CMOP):

    _names = {
        1: "C1-DTLZ1",
        2: "C1-DTLZ3",
        3: "C2-DTLZ2",
        4: "Convex-C2-DTLZ2",
        5: "C3-DTLZ1",
        6: "C3-DTLZ4"
    }

    _nadir_points = {
        **dict.fromkeys([1], 0.5),
        **dict.fromkeys([2, 3, 4, 5], 1),
        **dict.fromkeys([6], 2)
    }

    def __init__(self, prob_id, n_var=10, n_obj=3, scale_var=False, scale_obj=False):

        if not n_var >= n_obj:
            raise ValueError('Number of variables must be greater than or equal to the number of objectives.')

        if prob_id not in range(1, 7):
            raise ValueError("Please select a valid prob id.")

        # Set dim
        n_iq_constr = 1 if prob_id in [1, 2, 3, 4] else n_obj

        # Set xl, xu
        xl = 0
        xu = 1

        # Set name
        name = self._names[prob_id]

        # Set k
        self.k = n_var - n_obj + 1

        super(CDTLZ, self).__init__(n_var=n_var,
                                    n_obj=n_obj,
                                    n_iq_constr=n_iq_constr,
                                    n_eq_constr=0,
                                    xl=xl,
                                    xu=xu,
                                    scale_var=scale_var,
                                    scale_obj=scale_obj,
                                    nadir_point=None,
                                    ideal_point=None,
                                    name=name)

    def g1(self, X_M):
        return 100 * (self.k + np.sum(np.square(X_M - 0.5) - np.cos(20 * np.pi * (X_M - 0.5)), axis=1, keepdims=True))

    def g2(self, X_M):
        return np.sum(np.square(X_M - 0.5), axis=1, keepdims=True)

    def obj_func(self, X_, g, alpha=1):
        f = []

        for i in range(0, self.n_obj):
            _f = (1 + g)
            _f *= np.prod(np.cos(np.power(X_[:, :X_.shape[1] - i], alpha) * np.pi / 2.0), axis=1, keepdims=True)
            if i > 0:
                _f *= np.sin(np.power(X_[:, [X_.shape[1] - i]], alpha) * np.pi / 2.0)

            f.append(_f)

        f = np.column_stack(f)
        return f

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("CDTLZ", fname))


class C1DTLZ1(CDTLZ):
    def __init__(self, n_var=10, n_obj=3, scale_var=False, scale_obj=False):
        super(C1DTLZ1, self).__init__(prob_id=1, n_var=n_var, n_obj=n_obj, scale_var=scale_var, scale_obj=scale_obj)

    def obj_func(self, X_, g, alpha=1):
        f = []

        for i in range(0, self.n_obj):
            _f = 0.5 * (1 + g)
            _f *= np.prod(X_[:, :X_.shape[1] - i], axis=1, keepdims=True)
            if i > 0:
                _f *= 1 - X_[:, [X_.shape[1] - i]]
            f.append(_f)

        return np.column_stack(f)

    def _fn(self, X, *args, **kwargs):
        X_, X_M = X[:, :self.n_obj - 1], X[:, self.n_obj - 1:]
        g = self.g1(X_M)
        F = self.obj_func(X_, g)
        c = 1 - F[:, [-1]] / 0.6 - np.sum(F[:, :-1] / 0.5, axis=1, keepdims=True)
        return np.column_stack([F, -c])


class C1DTLZ3(CDTLZ):

    _r_dict = {
        **dict.fromkeys([2], 6),
        **dict.fromkeys([3, 4], 9),
        **dict.fromkeys([5, 6, 7, 8, 9], 12.5),
    }

    def __init__(self, n_var=10, n_obj=3, scale_var=False, scale_obj=False):
        self.r = self._r_dict[n_obj] if n_obj < 10 else 15
        super(C1DTLZ3, self).__init__(prob_id=2, n_var=n_var, n_obj=n_obj, scale_var=scale_var, scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        X_, X_M = X[:, :self.n_obj - 1], X[:, self.n_obj - 1:]
        g = self.g1(X_M)
        F = self.obj_func(X_, g, alpha=1)
        h = np.sum(F * F, axis=1, keepdims=True)
        c = (h - 16) * (h - self.r ** 2)
        return np.column_stack([F, -c])


class C2DTLZ2(CDTLZ):

    _r_dict = {2: 0.1, 3: 0.4}

    def __init__(self, n_var=10, n_obj=3, scale_var=False, scale_obj=False):
        self.r = self._r_dict[n_obj] if n_obj < 4 else 0.5
        super(C2DTLZ2, self).__init__(prob_id=3, n_var=n_var, n_obj=n_obj, scale_var=scale_var, scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        X_, X_M = X[:, :self.n_obj - 1], X[:, self.n_obj - 1:]
        g = self.g2(X_M)
        F = self.obj_func(X_, g, alpha=1)
        h1 = np.sum(F ** 2, axis=1, keepdims=True) - self.r ** 2 + 1 - 2 * F
        h1 = np.min(h1, axis=1, keepdims=True)
        h2 = np.sum((F - 1 / np.sqrt(self.n_obj)) ** 2, axis=1, keepdims=True) - self.r ** 2
        c = np.min(np.column_stack([h1, h2]), axis=1, keepdims=True)
        return np.column_stack([F, c])


class ConvexC2DTLZ2(CDTLZ):

    _r_dict = {
        **dict.fromkeys([2], 0.1),
        **dict.fromkeys([3, 4, 5], 0.225),
        **dict.fromkeys([6, 7, 8, 9, 10, 11, 12, 13, 14], 0.26)
    }

    def __init__(self, n_var=10, n_obj=3, scale_var=False, scale_obj=False):
        self.r = self._r_dict[n_obj] if n_obj < 15 else 0.27
        super(ConvexC2DTLZ2, self).__init__(prob_id=4, n_var=n_var, n_obj=n_obj, scale_var=scale_var,
                                            scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        X_, X_M = X[:, :self.n_obj - 1], X[:, self.n_obj - 1:]
        g = self.g2(X_M)
        F = self.obj_func(X_, g, alpha=1)
        F[:, :-1], F[:, [-1]] = F[:, :-1] ** 4, F[:, [-1]] ** 2
        h1 = np.mean(F, axis=1, keepdims=True)
        h2 = np.sum((F - h1) ** 2, axis=1, keepdims=True)
        c = h2 - self.r ** 2
        return np.column_stack([F, -c])


class C3DTLZ1(CDTLZ):
    def __init__(self, n_var=10, n_obj=3, scale_var=False, scale_obj=False):
        super(C3DTLZ1, self).__init__(prob_id=5, n_var=n_var, n_obj=n_obj, scale_var=scale_var, scale_obj=scale_obj)

    def obj_func(self, X_, g, alpha=1):
        f = []

        for i in range(0, self.n_obj):
            _f = 0.5 * (1 + g)
            _f *= np.prod(X_[:, :X_.shape[1] - i], axis=1, keepdims=True)
            if i > 0:
                _f *= 1 - X_[:, [X_.shape[1] - i]]
            f.append(_f)

        return np.column_stack(f)

    def _fn(self, X, *args, **kwargs):
        X_, X_M = X[:, :self.n_obj - 1], X[:, self.n_obj - 1:]
        g = self.g1(X_M)
        F = self.obj_func(X_, g)
        C = np.sum(F, axis=1, keepdims=True) + F - 1
        return np.column_stack([F, -C])


class C3DTLZ4(CDTLZ):
    def __init__(self, n_var=10, n_obj=3, scale_var=False, scale_obj=False, alpha=100):
        self.alpha = alpha

        if not n_var > 2:
            raise ValueError('Number of variables must be greater than two.')

        super(C3DTLZ4, self).__init__(prob_id=6, n_var=n_var, n_obj=n_obj, scale_var=scale_var, scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        X_, X_M = X[:, :self.n_obj - 1], X[:, self.n_obj - 1:]
        g = self.g2(X_M)
        F = self.obj_func(X_, g, alpha=self.alpha)
        C = np.sum(F ** 2, axis=1, keepdims=True) - 3 / 4 * F ** 2 - 1
        return np.column_stack([F, -C])
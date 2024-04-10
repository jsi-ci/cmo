import numpy as np
import os
from cmop.cdtlz import CDTLZ
from cmop.utils import load_pareto_front_from_file


__all__ = ['DC1DTLZ1', 'DC1DTLZ3', 'DC2DTLZ1', 'DC2DTLZ3', 'DC3DTLZ1', 'DC3DTLZ3']


class DCDTLZ(CDTLZ):

    _names = {
        1: "DC1-DTLZ1",
        2: "DC1-DTLZ3",
        3: "DC2-DTLZ1",
        4: "DC2-DTLZ3",
        5: "DC3-DTLZ1",
        6: "DC3-DTLZ3"
    }

    _nadir_points = {
        **dict.fromkeys([1, 3, 5], 0.5),
        **dict.fromkeys([2, 4, 6], 1)
    }

    _n_iq_constrs = {
        **dict.fromkeys([1, 2], 1),
        **dict.fromkeys([3, 4], 2)
    }

    def __init__(self, prob_id, n_var=10, n_obj=3, scale_var=False, scale_obj=False, a=3, b=0.5):

        if prob_id not in range(1, 7):
            raise ValueError("Please select a valid prob id.")

        # Set dim
        n_iq_constr = self._n_iq_constrs[prob_id] if prob_id < 5 else (n_obj + 1)

        # Set xl, xu
        xl = 0
        xu = 1

        # Set nadir and ideal points
        ideal_point = None
        nadir_point = None

        # Set name
        name = self._names[prob_id]

        # Set k, a, b
        self.k = n_var - n_obj + 1
        self.a = a
        self.b = b

        super(CDTLZ, self).__init__(n_var=n_var,
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

    def constraint_dc1(self, X):
        G = self.b - np.cos(self.a * np.pi * X[:, [0]])
        return G

    def constraints_dc2(self, gx):
        G = np.column_stack([
            self.b - np.cos(gx * np.pi * self.a),
            self.b - np.exp(-gx)
        ])
        return G

    def constraints_dc3(self, X, gx):
        Ggx = self.b - np.cos(self.a * np.pi * gx)
        Gx = self.b - np.cos(self.a * np.pi * X)
        return np.column_stack([Ggx, Gx])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("DCDTLZ", fname))


class DC1DTLZ1(DCDTLZ):
    def __init__(self, n_var=10, n_obj=3, scale_var=False, scale_obj=False, a=3, b=0.5):
        super(DC1DTLZ1, self).__init__(prob_id=1, n_var=n_var, n_obj=n_obj, scale_var=scale_var,
                                       scale_obj=scale_obj, a=a, b=b)

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
        c = self.constraint_dc1(X)
        return np.column_stack([F, c])


class DC1DTLZ3(DCDTLZ):
    def __init__(self, n_var=10, n_obj=3, scale_var=False, scale_obj=False, a=3, b=0.5):
        super(DC1DTLZ3, self).__init__(prob_id=2, n_var=n_var, n_obj=n_obj, scale_var=scale_var,
                                       scale_obj=scale_obj, a=a, b=b)

    def _fn(self, X, *args, **kwargs):
        X_, X_M = X[:, :self.n_obj - 1], X[:, self.n_obj - 1:]
        g = self.g1(X_M)
        F = self.obj_func(X_, g, alpha=1)
        c = self.constraint_dc1(X)
        return np.column_stack([F, c])


class DC2DTLZ1(DCDTLZ):
    def __init__(self, n_var=10, n_obj=3, scale_var=False, scale_obj=False, a=3, b=0.5):
        super(DC2DTLZ1, self).__init__(prob_id=3, n_var=n_var, n_obj=n_obj, scale_var=scale_var,
                                       scale_obj=scale_obj, a=a, b=b)

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
        C = self.constraints_dc2(g)
        return np.column_stack([F, C])


class DC2DTLZ3(DCDTLZ):
    def __init__(self, n_var=10, n_obj=3, scale_var=False, scale_obj=False, a=3, b=0.5):
        super(DC2DTLZ3, self).__init__(prob_id=4, n_var=n_var, n_obj=n_obj, scale_var=scale_var,
                                       scale_obj=scale_obj, a=a, b=b)

    def _fn(self, X, *args, **kwargs):
        X_, X_M = X[:, :self.n_obj - 1], X[:, self.n_obj - 1:]
        g = self.g1(X_M)
        F = self.obj_func(X_, g, alpha=1)
        C = self.constraints_dc2(g)
        return np.column_stack([F, C])


class DC3DTLZ1(DCDTLZ):
    def __init__(self, n_var=10, n_obj=3, scale_var=False, scale_obj=False, a=3, b=0.5):
        super(DC3DTLZ1, self).__init__(prob_id=5, n_var=n_var, n_obj=n_obj, scale_var=scale_var,
                                       scale_obj=scale_obj, a=a, b=b)

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
        X_, X1_, X_M = X[:, :self.n_obj - 1], X[:, :self.n_obj], X[:, self.n_obj - 1:]
        g = self.g1(X_M)
        F = self.obj_func(X_, g)
        C = self.constraints_dc3(X1_, g)
        return np.column_stack([F, C])


class DC3DTLZ3(DCDTLZ):
    def __init__(self, n_var=10, n_obj=3, scale_var=False, scale_obj=False, a=3, b=0.5):
        super(DC3DTLZ3, self).__init__(prob_id=6, n_var=n_var, n_obj=n_obj, scale_var=scale_var,
                                       scale_obj=scale_obj, a=a, b=b)

    def _fn(self, X, *args, **kwargs):
        X_, X1_, X_M = X[:, :self.n_obj - 1], X[:, :self.n_obj], X[:, self.n_obj - 1:]
        g = self.g1(X_M)
        F = self.obj_func(X_, g, alpha=1)
        C = self.constraints_dc3(X1_, g)
        return np.column_stack([F, C])
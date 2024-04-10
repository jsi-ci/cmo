import os
import numpy as np
from cmo.problems.utils import CMOP, load_pareto_front_from_file

__all__ = ['LIRCMOP1', 'LIRCMOP2', 'LIRCMOP3', 'LIRCMOP4', 'LIRCMOP5', 'LIRCMOP6', 'LIRCMOP7',
           'LIRCMOP8', 'LIRCMOP9', 'LIRCMOP10', 'LIRCMOP11', 'LIRCMOP12', 'LIRCMOP13', 'LIRCMOP14']


class LIRCMOP(CMOP):
    def __init__(self, n_var, n_obj, n_iq_constr, scale_var=False, scale_obj=False, **kwargs):
        name = 'LIR-CMOP{}'.format(str(self.__class__.__name__)[7:])

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_iq_constr=n_iq_constr,
                         n_eq_constr=0,
                         xl=0.,
                         xu=1.,
                         scale_var=scale_var,
                         scale_obj=scale_obj,
                         name=name,
                         **kwargs)


class LIRCMOP1(LIRCMOP):
    def __init__(self, n_var=30, scale_var=False, scale_obj=False, n_obj=2, n_iq_constr=2):
        if n_var != 30:
            raise ValueError('Number of variables must equal 30.')
        if n_obj != 2:
            raise ValueError('Number of objectives must equal 2.')

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_iq_constr=n_iq_constr,
                         scale_var=scale_var,
                         scale_obj=scale_obj)

    def g1(self, X):
        h = np.sin(0.5 * np.pi * X[:, [0]])
        g = np.sum((X[:, 2::2] - h) ** 2, axis=1, keepdims=True)
        return g

    def g2(self, X):
        h = np.cos(0.5 * np.pi * X[:, [0]])
        g = np.sum((X[:, 1::2] - h) ** 2, axis=1, keepdims=True)
        return g

    def f1(self, X, g):
        return X[:, [0]] + g

    def f2(self, X, g):
        return 1 - X[:, [0]] ** 2 + g

    def c(self, g, a=0.51, b=0.5):
        return -(a - g) * (g - b)

    def _fn(self, X, *args, **kwargs):
        g1 = self.g1(X)
        g2 = self.g2(X)
        f1 = self.f1(X, g1)
        f2 = self.f2(X, g2)
        c1 = self.c(g1)
        c2 = self.c(g2)
        return np.column_stack([f1, f2, c1, c2])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("LIRCMOP", fname))


class LIRCMOP2(LIRCMOP1):
    def __init__(self, n_var=30, scale_var=False, scale_obj=False, n_obj=2, n_iq_constr=2):
        if n_var != 30:
            raise ValueError('Number of variables must equal 30.')
        if n_obj != 2:
            raise ValueError('Number of objectives must equal 2.')

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_iq_constr=n_iq_constr,
                         scale_var=scale_var,
                         scale_obj=scale_obj)

    def f2(self, X, g):
        return 1 - np.sqrt(X[:, [0]]) + g

    def _fn(self, X, *args, **kwargs):
        g1 = self.g1(X)
        g2 = self.g2(X)
        f1 = self.f1(X, g1)
        f2 = self.f2(X, g2)
        c1 = self.c(g1)
        c2 = self.c(g2)
        return np.column_stack([f1, f2, c1, c2])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("LIRCMOP", fname))


class LIRCMOP3(LIRCMOP1):
    def __init__(self, n_var=30, scale_var=False, scale_obj=False, n_obj=2, n_iq_constr=3):
        if n_var != 30:
            raise ValueError('Number of variables must equal 30.')
        if n_obj != 2:
            raise ValueError('Number of objectives must equal 2.')

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_iq_constr=n_iq_constr,
                         scale_var=scale_var,
                         scale_obj=scale_obj)

    def c3(self, X, c=20):
        return 0.5 - np.sin(c * np.pi * X[:, [0]])

    def _fn(self, X, *args, **kwargs):
        g1 = self.g1(X)
        g2 = self.g2(X)
        f1 = self.f1(X, g1)
        f2 = self.f2(X, g2)
        c1 = self.c(g1)
        c2 = self.c(g2)
        c3 = self.c3(X)
        return np.column_stack([f1, f2, c1, c2, c3])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("LIRCMOP", fname))


class LIRCMOP4(LIRCMOP2):
    def __init__(self, n_var=30, scale_var=False, scale_obj=False, n_obj=2, n_iq_constr=3):
        if n_var != 30:
            raise ValueError('Number of variables must equal 30.')
        if n_obj != 2:
            raise ValueError('Number of objectives must equal 2.')

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_iq_constr=n_iq_constr,
                         scale_var=scale_var,
                         scale_obj=scale_obj)

    def c3(self, X, c=20):
        return 0.5 - np.sin(c * np.pi * X[:, [0]])

    def _fn(self, X, *args, **kwargs):
        g1 = self.g1(X)
        g2 = self.g2(X)
        f1 = self.f1(X, g1)
        f2 = self.f2(X, g2)
        c1 = self.c(g1)
        c2 = self.c(g2)
        c3 = self.c3(X)
        return np.column_stack([f1, f2, c1, c2, c3])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("LIRCMOP", fname))


class LIRCMOP5(LIRCMOP):
    def __init__(self, n_var=30, scale_var=False, scale_obj=False, n_obj=2, n_iq_constr=2):
        if n_var != 30:
            raise ValueError('Number of variables must equal 30.')
        if n_obj != 2:
            raise ValueError('Number of objectives must equal 2.')

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_iq_constr=n_iq_constr,
                         scale_var=scale_var,
                         scale_obj=scale_obj)

    def g1(self, X):
        i = np.arange(2, self.n_var, 2)
        h = np.sin(0.5 * np.pi * X[:, [0]] * i / self.n_var)
        g = np.sum((X[:, 2::2] - h) ** 2, axis=1, keepdims=True)
        return g

    def g2(self, X):
        i = np.arange(1, self.n_var, 2)
        h = np.cos(0.5 * np.pi * X[:, [0]] * i / self.n_var)
        g = np.sum((X[:, 1::2] - h) ** 2, axis=1, keepdims=True)
        return g

    def f1(self, X, g):
        return X[:, [0]] + 10 * g + 0.7057

    def f2(self, X, g):
        return 1 - np.sqrt(X[:, [0]]) + 10 * g + 0.7057

    def c(self, f1, f2, p, q, theta, a, b, r):
        h1 = ((f1 - p) * np.cos(theta) - (f2 - q) * np.sin(theta)) ** 2 / a ** 2
        h2 = ((f1 - p) * np.sin(theta) + (f2 - q) * np.cos(theta)) ** 2 / b ** 2
        return r - h1 - h2

    def _fn(self, X, *args, **kwargs):
        g1 = self.g1(X)
        g2 = self.g2(X)
        f1 = self.f1(X, g1)
        f2 = self.f2(X, g2)
        c1 = self.c(f1, f2, p=1.6, q=1.6, theta=-0.25 * np.pi, a=2.0, b=4.0, r=0.1)
        c2 = self.c(f1, f2, p=2.5, q=2.5, theta=-0.25 * np.pi, a=2.0, b=8.0, r=0.1)
        return np.column_stack([f1, f2, c1, c2])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("LIRCMOP", fname))


class LIRCMOP6(LIRCMOP5):
    def __init__(self, n_var=30, scale_var=False, scale_obj=False, n_obj=2, n_iq_constr=2):
        if n_var != 30:
            raise ValueError('Number of variables must equal 30.')
        if n_obj != 2:
            raise ValueError('Number of objectives must equal 2.')

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_iq_constr=n_iq_constr,
                         scale_var=scale_var,
                         scale_obj=scale_obj)

    def f2(self, X, g):
        return 1 - X[:, [0]] ** 2 + 10 * g + 0.7057

    def _fn(self, X, *args, **kwargs):
        g1 = self.g1(X)
        g2 = self.g2(X)
        f1 = self.f1(X, g1)
        f2 = self.f2(X, g2)
        c1 = self.c(f1, f2, p=1.8, q=1.8, theta=-0.25 * np.pi, a=2.0, b=8.0, r=0.1)
        c2 = self.c(f1, f2, p=2.8, q=2.8, theta=-0.25 * np.pi, a=2.0, b=8.0, r=0.1)
        return np.column_stack([f1, f2, c1, c2])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("LIRCMOP", fname))


class LIRCMOP7(LIRCMOP5):
    def __init__(self, n_var=30, scale_var=False, scale_obj=False, n_obj=2, n_iq_constr=3):
        if n_var != 30:
            raise ValueError('Number of variables must equal 30.')
        if n_obj != 2:
            raise ValueError('Number of objectives must equal 2.')

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_iq_constr=n_iq_constr,
                         scale_var=scale_var,
                         scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        g1 = self.g1(X)
        g2 = self.g2(X)
        f1 = self.f1(X, g1)
        f2 = self.f2(X, g2)
        c1 = self.c(f1, f2, p=1.20, q=1.20, theta=-0.25 * np.pi, a=2.0, b=6.00, r=0.1)
        c2 = self.c(f1, f2, p=2.25, q=2.25, theta=-0.25 * np.pi, a=2.5, b=12.0, r=0.1)
        c3 = self.c(f1, f2, p=3.50, q=3.50, theta=-0.25 * np.pi, a=2.5, b=10.0, r=0.1)
        return np.column_stack([f1, f2, c1, c2, c3])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("LIRCMOP", fname))


class LIRCMOP8(LIRCMOP6):
    def __init__(self, n_var=30, scale_var=False, scale_obj=False, n_obj=2, n_iq_constr=3):
        if n_var != 30:
            raise ValueError('Number of variables must equal 30.')
        if n_obj != 2:
            raise ValueError('Number of objectives must equal 2.')

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_iq_constr=n_iq_constr,
                         scale_var=scale_var,
                         scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        g1 = self.g1(X)
        g2 = self.g2(X)
        f1 = self.f1(X, g1)
        f2 = self.f2(X, g2)
        c1 = self.c(f1, f2, p=1.20, q=1.20, theta=-0.25 * np.pi, a=2.0, b=6.00, r=0.1)
        c2 = self.c(f1, f2, p=2.25, q=2.25, theta=-0.25 * np.pi, a=2.5, b=12.0, r=0.1)
        c3 = self.c(f1, f2, p=3.50, q=3.50, theta=-0.25 * np.pi, a=2.5, b=10.0, r=0.1)
        return np.column_stack([f1, f2, c1, c2, c3])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("LIRCMOP", fname))


class LIRCMOP9(LIRCMOP5):
    def __init__(self, n_var=30, scale_var=False, scale_obj=False, n_obj=2, n_iq_constr=2):
        if n_var != 30:
            raise ValueError('Number of variables must equal 30.')
        if n_obj != 2:
            raise ValueError('Number of objectives must equal 2.')

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_iq_constr=n_iq_constr,
                         scale_var=scale_var,
                         scale_obj=scale_obj)

    def f1(self, X, g):
        return 1.7057 * X[:, [0]] * (10 * g + 1)

    def f2(self, X, g):
        return 1.7057 * (1 - X[:, [0]] ** 2) * (10 * g + 1)

    def c1(self, f1, f2, p, q, theta, a, b, r):
        return self.c(f1, f2, p, q, theta, a, b, r)

    def c2(self, f1, f2, alpha, c):
        h1 = f1 * np.sin(alpha) + f2 * np.cos(alpha)
        h2 = np.sin(4 * np.pi * (f1 * np.cos(alpha) - f2 * np.sin(alpha)))
        return c - h1 + h2

    def _fn(self, X, *args, **kwargs):
        g1 = self.g1(X)
        g2 = self.g2(X)
        f1 = self.f1(X, g1)
        f2 = self.f2(X, g2)
        c1 = self.c1(f1, f2, p=1.4, q=1.4, theta=-0.25 * np.pi, a=1.5, b=6.0, r=0.1)
        c2 = self.c2(f1, f2, alpha=0.25 * np.pi, c=2.0)
        return np.column_stack([f1, f2, c1, c2])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("LIRCMOP", fname))


class LIRCMOP10(LIRCMOP9):
    def __init__(self, n_var=30, scale_var=False, scale_obj=False, n_obj=2, n_iq_constr=2):
        if n_var != 30:
            raise ValueError('Number of variables must equal 30.')
        if n_obj != 2:
            raise ValueError('Number of objectives must equal 2.')

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_iq_constr=n_iq_constr,
                         scale_var=scale_var,
                         scale_obj=scale_obj)

    def f2(self, X, g):
        return 1.7057 * (1 - np.sqrt(X[:, [0]])) * (10 * g + 1)

    def _fn(self, X, *args, **kwargs):
        g1 = self.g1(X)
        g2 = self.g2(X)
        f1 = self.f1(X, g1)
        f2 = self.f2(X, g2)
        c1 = self.c1(f1, f2, p=1.1, q=1.2, theta=-0.25 * np.pi, a=2.0, b=4.0, r=0.1)
        c2 = self.c2(f1, f2, alpha=0.25 * np.pi, c=1.0)
        return np.column_stack([f1, f2, c1, c2])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("LIRCMOP", fname))


class LIRCMOP11(LIRCMOP10):
    def __init__(self, n_var=30, scale_var=False, scale_obj=False, n_obj=2, n_iq_constr=2):
        if n_var != 30:
            raise ValueError('Number of variables must equal 30.')
        if n_obj != 2:
            raise ValueError('Number of objectives must equal 2.')

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_iq_constr=n_iq_constr,
                         scale_var=scale_var,
                         scale_obj=scale_obj)

    def f2(self, X, g):
        return 1.7057 * (1 - np.sqrt(X[:, [0]])) * (10 * g + 1)

    def _fn(self, X, *args, **kwargs):
        g1 = self.g1(X)
        g2 = self.g2(X)
        f1 = self.f1(X, g1)
        f2 = self.f2(X, g2)
        c1 = self.c1(f1, f2, p=1.2, q=1.2, theta=-0.25 * np.pi, a=1.5, b=5.0, r=0.1)
        c2 = self.c2(f1, f2, alpha=0.25 * np.pi, c=2.1)
        return np.column_stack([f1, f2, c1, c2])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("LIRCMOP", fname))


class LIRCMOP12(LIRCMOP9):
    def __init__(self, n_var=30, scale_var=False, scale_obj=False, n_obj=2, n_iq_constr=2):
        if n_var != 30:
            raise ValueError('Number of variables must equal 30.')
        if n_obj != 2:
            raise ValueError('Number of objectives must equal 2.')

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_iq_constr=n_iq_constr,
                         scale_var=scale_var,
                         scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        g1 = self.g1(X)
        g2 = self.g2(X)
        f1 = self.f1(X, g1)
        f2 = self.f2(X, g2)
        c1 = self.c1(f1, f2, p=1.6, q=1.6, theta=-0.25 * np.pi, a=1.5, b=6.0, r=0.1)
        c2 = self.c2(f1, f2, alpha=0.25 * np.pi, c=2.5)
        return np.column_stack([f1, f2, c1, c2])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("LIRCMOP", fname))


class LIRCMOP13(LIRCMOP):
    def __init__(self, n_var=30, scale_var=False, scale_obj=False, n_obj=3, n_iq_constr=2):
        if n_var != 30:
            raise ValueError('Number of variables must equal 30.')
        if n_obj != 3:
            raise ValueError('Number of objectives must equal 3.')

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_iq_constr=n_iq_constr,
                         scale_var=scale_var,
                         scale_obj=scale_obj)

    def g1(self, X):
        return np.sum(10 * (X[:, 2:] - 0.5) ** 2, axis=1, keepdims=True)

    def f1(self, X, g):
        return (1.7057 + g) * np.cos(0.5 * np.pi * X[:, [0]]) * np.cos(0.5 * np.pi * X[:, [1]])

    def f2(self, X, g):
        return (1.7057 + g) * np.cos(0.5 * np.pi * X[:, [0]]) * np.sin(0.5 * np.pi * X[:, [1]])

    def f3(self, X, g):
        return (1.7057 + g) * np.sin(0.5 * np.pi * X[:, [0]])

    def c(self, f1, f2, f3, a, b):
        h = f1 ** 2 + f2 ** 2 + f3 ** 2
        return -(h - a) * (h - b)

    def _fn(self, X, *args, **kwargs):
        g1 = self.g1(X)
        f1 = self.f1(X, g1)
        f2 = self.f2(X, g1)
        f3 = self.f3(X, g1)
        c1 = self.c(f1, f2, f3, a=9.00, b=4.00)
        c2 = self.c(f1, f2, f3, a=3.61, b=3.24)
        return np.column_stack([f1, f2, f3, c1, c2])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("LIRCMOP", fname))


class LIRCMOP14(LIRCMOP13):
    def __init__(self, n_var=30, scale_var=False, scale_obj=False, n_obj=3, n_iq_constr=3):
        if n_var != 30:
            raise ValueError('Number of variables must equal 30.')
        if n_obj != 3:
            raise ValueError('Number of objectives must equal 3.')

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_iq_constr=n_iq_constr,
                         scale_var=scale_var,
                         scale_obj=scale_obj)

    def _fn(self, X, *args, **kwargs):
        g1 = self.g1(X)
        f1 = self.f1(X, g1)
        f2 = self.f2(X, g1)
        f3 = self.f3(X, g1)
        c1 = self.c(f1, f2, f3, a=9.00, b=4.00)
        c2 = self.c(f1, f2, f3, a=3.61, b=3.24)
        c3 = self.c(f1, f2, f3, a=3.0625, b=2.56)
        return np.column_stack([f1, f2, f3, c1, c2, c3])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("LIRCMOP", fname))

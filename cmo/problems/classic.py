import os

from cmop.utils import CMOP, load_pareto_front_from_file
import numpy as np


__all__ = ['BNH', 'TNK', 'SRN', 'OSY', 'WB']


class BNH(CMOP):

    def __init__(self, n_var=2, n_obj=2, scale_var=False, scale_obj=False):
        xl = np.zeros(2)
        xu = np.array([5.0, 3.0])
        if n_var != 2 or n_obj != 2:
            raise ValueError('Incorrect number of objectives and/or variables.')

        super(BNH, self).__init__(n_var=n_var, n_obj=n_obj, n_iq_constr=2, n_eq_constr=0, xl=xl, xu=xu,
                                  scale_var=scale_var, scale_obj=scale_obj, name="BNH")

    def _fn(self, X, *args, **kwargs):
        X1, X2 = X[:, [0]], X[:, [1]]
        F1 = 4 * X1 ** 2 + 4 * X2 ** 2
        F2 = (X1 - 5) ** 2 + (X2 - 5) ** 2
        C1 = (X1 - 5) ** 2 + X2 ** 2 - 25
        C2 = (X1 - 8) ** 2 + (X2 + 3) ** 2 - 7.7
        return np.column_stack([F1, F2, C1, -C2])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("Classic", fname))


class TNK(CMOP):

    def __init__(self, n_var=2, n_obj=2, scale_var=False, scale_obj=False):
        xl = np.array([1e-15, 1e-15])
        xu = np.array([np.pi, np.pi])
        if n_var != 2 or n_obj != 2:
            raise ValueError('Incorrect number of objectives and/or variables.')

        super(TNK, self).__init__(n_var=n_var, n_obj=n_obj, n_iq_constr=2, n_eq_constr=0, xl=xl, xu=xu,
                                  scale_var=scale_var, scale_obj=scale_obj, name="TNK")

    def _fn(self, X, *args, **kwargs):
        X1, X2 = X[:, [0]], X[:, [1]]
        C1 = X1 ** 2 + X2 ** 2 - 1 - 0.1 * np.cos(16.0 * np.arctan(X1 / X2))
        C2 = (X1 - 0.5) ** 2 + (X2 - 0.5) ** 2 - 0.5
        return np.column_stack([X1, X2, -C1, C2])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("Classic", fname))


class SRN(CMOP):

    def __init__(self, n_var=2, n_obj=2, scale_var=False, scale_obj=False):
        xl = np.array([-20, -20])
        xu = np.array([20, 20])
        if n_var != 2 or n_obj != 2:
            raise ValueError('Incorrect number of objectives and/or variables.')

        super(SRN, self).__init__(n_var=n_var, n_obj=n_obj, n_iq_constr=2, n_eq_constr=0, xl=xl, xu=xu,
                                  scale_var=scale_var, scale_obj=scale_obj, name="SRN")

    def _fn(self, X, *args, **kwargs):
        X1, X2 = X[:, [0]], X[:, [1]]
        F1 = 2 + (X1 - 2) ** 2 + (X2 - 2) ** 2
        F2 = 9 * X1 - (X2 - 1) ** 2
        C1 = X1 ** 2 + X2 ** 2 - 225
        C2 = X1 - 3 * X2 + 10
        return np.column_stack([F1, F2, C1, C2])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("Classic", fname))


class OSY(CMOP):
    def __init__(self, n_var=6, n_obj=2):
        xl = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        xu = np.array([10.0, 10.0, 5.0, 6.0, 5.0, 10.0])
        if n_var != 6 or n_obj != 2:
            raise ValueError('Incorrect number of objectives and/or variables.')

        super().__init__(n_var=n_var, n_obj=n_obj, n_iq_constr=6, n_eq_constr=0, xl=xl, xu=xu, name="OSY")

    def _fn(self, x, *args, **kwargs):
        f1 = - (25 * (x[:, 0] - 2) ** 2 + (x[:, 1] - 2) ** 2 + (x[:, 2] - 1) ** 2 + (x[:, 3] - 4) ** 2 + (
                x[:, 4] - 1) ** 2)
        f2 = np.sum(np.square(x), axis=1)

        g1 = (x[:, 0] + x[:, 1] - 2.0) / 2.0
        g2 = (6.0 - x[:, 0] - x[:, 1]) / 6.0
        g3 = (2.0 - x[:, 1] + x[:, 0]) / 2.0
        g4 = (2.0 - x[:, 0] + 3.0 * x[:, 1]) / 2.0
        g5 = (4.0 - (x[:, 2] - 3.0) ** 2 - x[:, 3]) / 4.0
        g6 = ((x[:, 4] - 3.0) ** 2 + x[:, 5] - 4.0) / 4.0

        return np.column_stack([f1, f2, -g1, -g2, -g3, -g4, -g5, -g6])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("Classic", fname))


class WB(CMOP):
    def __init__(self, n_var=4, n_obj=2):
        xl = np.array([0.125, 0.1, 0.1, 0.125])
        xu = np.array([5.0, 10.0, 10.0, 5.0])
        if n_var != 4 or n_obj != 2:
            raise ValueError('Incorrect number of objectives and/or variables.')

        super().__init__(n_var=n_var, n_obj=n_obj, n_iq_constr=4, n_eq_constr=0, xl=xl, xu=xu, name="WB")

    def _fn(self, x, *args, **kwargs):
        f1 = 1.10471 * x[:, 0] ** 2 * x[:, 1] + 0.04811 * x[:, 2] * x[:, 3] * (14.0 + x[:, 1])
        f2 = 2.1952 / (x[:, 3] * x[:, 2] ** 3)

        P = 6000
        L = 14
        t_max = 13600
        s_max = 30000

        R = np.sqrt(0.25 * (x[:, 1] ** 2 + (x[:, 0] + x[:, 2]) ** 2))
        M = P * (L + x[:, 1] / 2)
        J = 2 * np.sqrt(0.5) * x[:, 0] * x[:, 1] * (x[:, 1] ** 2 / 12 + 0.25 * (x[:, 0] + x[:, 2]) ** 2)
        t1 = P / (np.sqrt(2) * x[:, 0] * x[:, 1])
        t2 = M * R / J
        t = np.sqrt(t1 ** 2 + t2 ** 2 + t1 * t2 * x[:, 1] / R)
        s = 6 * P * L / (x[:, 3] * x[:, 2] ** 2)
        P_c = 64746.022 * (1 - 0.0282346 * x[:, 2]) * x[:, 2] * x[:, 3] ** 3

        g1 = (1 / t_max) * (t - t_max)
        g2 = (1 / s_max) * (s - s_max)
        g3 = (1 / (5 - 0.125)) * (x[:, 0] - x[:, 3])
        g4 = (1 / P) * (P - P_c)

        return np.column_stack([f1, f2, g1, g2, g3, g4])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("Classic", fname))

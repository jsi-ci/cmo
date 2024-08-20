import os
import numpy as np
from cmo.problems.utils import CMOP, load_pareto_front_from_file

__all__ = ['DOC1', 'DOC2', 'DOC3', 'DOC4', 'DOC5', 'DOC6', 'DOC7', 'DOC8', 'DOC9']


class DOC1(CMOP):
    def __init__(self):
        self.M = 2
        self.D = 6
        self.lower = np.array([0, 78, 33, 27, 27, 27])
        self.upper = np.array([1, 102, 45, 45, 45, 45])

        super(DOC1, self).__init__(n_var=self.D,
                                   n_obj=self.M,
                                   n_iq_constr=7,
                                   n_eq_constr=0,
                                   xl=self.lower,
                                   xu=self.upper,
                                   name=self.__class__.__name__.upper())

    def _fn(self, X):
        X = np.maximum(np.minimum(X, np.tile(self.upper, (X.shape[0], 1))), np.tile(self.lower, (X.shape[0], 1)))
        g = 5.3578547 * X[:, 3] ** 2 + 0.8356891 * X[:, 1] * X[:, 5] + 37.293239 * X[:,
                                                                                   1] - 40792.141 + 30665.5386717834 + 1

        f1 = X[:, 0]
        f2 = g * (1 - np.sqrt(f1) / g)

        # Constraints in objective space
        c1 = np.maximum(-(f1 ** 2 + f2 ** 2 - 1), 0)

        # Constraints in decision space
        c2 = 85.334407 + 0.0056858 * X[:, 2] * X[:, 5] + 0.0006262 * X[:, 1] * X[:, 4] - 0.0022053 * X[:, 3] * X[:,
                                                                                                               5] - 92
        c3 = -85.334407 - 0.0056858 * X[:, 2] * X[:, 5] - 0.0006262 * X[:, 1] * X[:, 4] + 0.0022053 * X[:, 3] * X[:, 5]
        c4 = 80.51249 + 0.0071317 * X[:, 2] * X[:, 5] + 0.0029955 * X[:, 1] * X[:, 2] + 0.0021813 * X[:, 3] ** 2 - 110
        c5 = -80.51249 - 0.0071317 * X[:, 2] * X[:, 5] - 0.0029955 * X[:, 1] * X[:, 2] - 0.0021813 * X[:, 3] ** 2 + 90
        c6 = 9.300961 + 0.0047026 * X[:, 3] * X[:, 5] + 0.0012547 * X[:, 1] * X[:, 3] + 0.0019085 * X[:, 3] * X[:,
                                                                                                              4] - 25
        c7 = -9.300961 - 0.0047026 * X[:, 3] * X[:, 5] - 0.0012547 * X[:, 1] * X[:, 3] - 0.0019085 * X[:, 3] * X[:,
                                                                                                               4] + 20

        return np.column_stack([f1, f2, c1, c2, c3, c4, c5, c6, c7])

    def _calc_pareto_front(self, **kwargs):
        return load_pareto_front_from_file(os.path.join("DOC", f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"))


class DOC2(CMOP):
    def __init__(self):
        self.M = 2
        self.D = 16
        self.lower = np.array([0] + [0] * 15)
        self.upper = np.array([1] + [10] * 15)

        super(DOC2, self).__init__(n_var=self.D,
                                   n_obj=self.M,
                                   n_iq_constr=7,
                                   n_eq_constr=0,
                                   xl=self.lower,
                                   xu=self.upper,
                                   name=self.__class__.__name__.upper())

    def _fn(self, X):
        X = np.maximum(np.minimum(X, np.tile(self.upper, (X.shape[0], 1))), np.tile(self.lower, (X.shape[0], 1)))

        popsize, _ = X.shape

        b = np.array([-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1])
        c1 = np.array([[30, -20, -10, 32, -10],
                       [-20, 39, -6, -31, 32],
                       [-10, -6, 10, -6, -10],
                       [32, -31, -6, 39, -20],
                       [-10, 32, -10, -20, 30]])
        d = np.array([4, 8, 10, 6, 2])

        g_temp = np.sum(np.tile(c1[:5, 0], (popsize, 1)) * X[:, 11:16], axis=1) * X[:, 11] + \
                 np.sum(np.tile(c1[:5, 1], (popsize, 1)) * X[:, 11:16], axis=1) * X[:, 12] + \
                 np.sum(np.tile(c1[:5, 2], (popsize, 1)) * X[:, 11:16], axis=1) * X[:, 13] + \
                 np.sum(np.tile(c1[:5, 3], (popsize, 1)) * X[:, 11:16], axis=1) * X[:, 14] + \
                 np.sum(np.tile(c1[:5, 4], (popsize, 1)) * X[:, 11:16], axis=1) * X[:, 15] + \
                 2 * np.sum(np.tile(d, (popsize, 1)) * X[:, 11:16] ** 3, axis=1) - \
                 np.sum(np.tile(b, (popsize, 1)) * X[:, 1:11], axis=1)

        g = (g_temp - 32.6555929502) + 1

        f1 = X[:, 0]
        f2 = g * (1 - (f1) ** (1 / 3) / g)

        # Constraints in objective space
        g1 = np.maximum(-(np.sqrt(f1) + f2 - 1), 0)
        d1 = np.zeros((popsize, 3))
        d1[:, 0] = np.maximum((f1 - 1 / 8) ** 2 + (f2 - 1 + np.sqrt(1 / 8)) ** 2 - 0.15 ** 2, 0)
        d1[:, 1] = np.maximum((f1 - 1 / 2) ** 2 + (f2 - 1 + np.sqrt(1 / 2)) ** 2 - 0.15 ** 2, 0)
        d1[:, 2] = np.maximum((f1 - 7 / 8) ** 2 + (f2 - 1 + np.sqrt(7 / 8)) ** 2 - 0.15 ** 2, 0)
        g2 = np.min(d1, axis=1)

        a = np.array([[-16, 2, 0, 1, 0],
                      [0, -2, 0, 0.4, 2],
                      [-3.5, 0, 2, 0, 0],
                      [0, -2, 0, -4, -1],
                      [0, -9, -2, 1, -2.8],
                      [2, 0, -4, 0, 0],
                      [-1, -1, -1, -1, -1],
                      [-1, -2, -3, -2, -1],
                      [1, 2, 3, 4, 5],
                      [1, 1, 1, 1, 1]])

        c1 = np.array([[30, -20, -10, 32, -10],
                       [-20, 39, -6, -31, 32],
                       [-10, -6, 10, -6, -10],
                       [32, -31, -6, 39, -20],
                       [-10, 32, -10, -20, 30]])
        d = np.array([4, 8, 10, 6, 2])
        e = np.array([-15, -27, -36, -18, -12])

        # Constraints in decision space
        g3 = -2 * np.sum(np.tile(c1[:5, 0], (popsize, 1)) * X[:, 11:16], axis=1) - \
             3 * d[0] * X[:, 11] ** 2 - e[0] + np.sum(np.tile(a[:10, 0], (popsize, 1)) * X[:, 1:11], axis=1)
        g4 = -2 * np.sum(np.tile(c1[:5, 1], (popsize, 1)) * X[:, 11:16], axis=1) - \
             3 * d[1] * X[:, 12] ** 2 - e[1] + np.sum(np.tile(a[:10, 1], (popsize, 1)) * X[:, 1:11], axis=1)
        g5 = -2 * np.sum(np.tile(c1[:5, 2], (popsize, 1)) * X[:, 11:16], axis=1) - \
             3 * d[2] * X[:, 13] ** 2 - e[2] + np.sum(np.tile(a[:10, 2], (popsize, 1)) * X[:, 1:11], axis=1)
        g6 = -2 * np.sum(np.tile(c1[:5, 3], (popsize, 1)) * X[:, 11:16], axis=1) - \
             3 * d[3] * X[:, 14] ** 2 - e[3] + np.sum(np.tile(a[:10, 3], (popsize, 1)) * X[:, 1:11], axis=1)
        g7 = -2 * np.sum(np.tile(c1[:5, 4], (popsize, 1)) * X[:, 11:16], axis=1) - \
             3 * d[4] * X[:, 15] ** 2 - e[4] + np.sum(np.tile(a[:10, 4], (popsize, 1)) * X[:, 1:11], axis=1)

        return np.column_stack([f1, f2, g1, g2, g3, g4, g5, g6, g7])

    def _calc_pareto_front(self, **kwargs):
        return load_pareto_front_from_file(os.path.join("DOC", f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"))


class DOC3(CMOP):
    def __init__(self):
        self.M = 2
        self.D = 10
        self.lower = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01])
        self.upper = np.array([1, 1, 300, 100, 200, 100, 1, 100, 200, 0.03])

        super(DOC3, self).__init__(n_var=self.D,
                                   n_obj=self.M,
                                   n_iq_constr=10,
                                   n_eq_constr=0,
                                   xl=self.lower,
                                   xu=self.upper,
                                   name=self.__class__.__name__.upper())

    def _fn(self, X):
        X = np.maximum(np.minimum(X, np.tile(self.upper, (X.shape[0], 1))),
                       np.tile(self.lower, (X.shape[0], 1)))

        g_temp = -9 * X[:, 5] - 15 * X[:, 8] + 6 * X[:, 1] + 16 * X[:, 2] + 10 * (X[:, 6] + X[:, 7])
        g = (g_temp + 400.0551) + 1

        f1 = X[:, 0]
        f2 = g * (1 - f1 / g)

        # Constraints in objective space
        c1 = np.maximum(-(f1 ** 2 + f2 ** 2 - 1), 0)
        c2 = np.maximum(-(abs((-f1 + f2 - 0.5) / np.sqrt(2)) - 0.1 / np.sqrt(2)), 0)
        c3 = np.maximum(-(abs((-f1 + f2 - 0) / np.sqrt(2)) - 0.1 / np.sqrt(2)), 0)
        c4 = np.maximum(-(abs((-f1 + f2 + 0.5) / np.sqrt(2)) - 0.1 / np.sqrt(2)), 0)

        # Constraints in decision space
        c5 = X[:, 9] * X[:, 3] + 0.02 * X[:, 6] - 0.025 * X[:, 5]
        c6 = X[:, 9] * X[:, 4] + 0.02 * X[:, 7] - 0.015 * X[:, 8]
        c7 = abs(X[:, 1] + X[:, 2] - X[:, 3] - X[:, 4]) - 0.0001
        c8 = abs(0.03 * X[:, 1] + 0.01 * X[:, 2] - X[:, 9] * (X[:, 3] + X[:, 4])) - 0.0001
        c9 = abs(X[:, 3] + X[:, 6] - X[:, 5]) - 0.0001
        c10 = abs(X[:, 4] + X[:, 7] - X[:, 8]) - 0.0001

        return np.column_stack([f1, f2, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])

    def _calc_pareto_front(self, **kwargs):
        return load_pareto_front_from_file(os.path.join("DOC", f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"))


class DOC4(CMOP):
    def __init__(self):
        self.M = 2
        self.D = 8
        self.lower = np.array([0, -10, -10, -10, -10, -10, -10, -10])
        self.upper = np.array([1, 10, 10, 10, 10, 10, 10, 10])

        super(DOC4, self).__init__(n_var=self.D,
                                   n_obj=self.M,
                                   n_iq_constr=6,
                                   n_eq_constr=0,
                                   xl=self.lower,
                                   xu=self.upper,
                                   name=self.__class__.__name__.upper())

    def _fn(self, X):
        X = np.maximum(np.minimum(X, np.tile(self.upper, (X.shape[0], 1))),
                       np.tile(self.lower, (X.shape[0], 1)))

        g_temp = (X[:, 1] - 10) ** 2 + 5 * (X[:, 2] - 12) ** 2 + X[:, 3] ** 4 + 3 * (X[:, 4] - 11) ** 2 + \
                 10 * X[:, 5] ** 6 + 7 * X[:, 6] ** 2 + X[:, 7] ** 4 - 4 * X[:, 6] * X[:, 7] - 10 * X[:, 6] - 8 * X[:,
                                                                                                                  7]
        g = g_temp - 680.6300573745 + 1

        f1 = X[:, 0]
        f2 = g * (1 - np.sqrt(f1) / g)

        # Constraints in objective space
        c1 = np.maximum(-(f1 + f2 - 1), 0)
        c2 = np.maximum(-(f1 + f2 - 1 - np.abs(np.sin(10 * np.pi * (f1 - f2 + 1)))), 0)

        # Constraints in decision space
        c3 = -127 + 2 * X[:, 1] ** 2 + 3 * X[:, 2] ** 4 + X[:, 3] + 4 * X[:, 4] ** 2 + 5 * X[:, 5]
        c4 = -282 + 7 * X[:, 1] + 3 * X[:, 2] + 10 * X[:, 3] ** 2 + X[:, 4] - X[:, 5]
        c5 = -196 + 23 * X[:, 1] + X[:, 2] ** 2 + 6 * X[:, 6] ** 2 - 8 * X[:, 7]
        c6 = 4 * X[:, 1] ** 2 + X[:, 2] ** 2 - 3 * X[:, 1] * X[:, 2] + 2 * X[:, 3] ** 2 + 5 * X[:, 6] - 11 * X[:, 7]

        return np.column_stack([f1, f2, c1, c2, c3, c4, c5, c6])

    def _calc_pareto_front(self, **kwargs):
        return load_pareto_front_from_file(os.path.join("DOC", f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"))


class DOC5(CMOP):
    def __init__(self):
        self.M = 2
        self.D = 8
        self.lower = np.array([0, 0, 0, 0, 100, 6.3, 5.9, 4.5])
        self.upper = np.array([1, 1000, 40, 40, 300, 6.7, 6.4, 6.25])

        super(DOC5, self).__init__(n_var=self.D,
                                   n_obj=self.M,
                                   n_iq_constr=9,
                                   n_eq_constr=0,
                                   xl=self.lower,
                                   xu=self.upper,
                                   name=self.__class__.__name__.upper())

    def _fn(self, X):
        X = np.maximum(np.minimum(X, np.tile(self.upper, (X.shape[0], 1))), np.tile(self.lower, (X.shape[0], 1)))
        g_temp = X[:, 1]
        g = g_temp - 193.724510070035 + 1

        f1 = X[:, 0]
        f2 = g * (1 - np.sqrt(f1) / g)

        # Constraints in objective space
        c1 = np.maximum(-(f1 + f2 - 1), 0)
        c2 = np.maximum(
            -(f1 + f2 - 1 - np.abs(np.sin(10 * np.pi * (f1 - f2 + 1)))), 0)

        c3 = np.maximum((f1 - 0.8) * (f2 - 0.6), 0)

        # Constraints in decision space
        c4 = -X[:, 1] + 35 * X[:, 2] ** 0.6 + 35 * X[:, 3] ** 0.6
        c5 = np.abs(-300 * X[:, 3] + 7500 * X[:, 5] - 7500 * X[:, 6] - 25 * X[:, 4] * X[:, 5]
                    + 25 * X[:, 4] * X[:, 6] + X[:, 3] * X[:, 4]) - 0.0001
        c6 = np.abs(100 * X[:, 2] + 155.365 * X[:, 4] + 2500 * X[:, 7] -
                    X[:, 2] * X[:, 4] - 25 * X[:, 4] * X[:, 7] - 15536.5) - 0.0001
        c7 = np.abs(-X[:, 5] + np.log(-X[:, 4] + 900)) - 0.0001
        c8 = np.abs(-X[:, 6] + np.log(X[:, 4] + 300)) - 0.0001
        c9 = np.abs(-X[:, 7] + np.log(-2 * X[:, 4] + 700)) - 0.0001

        return np.column_stack([f1, f2, c1, c2, c3, c4, c5, c6, c7, c8, c9])

    def _calc_pareto_front(self, **kwargs):
        return load_pareto_front_from_file(os.path.join("DOC", f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"))


class DOC6(CMOP):
    def __init__(self):
        self.M = 2
        self.D = 11
        self.lower = [0, -10] + [-10] * 9
        self.upper = [1, 10] + [10] * 9

        super(DOC6, self).__init__(n_var=self.D,
                                   n_obj=self.M,
                                   n_iq_constr=10,
                                   n_eq_constr=0,
                                   xl=self.lower,
                                   xu=self.upper,
                                   name=self.__class__.__name__.upper())

    def _fn(self, X):
        X = np.maximum(np.minimum(X, np.tile(self.upper, (X.shape[0], 1))),
                       np.tile(self.lower, (X.shape[0], 1)))

        g_temp = (X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 1] * X[:, 2] - 14 * X[:, 1] -
                  16 * X[:, 2] + (X[:, 3] - 10) ** 2 + 4 * (X[:, 4] - 5) ** 2 +
                  (X[:, 5] - 3) ** 2 + 2 * (X[:, 6] - 1) ** 2 + 5 * X[:, 7] ** 2 +
                  7 * (X[:, 8] - 11) ** 2 + 2 * (X[:, 9] - 10) ** 2 + (X[:, 10] - 7) ** 2 + 45)

        g = g_temp - 24.3062090681 + 1

        f1 = X[:, 0]
        f2 = g * (1 - np.sqrt(f1) / g)

        # Constraints in objective space
        c1 = np.maximum(-(f1 + f2 - 1), 0)
        c2 = np.maximum(-(f1 - 0.5) * (f1 + f2 - 1 - np.abs(np.sin(10 * np.pi * (f1 - f2 + 1)))), 0)

        # Constraints in decision space
        c3 = -105 + 4 * X[:, 1] + 5 * X[:, 2] - 3 * X[:, 7] + 9 * X[:, 8]
        c4 = 10 * X[:, 1] - 8 * X[:, 2] - 17 * X[:, 7] + 2 * X[:, 8]
        c5 = -8 * X[:, 1] + 2 * X[:, 2] + 5 * X[:, 9] - 2 * X[:, 10] - 12
        c6 = 3 * (X[:, 1] - 2) ** 2 + 4 * (X[:, 2] - 3) ** 2 + 2 * X[:, 3] ** 2 - 7 * X[:, 4] - 120
        c7 = 5 * X[:, 1] ** 2 + 8 * X[:, 2] + (X[:, 3] - 6) ** 2 - 2 * X[:, 4] - 40
        c8 = X[:, 1] ** 2 + 2 * (X[:, 2] - 2) ** 2 - 2 * X[:, 1] * X[:, 2] + 14 * X[:, 5] - 6 * X[:, 6]
        c9 = 0.5 * (X[:, 1] - 8) ** 2 + 2 * (X[:, 2] - 4) ** 2 + 3 * X[:, 5] ** 2 - X[:, 6] - 30
        c10 = -3 * X[:, 1] + 6 * X[:, 2] + 12 * (X[:, 9] - 8) ** 2 - 7 * X[:, 10]

        return np.column_stack([f1, f2, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])

    def _calc_pareto_front(self, **kwargs):
        return load_pareto_front_from_file(os.path.join("DOC", f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"))


class DOC7(CMOP):
    def __init__(self):
        self.M = 2
        self.D = 11
        self.lower = [0] + [0] * 10
        self.upper = [1] + [10] * 10

        super(DOC7, self).__init__(n_var=self.D,
                                   n_obj=self.M,
                                   n_iq_constr=6,
                                   n_eq_constr=0,
                                   xl=self.lower,
                                   xu=self.upper,
                                   name=self.__class__.__name__.upper())

    def _fn(self, X):
        X = np.clip(X, np.array(self.lower), np.array(self.upper))
        c1 = [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.1, -10.708, -26.662, -22.179]
        X_temp = X[:, 1:11]
        sum_X = np.sum(X_temp, axis=1, keepdims=True)

        g_temp = np.sum(X_temp * (np.tile(c1, (X.shape[0], 1)) + np.log(1E-30 + X_temp / (1E-30 + sum_X))), axis=1)
        g = g_temp + 47.7648884595 + 1

        f1 = X[:, 0]
        f2 = g * (1 - np.sqrt(f1) / g)

        # Constraints in objective space
        c1 = np.maximum(-(f1 + f2 - 1), 0)
        c2 = np.maximum(-((f1 - 0.5) * (
                f1 + f2 - 1 - np.abs(np.sin(10 * np.pi * (f1 - f2 + 1))))),
                        0)
        c3 = np.maximum(-(np.abs(-f1 + f2) / np.sqrt(2) - 0.1 / np.sqrt(2)), 0)

        # Constraints in decision space
        c4 = np.abs(X[:, 2] + 2 * X[:, 3] + 2 * X[:, 4] + X[:, 7] + X[:, 10] - 2) - 0.0001
        c5 = np.abs(X[:, 5] + 2 * X[:, 6] + X[:, 7] + X[:, 8] - 1) - 0.0001
        c6 = np.abs(X[:, 4] + X[:, 8] + X[:, 9] + 2 * X[:, 10] + X[:, 10] - 1) - 0.0001

        return np.column_stack([f1, f2, c1, c2, c3, c4, c5, c6])

    def _calc_pareto_front(self, **kwargs):
        return load_pareto_front_from_file(os.path.join("DOC", f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"))


class DOC8(CMOP):
    def __init__(self):
        self.M = 3
        self.D = 10
        self.lower = np.array([0, 0, 500, 1000, 5000, 100, 100, 100, 100, 100])
        self.upper = np.array([1, 1, 1000, 2000, 6000, 500, 500, 500, 500, 500])

        super(DOC8, self).__init__(n_var=self.D,
                                   n_obj=self.M,
                                   n_iq_constr=7,
                                   n_eq_constr=0,
                                   xl=self.lower,
                                   xu=self.upper,
                                   name=self.__class__.__name__.upper())

    def _fn(self, X):
        X = np.maximum(np.minimum(X, np.tile(self.upper, (X.shape[0], 1))),
                       np.tile(self.lower, (X.shape[0], 1)))
        g_temp = X[:, 2] + X[:, 3] + X[:, 4]
        g = g_temp - 7049.2480205286 + 1

        f1 = (X[:, 0] * X[:, 1]) * g
        f2 = (X[:, 0] * (1 - X[:, 1])) * g
        f3 = (1 - X[:, 0]) * g

        # Constraints in objective space
        c1 = np.maximum(-(f3 - 0.4) * (f3 - 0.6), 0)

        # Constraints in decision space
        c2 = -1 + 0.0025 * (X[:, 5] + X[:, 7])
        c3 = -1 + 0.0025 * (X[:, 6] + X[:, 8] - X[:, 5])
        c4 = -1 + 0.01 * (X[:, 9] - X[:, 6])
        c5 = -X[:, 2] * X[:, 7] + 833.33252 * X[:, 5] + 100 * X[:, 2] - 83333.333
        c6 = -X[:, 3] * X[:, 8] + 1250 * X[:, 6] + X[:, 3] * X[:, 5] - 1250 * X[:, 5]
        c7 = -X[:, 4] * X[:, 9] + 1250000 + X[:, 4] * X[:, 6] - 2500 * X[:, 6]

        return np.column_stack([f1, f2, f3, c1, c2, c3, c4, c5, c6, c7])

    def _calc_pareto_front(self, **kwargs):
        return load_pareto_front_from_file(os.path.join("DOC", f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"))


class DOC9(CMOP):
    def __init__(self):
        self.M = 3
        self.D = 11
        self.lower = [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        self.upper = [1, 1, 10, 10, 10, 10, 10, 10, 10, 10, 10]

        super(DOC9, self).__init__(n_var=self.D,
                                   n_obj=self.M,
                                   n_iq_constr=14,
                                   n_eq_constr=0,
                                   xl=self.lower,
                                   xu=self.upper,
                                   name=self.__class__.__name__.upper())

    def _fn(self, X):
        X = np.maximum(np.minimum(X, np.tile(self.upper, (X.shape[0], 1))), np.tile(self.lower, (X.shape[0], 1)))
        g_temp = -0.5 * (X[:, 2] * X[:, 5] - X[:, 3] * X[:, 4] +
                         X[:, 4] * X[:, 10] - X[:, 6] * X[:, 10] +
                         X[:, 6] * X[:, 9] - X[:, 7] * X[:, 8])
        g = g_temp + 0.8660254038 + 1

        f1 = np.cos(0.5 * np.pi * X[:, 0]) * np.cos(0.5 * np.pi * X[:, 1]) * g
        f2 = np.cos(0.5 * np.pi * X[:, 0]) * np.sin(0.5 * np.pi * X[:, 1]) * g
        f3 = np.sin(0.5 * np.pi * X[:, 0]) * g

        # Constraints in objective space
        c1 = np.maximum(-(f1 ** 2 + f2 ** 2 - 1), 0)

        # Constraints in decision space
        c2 = (X[:, 4] ** 2 + X[:, 5] ** 2 - 1)[:, np.newaxis]
        c3 = (X[:, 10] ** 2 - 1)[:, np.newaxis]
        c4 = (X[:, 6] ** 2 + X[:, 7] ** 2 - 1)[:, np.newaxis]
        c5 = (X[:, 2] ** 2 + (X[:, 3] - X[:, 10]) ** 2 - 1)[:, np.newaxis]
        c6 = ((X[:, 2] - X[:, 6]) ** 2 + (X[:, 3] - X[:, 7]) ** 2 - 1)[:, np.newaxis]
        c7 = ((X[:, 2] - X[:, 8]) ** 2 + (X[:, 3] - X[:, 9]) ** 2 - 1)[:, np.newaxis]
        c8 = ((X[:, 4] - X[:, 6]) ** 2 + (X[:, 5] - X[:, 7]) ** 2 - 1)[:, np.newaxis]
        c9 = ((X[:, 4] - X[:, 8]) ** 2 + (X[:, 5] - X[:, 9]) ** 2 - 1)[:, np.newaxis]
        c10 = (X[:, 8] ** 2 + (X[:, 9] - X[:, 10]) ** 2 - 1)[:, np.newaxis]
        c11 = (X[:, 3] * X[:, 4] - X[:, 2] * X[:, 5])[:, np.newaxis]
        c12 = (-X[:, 5] * X[:, 10])[:, np.newaxis]
        c13 = (X[:, 7] * X[:, 10])[:, np.newaxis]
        c14 = (X[:, 8] * X[:, 9] - X[:, 7] * X[:, 10])[:, np.newaxis]

        return np.column_stack([f1, f2, f3, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14])

    def _calc_pareto_front(self, **kwargs):
        return load_pareto_front_from_file(os.path.join("DOC", f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"))

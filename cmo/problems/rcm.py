import os
import numpy as np
from cmop.utils import CMOP, load_pareto_front_from_file

__all__ = ['RCM1', 'RCM2', 'RCM3', 'RCM4', 'RCM5', 'RCM6', 'RCM7', 'RCM8', 'RCM10',
           'RCM11', 'RCM12', 'RCM13', 'RCM14', 'RCM15', 'RCM16', 'RCM17', 'RCM18', 'RCM19', 'RCM20',
           'RCM21', 'RCM25', 'RCM27', 'RCM29']


class RCM1(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([0.51, 0.51, 10, 10])
        xu = np.array([99.49, 99.49, 200, 200])
        super(RCM1, self).__init__(n_var=4, n_obj=2, n_iq_constr=2, n_eq_constr=0, xl=xl, xu=xu,
                                   scale_var=scale_var, scale_obj=scale_obj, name="RCM1")

    def _fn(self, X):
        x1 = np.round(X[:, 0])
        x2 = np.round(X[:, 1])
        x3 = X[:, 2]
        x4 = X[:, 3]

        z1 = 0.0625 * x1
        z2 = 0.0625 * x2

        # Objectives
        f1 = 1.7781 * z1 * x3 ** 2 + 0.6224 * z1 * x2 * x4 + 3.1661 * z1 ** 2 * x4 + 19.84 * z1 ** 2 * x3
        f2 = -np.pi * x3 ** 2 * x4 - (4 / 3) * np.pi * x3 ** 3

        # Constraints
        g1 = 0.00954 * x3 - z2
        g2 = 0.0193 * x3 - z1

        return np.column_stack([f1, f2, g1, g2])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM2(CMOP):

    def __init__(self, scale_var=False, scale_obj=False, *args, **kwargs):
        xl = np.array([0.05, 0.2, 0.2, 0.35, 3])
        xu = np.array([0.5, 0.5, 0.6, 0.5, 6])
        super(RCM2, self).__init__(n_var=5, n_obj=2, n_iq_constr=5, n_eq_constr=0, xl=xl, xu=xu,
                                   scale_var=scale_var, scale_obj=scale_obj, name="RCM2")

    def _fn(self, X):
        d1, d2, d3, b, L = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

        # Constants
        rho1 = 100
        rho2 = 2770
        rho3 = 7780
        E1 = 1.6
        E2 = 70
        E3 = 200
        c1 = 500
        c2 = 1500
        c3 = 800

        mu = 2 * b * (rho1 * d1 + rho2 * (d2 - d1) + rho3 * (d3 - d2))
        EI = (2 * b / 3) * (E1 * d1 ** 3 + E2 * (d2 ** 3 - d1 ** 3) + E3 * (d3 - d2))

        # Objectives
        f1 = (-np.pi) / (2 * L) ** 2 * (np.abs(EI / mu)) ** 0.5
        f2 = 2 * b * L * (c1 * d1 + c2 * (d2 - d1) + c3 * (d3 - d2))

        # Constraints
        g1 = mu * L - 2800
        g2 = d1 - d2
        g3 = d2 - d1 - 0.15
        g4 = d2 - d3
        g5 = d3 - d2 - 0.01

        return np.column_stack([f1, f2, g1, g2, g3, g4, g5])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM3(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([1e-5, 1e-5, 1])
        xu = np.array([100, 100, 3])
        super(RCM3, self).__init__(n_var=3, n_obj=2, n_iq_constr=3, n_eq_constr=0, xl=xl, xu=xu,
                                   scale_var=scale_var, scale_obj=scale_obj, name="RCM3")

    def _fn(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]

        # Objectives
        f1 = x1 * np.sqrt(16 + x3 ** 2) + x2 * np.sqrt(1 + x3 ** 2)
        f2 = (20 * np.sqrt(16 + x3 ** 2)) / (x3 * x1)

        # Constraints
        g1 = f1 - 0.1
        g2 = f2 - 1e5
        g3 = (80 * np.sqrt(1 + x3 ** 2)) / (x3 * x2) - 1e5

        return np.column_stack([f1, f2, g1, g2, g3])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM4(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([0.125, 0.1, 0.1, 0.125])
        xu = np.array([5, 10, 10, 5])
        super(RCM4, self).__init__(n_var=4, n_obj=2, n_iq_constr=4, n_eq_constr=0, xl=xl, xu=xu,
                                   scale_var=scale_var, scale_obj=scale_obj, name="RCM4")

    def _fn(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]

        # Constants
        P = 6000
        L = 14
        E = 30e6
        tmax = 13600
        sigmax = 30000
        G = 12e6

        Pc = (4.013 * E * ((x3 ** 2 + x4 ** 6) / 36) ** 0.5) / (L ** 2) * (1 - x3 / (2 * L) * (E / (4 * G)) ** 0.5)
        sigma = (6 * P * L) / (x4 * x3 ** 2)
        J = 2 * (np.sqrt(2) * x1 * x2 * (x2 ** 2 / 12 + ((x1 + x3) / 2) ** 2))
        R = np.sqrt(x2 ** 2 / 4 + ((x1 + x3) / 2) ** 2)
        M = P * (L + x2 / 2)
        tho1 = P / (np.sqrt(2) * x1 * x2)
        tho2 = M * R / J
        tho = np.sqrt(tho1 ** 2 + 2 * tho1 * tho2 * x2 / (2 * R) + tho2 ** 2)

        # Objectives
        f1 = 1.10471 * x1 ** 2 * x2 + 0.04811 * x3 * x4 * (14 + x2)
        f2 = (4 * P * L ** 3) / (E * x4 * x3 ** 3)

        # Constraints
        g1 = tho - tmax
        g2 = sigma - sigmax
        g3 = x1 - x4
        g4 = P - Pc

        return np.column_stack([f1, f2, g1, g2, g3, g4])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM5(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([55, 75, 1000, 11])
        xu = np.array([80, 110, 3000, 20])
        super(RCM5, self).__init__(n_var=4, n_obj=2, n_iq_constr=4, n_eq_constr=0, xl=xl, xu=xu,
                                   scale_var=scale_var, scale_obj=scale_obj, name="RCM5")

    def _fn(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]

        # Objectives
        f1 = 4.9e-5 * (x2 ** 2 - x1 ** 2) * (x4 - 1)
        f2 = 9.82e6 * (x2 ** 2 - x1 ** 2) / (x3 * x4 * (x2 ** 3 - x1 ** 3))

        # Constraints
        g1 = 20 - (x2 - x1)
        g2 = x3 / (3.14 * (x2 ** 2 - x1 ** 2)) - 0.4
        g3 = 2.22e-3 * x3 * (x2 ** 3 - x1 ** 3) / (x2 ** 2 - x1 ** 2) ** 2 - 1
        g4 = 900 - 2.66e-2 * x3 * x4 * (x2 ** 3 - x1 ** 3) / (x2 ** 2 - x1 ** 2)

        return np.column_stack([f1, f2, g1, g2, g3, g4])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM6(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([2.6, 0.7, 16.51, 7.3, 7.3, 2.9, 5])
        xu = np.array([3.6, 0.8, 28.49, 8.3, 8.3, 3.9, 5.5])
        super(RCM6, self).__init__(n_var=7, n_obj=2, n_iq_constr=11, n_eq_constr=0, xl=xl, xu=xu,
                                   scale_var=scale_var, scale_obj=scale_obj, name="RCM6")

    def _fn(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = np.round(X[:, 2])
        x4 = X[:, 3]
        x5 = X[:, 4]
        x6 = X[:, 5]
        x7 = X[:, 6]

        # Objectives
        f1 = 0.7854 * x1 * x2 ** 2 * (10 * x3 ** 2 / 3 + 14.933 * x3 - 43.0934) - 1.508 * (
                x6 ** 2 + x7 ** 2) + 7.477 * (x6 ** 3 + x7 ** 3) + 0.7854 * (x4 * x6 ** 2 + x5 * x7 ** 2)
        f2 = np.sqrt((745 * x4 / (x2 * x3)) ** 2 + 1.69e7) / (0.1 * x6 ** 3)

        # Constraints
        g1 = 1 / (x1 * x2 ** 2 * x3) - 1 / 27
        g2 = 1 / (x1 * x2 ** 2 * x3 ** 2) - 1 / 397.5
        g3 = x4 ** 3 / (x2 * x3 * x6 ** 4) - 1 / 1.93
        g4 = x5 ** 3 / (x2 * x3 * x7 ** 4) - 1 / 1.93
        g5 = x2 * x3 - 40
        g6 = x1 / x2 - 12
        g7 = -x1 / x2 + 5
        g8 = 1.9 - x4 + 1.5 * x6
        g9 = 1.9 - x5 + 1.1 * x7
        g10 = f2 - 1300
        g11 = np.sqrt((745 * x5 / (x2 * x3)) ** 2 + 1.575e8) / (0.1 * x7 ** 3) - 850

        return np.column_stack([f1, f2, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM7(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([11.51, 11.51, 11.51, 11.51])
        xu = np.array([60.49, 60.49, 60.49, 60.49])
        super(RCM7, self).__init__(n_var=4, n_obj=2, n_iq_constr=1, n_eq_constr=0, xl=xl, xu=xu,
                                   scale_var=scale_var, scale_obj=scale_obj, name="RCM7")

    def _fn(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]

        # Objectives
        f1 = np.abs(6.931 - x3 * x4 / (x1 * x2))
        f2 = np.max(X, axis=1)

        # Constraints
        g1 = f1 / 6.931 - 0.5

        return np.column_stack([f1, f2, g1])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM8(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4])
        xu = np.array([1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2])
        super(RCM8, self).__init__(n_var=7, n_obj=3, n_iq_constr=9, n_eq_constr=0, xl=xl, xu=xu,
                                   scale_var=scale_var, scale_obj=scale_obj, name="RCM8")

    def _fn(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]
        x6 = X[:, 5]
        x7 = X[:, 6]

        VMBP = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        VFD = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6

        # Objectives
        f1 = 1.98 + 4.9 * x1 * 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 1e-5 * x6 + 2.73 * x7
        f2 = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
        f3 = 0.5 * (VMBP + VFD)

        # Constraints
        g1 = -1 + 1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3
        g2 = -0.32 + 0.261 - 0.0159 * x1 * x2 - 0.06486 * x1 - 0.019 * x2 * x7 + 0.0144 * x2 * x5 + 0.0154464 * x6
        g3 = -0.32 + 0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 ** 2
        g4 = -0.32 + 0.214 + 0.00817 * x5 - 0.045195 * x1 - 0.0135168 * x1 + 0.03099 * x2 * x6 - 0.018 * x2 * x7 + 0.007176 * x3 + 0.023232 * x3 - 0.00364 * x5 * x6 - 0.018 * x2 ** 2
        g5 = -32 + 33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728
        g6 = -32 + 28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7
        g7 = -32 + 46.36 - 9.9 * x2 - 4.4505 * x1
        g8 = f2 - 4
        g9 = VMBP - 9.9

        return np.column_stack([f1, f2, f3, g1, g2, g3, g4, g5, g6, g7, g8, g9])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM10(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([0.1, 0.5])
        xu = np.array([2.25, 2.5])
        super(RCM10, self).__init__(n_var=2, n_obj=2, n_iq_constr=2, n_eq_constr=0, xl=xl, xu=xu,
                                    scale_var=scale_var, scale_obj=scale_obj, name="RCM10")

    def _fn(self, X):
        x1, x2 = X[:, [0]], X[:, [1]]

        # Constants
        rho = 0.283
        h = 100
        P = 1e4
        E = 3e7
        sigma0 = 2e4

        # Objectives
        f1 = 2 * rho * h * x2 * np.sqrt(1 + x1 ** 2)
        f2 = P * h * (1 + x1 ** 2) ** 1.5 * np.sqrt(1 + x1 ** 4) / (2 * np.sqrt(2) * E * x1 ** 2 * x2)

        # Constraints
        c1 = P * (1 + x1) * np.sqrt(1 + x1 ** 2) / (2 * np.sqrt(2) * x1 * x2) - sigma0
        c2 = P * (1 - x1) * np.sqrt(1 + x1 ** 2) / (2 * np.sqrt(2) * x1 * x2) - sigma0

        return np.column_stack([f1, f2, c1, c2])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM11(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([0., 0.01, 0.01])
        xu = np.array([0.45, 0.1, 0.1])
        super(RCM11, self).__init__(n_var=3, n_obj=5, n_iq_constr=7, n_eq_constr=0, xl=xl, xu=xu,
                                    scale_var=scale_var, scale_obj=scale_obj, name="RCM11")

    def _fn(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]

        # Objectives
        f1 = 106780.37 * (x2 + x3) + 61704.67
        f2 = 3000 * x1
        f3 = 305700 * 2289 * x2 / (0.06 * 2289) ** 0.65
        f4 = 250 * 2289 * np.exp(-39.75 * x2 + 9.9 * x3 + 2.74)
        f5 = 25 * (1.39 / (x1 * x2) + 4940 * x3 - 80)

        # Constraints
        g1 = 1 - (0.00139 / (x1 * x2) + 4.94 * x3 - 0.08)
        g2 = 1 - (0.000306 / (x1 * x2) + 1.082 * x3 - 0.0986)
        g3 = 50000 - (12.307 / (x1 * x2) + 49408.24 * x3 + 4051.02)
        g4 = 16000 - (2.098 / (x1 * x2) + 8046.33 * x3 - 696.71)
        g5 = 10000 - (2.138 / (x1 * x2) + 7883.39 * x3 - 705.04)
        g6 = 2000 - (0.417 * x1 * x2 + 1721.26 * x3 - 136.54)
        g7 = 550 - (0.164 / (x1 * x2) + 631.13 * x3 - 54.48)

        return np.column_stack([f1, f2, f3, f4, f5, -g1, -g2, -g3, -g4, -g5, -g6, -g7])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM12(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([10, 10, 0.9, 0.9])
        xu = np.array([80, 50, 5, 5])
        super(RCM12, self).__init__(n_var=4, n_obj=2, n_iq_constr=1, n_eq_constr=0, xl=xl, xu=xu,
                                    scale_var=scale_var, scale_obj=scale_obj, name="RCM12")

    def _fn(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]

        # Constants
        P = 600
        L = 200
        E = 2e4

        # Objectives
        f1 = 2 * x2 * x4 + x3 * (x1 - 2 * x4)
        f2 = P * L ** 3 / (
                48 * E * (x3 * ((x1 - 2 * x4) ** 3) + 2 * x2 * x4 * (4 * x4 ** 2 + 3 * x1 * (x1 - 2 * x4))) / 12)

        # Constraints
        g1 = -16 + 180000 * x1 / (
                x3 * ((x1 - 2 * x4) ** 3) + 2 * x2 * x4 * (4 * x4 ** 2 + 3 * x1 * (x1 - 2 * x4))) + 15000 * x2 / (
                     (x1 - 2 * x4) * x3 ** 3 + 2 * x4 * x2 ** 3)

        return np.column_stack([f1, f2, g1])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM13(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([2.6, 0.7, 16.51, 7.3, 7.3, 2.9, 5])
        xu = np.array([3.6, 0.8, 28.49, 8.3, 8.3, 3.9, 5.5])
        super(RCM13, self).__init__(n_var=7, n_obj=3, n_iq_constr=11, n_eq_constr=0, xl=xl, xu=xu,
                                    scale_var=scale_var, scale_obj=scale_obj, name="RCM13")

    def _fn(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = np.round(X[:, 2])
        x4 = X[:, 3]
        x5 = X[:, 4]
        x6 = X[:, 5]
        x7 = X[:, 6]

        # Objectives
        f1 = 0.7854 * x1 * x2 ** 2 * (10 * x3 ** 2 / 3 + 14.933 * x3 - 43.0934) - 1.508 * (
                x6 ** 2 + x7 ** 2) + 7.477 * (x6 ** 3 + x7 ** 3) + 0.7854 * (x4 * x6 ** 2 + x5 * x7 ** 2)
        f2 = np.sqrt((745 * x4 / (x2 * x3)) ** 2 + 1.69e7) / (0.1 * x6 ** 3)
        f3 = np.sqrt((745 * x5 / (x2 * x3)) ** 2 + 1.575e8) / (0.1 * x7 ** 3)

        # Constraints
        g1 = 1 / (x1 * x2 ** 2 * x3) - 1 / 27
        g2 = 1 / (x1 * x2 ** 2 * x3 ** 2) - 1 / 397.5
        g3 = x4 ** 3 / (x2 * x3 * x6 ** 4) - 1 / 1.93
        g4 = x5 ** 3 / (x2 * x3 * x7 ** 4) - 1 / 1.93
        g5 = x2 * x3 - 40
        g6 = x1 / x2 - 12
        g7 = -x1 / x2 + 5
        g8 = 1.9 - x4 + 1.5 * x6
        g9 = 1.9 - x5 + 1.1 * x7
        g10 = f2 - 1300
        g11 = f3 - 1100

        return np.column_stack([f1, f2, f3, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM14(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([60, 90, 1, 0, 2])
        xu = np.array([80, 110, 3, 1000, 9])
        super(RCM14, self).__init__(n_var=5, n_obj=2, n_iq_constr=8, n_eq_constr=0, xl=xl, xu=xu,
                                    scale_var=scale_var, scale_obj=scale_obj, name="RCM14")

    def _fn(self, X):
        # Constants
        Mf = 3
        Ms = 40
        Iz = 55
        n = 250
        Tmax = 15
        s = 1.5
        delta = 0.5
        Vsrmax = 10
        rho = 0.0000078
        pmax = 1
        mu = 0.6
        Lmax = 30
        delR = 20

        Rsr = 2 / 3 * (X[:, 1] ** 3 - X[:, 0] ** 3) / (X[:, 1] ** 2 * X[:, 0] ** 2)
        Vsr = np.pi * Rsr * n / 30
        A = np.pi * (X[:, 1] ** 2 - X[:, 0] ** 2)
        Prz = X[:, 3] / A
        w = np.pi * n / 30
        Mh = 2 / 3 * mu * X[:, 3] * X[:, 4] * (X[:, 1] ** 3 - X[:, 0] ** 3) / (X[:, 1] ** 2 - X[:, 0] ** 2)
        T = Iz * w / (Mh + Mf)

        # Objectives
        f1 = np.pi * (X[:, 1] ** 2 - X[:, 0] ** 2) * X[:, 2] * (X[:, 4] + 1) * rho
        f2 = T

        # Constraints
        g1 = -X[:, 1] + X[:, 0] + delR
        g2 = (X[:, 4] + 1) * (X[:, 2] + delta) - Lmax
        g3 = Prz - pmax
        g4 = Prz * Vsr - pmax * Vsrmax
        g5 = Vsr - Vsrmax
        g6 = T - Tmax
        g7 = s * Ms - Mh
        g8 = -T

        return np.column_stack([f1, f2, g1, g2, g3, g4, g5, g6, g7, g8])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM15(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([0.51, 0.6, 0.51])
        xu = np.array([70.49, 3, 42.49])
        super(RCM15, self).__init__(n_var=3, n_obj=2, n_iq_constr=8, n_eq_constr=0, xl=xl, xu=xu,
                                    scale_var=scale_var, scale_obj=scale_obj, name="RCM15")

    def _fn(self, X):
        x1 = np.round(X[:, 0])
        x2 = X[:, 1]
        d = np.array([0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014,
                      0.015, 0.0162, 0.0173, 0.018, 0.020, 0.023, 0.025,
                      0.028, 0.032, 0.035, 0.041, 0.047, 0.054, 0.063,
                      0.072, 0.080, 0.092, 0.0105, 0.120, 0.135, 0.148,
                      0.162, 0.177, 0.192, 0.207, 0.225, 0.244, 0.263,
                      0.283, 0.307, 0.331, 0.362, 0.394, 0.4375, 0.500])
        x3 = d[np.clip(np.round(X[:, 2]).astype(int), 1, 42) - 1]

        # Constants
        cf = (4 * x2 / x3 - 1) / (4 * x2 / x3 - 4) + 0.615 * x3 / x2
        K = (11.5 * 10 ** 6 * x3 ** 4) / (8 * x1 * x2 ** 3)
        lf = 1000 / K + 1.05 * (x1 + 2) * x3
        sigp = 300 / K

        # Objective function
        f1 = (np.pi ** 2 * x2 * x3 ** 2 * (x1 + 2)) / 4
        f2 = (8000 * cf * x2) / (np.pi * x3 ** 3)

        # Constraints
        g1 = (8000 * cf * x2) / (np.pi * x3 ** 3) - 189000
        g2 = lf - 14
        g3 = 0.2 - x3
        g4 = x2 - 3
        g5 = 3 - x2 / x3
        g6 = sigp - 6
        g7 = sigp + 700 / K + 1.05 * (x1 + 2) * x3 - lf
        g8 = 1.25 - 700 / K

        return np.column_stack([f1, f2, g1, g2, g3, g4, g5, g6, g7, g8])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM16(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([0.01, 0.2])
        xu = np.array([0.05, 1.0])
        super(RCM16, self).__init__(n_var=2, n_obj=2, n_iq_constr=2, n_eq_constr=0, xl=xl, xu=xu,
                                    scale_var=scale_var, scale_obj=scale_obj, name="RCM16")

    def _fn(self, X):
        x1, x2 = X[:, [0]], X[:, [1]]

        # Constants
        rho = 7800.0
        P = 1000
        E = 2.07e11
        Sy = 3e8
        d_max = 0.005

        # Objectives
        f1 = 0.25 * rho * np.pi * x2 * x1 ** 2
        f2 = 64 * P * x2 ** 3 / (3 * E * np.pi * x1 ** 4)

        # Constraints
        c1 = 32 * P * x2 / (np.pi * x1 ** 3) - Sy
        c2 = f2 - d_max

        return np.column_stack([f1, f2, c1, c2])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM17(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([150.0, 20.0, 13.0, 10.0, 14.0, 0.63])
        xu = np.array([274.32, 32.31, 25.0, 11.71, 18.0, 0.75])
        super(RCM17, self).__init__(n_var=6, n_obj=3, n_iq_constr=9, n_eq_constr=0, xl=xl, xu=xu,
                                    scale_var=scale_var, scale_obj=scale_obj, name="RCM17")

    def _fn(self, X):
        L = X[:, 0]
        B = X[:, 1]
        D = X[:, 2]
        T = X[:, 3]
        V_k = X[:, 4]
        C_B = X[:, 5]

        a = 4977.06 * C_B ** 2 - 8105.61 * C_B + 4456.51
        b = -10847.2 * C_B ** 2 + 12817 * C_B - 6960.32
        F_n = 0.5144 / (9.8065 * L) ** 0.5
        P = ((1.025 * L * B * T * C_B) ** (2 / 3) * V_k ** 3) / (a + b * F_n)

        W_s = 0.034 * L ** 1.7 * B ** 0.6 * D ** 0.4 * C_B ** 0.5
        W_o = L ** 0.8 * B ** 0.6 * D ** 0.3 * C_B ** 0.1
        W_m = 0.17 * P ** 0.9
        ls = W_s + W_o + W_m

        D_wt = 1.025 * L * B * T * C_B - ls
        F_c = 0.19 * 24 * P / 1000 + 0.2
        D_cwt = D_wt - F_c * ((5000 * V_k) / 24 + 5) - 2 * D_wt ** 0.5
        R_trp = 350 / ((5000 * V_k) / 24 + 2 * (D_cwt / 8000 + 0.5))
        ac = D_cwt * R_trp
        S_d = 5000 * V_k / 24

        C_c = 0.2 * 1.3 * (2000 * W_s ** 0.85 + 3500 * W_o + 2400 * P ** 0.8)
        C_r = 40000 * D_wt ** 0.3
        C_v = (1.05 * 100 * F_c * S_d + 6.3 * D_wt ** 0.8) * R_trp

        # Objectives
        f1 = (C_c + C_r + C_v) / ac
        f2 = ls
        f3 = -ac

        # Constraints
        g1 = L / B - 6
        g2 = 15 - L / D
        g3 = 19 - L / T
        g4 = 0.45 * D_wt ** 0.31 - T
        g5 = 0.7 * D + 0.7 - T
        g6 = 0.32 - F_n
        g7 = 0.53 * T + ((0.085 * C_B - 0.002) * B ** 2) / (T * C_B) - (1 + 0.52 * D) - 0.07 * B
        g8 = D_wt - 3000
        g9 = 500000 - D_wt

        return np.column_stack([f1, f2, f3, -g1, -g2, -g3, -g4, -g5, -g6, -g7, -g8, -g9])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM18(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([136, 56, 1.4])
        xu = np.array([146, 68, 2.2])
        super(RCM18, self).__init__(n_var=3, n_obj=2, n_iq_constr=3, n_eq_constr=0, xl=xl, xu=xu,
                                    scale_var=scale_var, scale_obj=scale_obj, name="RCM18")

    def _fn(self, X):
        hh = X[:, 0]
        w = X[:, 1]
        t = X[:, 2]

        # Constants
        Ea = 14496.5
        Fa = 234.9

        E = -70973.4 + 958.656 * w + 614.173 * hh - 3.827 * w * hh + 57.023 * w * t + 63.274 * hh * t - 3.582 * w ** 2 - 1.4842 * hh ** 2 - 1890.174 * t ** 2
        F = 111.854 - 20.210 * w + 7.560 * hh - 0.025 * w * hh + 2.731 * w * t - 1.479 * hh * t + 0.165 * w ** 2

        # Objectives
        f1 = Ea / E
        f2 = F / Fa

        # Constraints
        g1 = (hh - 136) * (146 - hh)
        g2 = (w - 58) * (66 - w)
        g3 = (t - 1.4) * (2.2 - t)

        return np.column_stack([f1, f2, -g1, -g2, -g3])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM19(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([0.51, 0.51, 0.51, 250, 250, 250, 6, 4, 40, 10])
        xu = np.array([3.49, 3.49, 3.49, 2500, 2500, 2500, 20, 16, 700, 450])
        super(RCM19, self).__init__(n_var=10, n_obj=3, n_iq_constr=10, n_eq_constr=0, xl=xl, xu=xu,
                                    scale_var=scale_var, scale_obj=scale_obj, name="RCM19")

    def _fn(self, X):
        N1 = np.round(X[:, 0])
        N2 = np.round(X[:, 1])
        N3 = np.round(X[:, 2])

        V1 = X[:, 3]
        V2 = X[:, 4]
        V3 = X[:, 5]

        TL1 = X[:, 6]
        TL2 = X[:, 7]

        B1 = X[:, 8]
        B2 = X[:, 9]

        # Constants
        S = np.array([[2, 3, 4],
                      [4, 6, 3]])
        t = np.array([[8, 20, 8],
                      [16, 4, 4]])
        H = 6000
        alp = 250
        beta = 0.6
        Q1 = 40000
        Q2 = 20000

        # Objectives
        f1 = alp * (N1 * V1 ** beta + N2 * V2 ** beta + N3 * V3 ** beta)
        f2 = 65 * (Q1 / B1 + Q2 / B2) + 0.08 * Q1 + 0.1 * Q2
        f3 = Q1 * TL1 / B1 + Q2 * TL2 / B2

        # Constraints
        g1 = Q1 * TL1 / B1 + Q2 * TL2 / B2 - H
        g2 = S[0, 0] * B1 + S[1, 0] * B2 - V1
        g3 = S[0, 1] * B1 + S[1, 1] * B2 - V2
        g4 = S[0, 2] * B1 + S[1, 2] * B2 - V3
        g5 = t[0, 0] - N1 * TL1
        g6 = t[0, 1] - N2 * TL1
        g7 = t[0, 2] - N3 * TL1
        g8 = t[1, 0] - N1 * TL2
        g9 = t[1, 1] - N2 * TL2
        g10 = t[1, 2] - N3 * TL2

        return np.column_stack([f1, f2, f3, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM20(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([1, 1, 1e-6, 1])
        xu = np.array([16, 16, 16 * 1e-6, 16])
        super(RCM20, self).__init__(n_var=4, n_obj=2, n_iq_constr=7, n_eq_constr=0, xl=xl, xu=xu,
                                    scale_var=scale_var, scale_obj=scale_obj, name="RCM20")

    def _fn(self, X):
        R = X[:, 0]
        Ro = X[:, 1]
        mu = X[:, 2]
        Q = X[:, 3]

        # Constants
        gamma = 0.0307
        C = 0.5
        n = -3.55
        C1 = 10.04
        Ws = 101000
        Pmax = 1000
        delTmax = 50
        hmin = 0.001
        gg = 386.4
        N = 750

        P = np.log10(np.maximum(np.log10(8.122e6 * mu + 0.8) - C1, 1e-10)) / n
        delT = 2 * (10 ** P - 560)
        Ef = 9336 * Q * gamma * C * delT
        h = ((2 * np.pi * N / 60) ** 2 * 2 * np.pi * mu / Ef * (R ** 4 / 4 - Ro ** 4 / 4) - 1e-5)
        Po = (6 * mu * Q / (np.pi * h ** 3)) * np.log(R / Ro)
        W = np.pi * Po / 2 * (R ** 2 - Ro ** 2) / (np.log(R / Ro) - 1e-5)

        # Objectives
        f1 = (Q * Po / 0.7 + Ef) / 12
        f2 = gamma / (gg * Po) * (Q / (2 * np.pi * R * h))

        # Constraints
        g1 = Ws - W
        g2 = Po - Pmax
        g3 = delT - delTmax
        g4 = hmin - h
        g5 = Ro - R
        g6 = f2 - 0.001
        g7 = W / (np.pi * (R ** 2 - Ro ** 2) + 1e-5) - 5000

        return np.column_stack([f1, f2, g1, g2, g3, g4, g5, g6, g7])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM21(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([1.3, 2.5, 1.3, 1.3, 1.3, 1.3])
        xu = np.array([1.7, 3.5, 1.7, 1.7, 1.7, 1.7])
        super(RCM21, self).__init__(n_var=6, n_obj=2, n_iq_constr=4, n_eq_constr=0, xl=xl, xu=xu,
                                    scale_var=scale_var, scale_obj=scale_obj, name="RCM21")

    def _fn(self, X):
        x1, x2, x3, x4, x5, x6 = np.hsplit(X, 6)

        # Objectives
        f1 = 1.3667145844797 - 0.00904459793976106 * x1 - 0.0016193573938033 * x2 - 0.00758531275221425 * x3 - \
             0.00440727360327102 * x4 - 0.00572216860791644 * x5 - 0.00936039926190721 * x6 + \
             2.62510221107328e-6 * (x1 ** 2) + 4.92982681358861e-7 * (x2 ** 2) + 2.25524989067108e-6 * (x3 ** 2) + \
             1.84605439400301e-6 * (x4 ** 2) + 2.17175358243416e-6 * (x5 ** 2) + 3.90158043948054e-6 * (x6 ** 2) + \
             4.55276994245781e-7 * x1 * x2 - 6.37013576290982e-7 * x1 * x3 + 8.26736480446359e-7 * x1 * x4 + \
             5.66352809442276e-8 * x1 * x5 - 3.20213897443278e-7 * x1 * x6 + \
             1.18015467772812e-8 * x2 * x3 + 9.25820391546515e-8 * x2 * x4 - 1.05705364119837e-7 * x2 * x5 - \
             4.74797783014687e-7 * x2 * x6 - 5.02319867013788e-7 * x3 * x4 + 9.54284258085225e-7 * x3 * x5 + \
             1.80533309229454e-7 * x3 * x6 - 1.07938022118477e-6 * x4 * x5 - 1.81370642220182e-7 * x4 * x6 - \
             2.24238851688047e-7 * x5 * x6

        f2 = -1.19896668942683 + 3.04107017009774 * x1 + 1.23535701600191 * x2 + 2.13882039381528 * x3 + \
             2.33495178382303 * x4 + 2.68632494801975 * x5 + 3.43918953617606 * x6 - \
             7.89144544980703e-4 * (x1 ** 2) - 2.06085185698215e-4 * (x2 ** 2) - 7.15269900037858e-4 * (x3 ** 2) - \
             7.8449237573837e-4 * (x4 ** 2) - 9.31396896237177e-4 * (x5 ** 2) - 1.40826531972195e-3 * (x6 ** 2) - \
             1.60434988248392e-4 * x1 * x2 + 2.0824655419411e-4 * x1 * x3 - 3.0530659653553e-4 * x1 * x4 - \
             8.10145973591615e-5 * x1 * x5 + 6.94728759651311e-5 * x1 * x6 + \
             1.18015467772812e-8 * x2 * x3 + 9.25820391546515e-8 * x2 * x4 - 1.05705364119837e-7 * x2 * x5 + \
             1.69935290196781e-4 * x2 * x6 + 2.32421829190088e-5 * x3 * x4 - 2.0808624041163476e-4 * x3 * x5 + \
             1.75576341867273e-5 * x3 * x6 + 2.68422081654044e-4 * x4 * x5 + 4.39852066801981e-5 * x4 * x6 + \
             2.96785446021357e-5 * x5 * x6

        # Constraints
        g1 = f1 - 5
        g2 = -f1
        g3 = f2 - 28
        g4 = -f2

        return np.column_stack([f1, f2, g1, g2, g3, g4])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM25(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([0, -0.49])
        xu = np.array([1.6, 1.49])
        super(RCM25, self).__init__(n_var=2, n_obj=2, n_iq_constr=2, n_eq_constr=0, xl=xl, xu=xu,
                                    scale_var=scale_var, scale_obj=scale_obj, name="RCM25")

    def _fn(self, X):
        x1 = X[:, 0]
        x2 = np.round(X[:, 1])

        # Objectives
        f1 = x2 + 2 * x1
        f2 = -x1 ** 2 - x2

        # Constraints
        g1 = f2 + 1.25
        g2 = x1 + x2 - 1.6

        return np.column_stack([f1, f2, g1, g2])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM27(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([0.2, -2.22554, -0.49])
        xu = np.array([1, -1 ,1.49])
        super(RCM27, self).__init__(n_var=3, n_obj=2, n_iq_constr=3, n_eq_constr=0, xl=xl, xu=xu,
                                    scale_var=scale_var, scale_obj=scale_obj, name="RCM27")

    def _fn(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]

        # Objectives
        f1 = -0.7 * x3 + 0.8 + 5 * (0.5 - x1) ** 2
        f2 = x1 - x3

        # Constraints
        g1 = -(np.exp(x1 - 0.2) + x2)
        g2 = x2 + 1.1 * x3 - 1
        g3 = x1 - x3 - 0.2

        return np.column_stack([f1, f2, g1, g2, g3])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))


class RCM29(CMOP):

    def __init__(self, scale_var=False, scale_obj=False):
        xl = np.array([0, 0, 0, -0.49, -0.49, -0.49, -0.49])
        xu = np.array([100, 100, 100, 1.49, 1.49, 1.49, 1.49])
        super(RCM29, self).__init__(n_var=7, n_obj=2, n_iq_constr=9, n_eq_constr=0, xl=xl, xu=xu,
                                    scale_var=scale_var, scale_obj=scale_obj, name="RCM29")

    def _fn(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = np.round(X[:, 3])
        x5 = np.round(X[:, 4])
        x6 = np.round(X[:, 5])
        x7 = np.round(X[:, 6])

        # Objectives
        f1 = (1 - x4) ** 2 + (1 - x5) ** 2 + (1 - x6) ** 2 - np.log(np.abs(1 + x7) + 1e-6)
        f2 = (1 - x1) ** 2 + (2 - x2) ** 2 + (3 - x3) ** 2

        # Constraints
        g1 = x1 + x2 + x3 + x4 + x5 + x6 - 5
        g2 = x6 ** 3 + x1 ** 2 + x2 ** 2 + x3 ** 2 - 5.5
        g3 = x1 + x4 - 1.2
        g4 = x2 + x5 - 1.8
        g5 = x3 + x6 - 2.5
        g6 = x1 + x7 - 1.2
        g7 = x5 ** 2 + x2 ** 2 - 1.64
        g8 = x6 ** 2 + x3 ** 2 - 4.25
        g9 = x5 ** 2 + x3 ** 2 - 4.64

        return np.column_stack([f1, f2, g1, g2, g3, g4, g5, g6, g7, g8, g9])

    def _calc_pareto_front(self, *args, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("RCM", fname))

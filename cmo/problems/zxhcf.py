import os
import numpy as np

from cmo.problems.utils import CMOP, load_pareto_front_from_file

__all__ = ['ZXHCF1', 'ZXHCF2', 'ZXHCF3', 'ZXHCF4', 'ZXHCF5', 'ZXHCF6', 'ZXHCF7', 'ZXHCF8',
           'ZXHCF9', 'ZXHCF10', 'ZXHCF11', 'ZXHCF12', 'ZXHCF13', 'ZXHCF14', 'ZXHCF15', 'ZXHCF16']


class ZXHCF(CMOP):

    def __init__(self, name, n_iq_constr, k=False, n_obj=3, n_var=None):
        self.n_obj = n_obj
        if n_var is None:
            self.n_var = self.n_obj + 10
        else:
            self.n_var = n_var

        if not self.n_var > self.n_obj:
            raise ValueError('Number of variables must be greater than the number of objectives.')

        self.lower = np.full(self.n_var, 1e-10)
        self.upper = np.full(self.n_var, 1.0 - 1e-10)

        if k:
            if n_obj <= 3:
                self.k = n_obj - 1
            elif n_obj > 3 and n_obj <= 8:
                self.k = int(np.floor(n_obj / 2))
            else:
                self.k = 3
            n_iq_constr = self.k + n_iq_constr

        super(ZXHCF, self).__init__(n_var=self.n_var,
                                    n_obj=self.n_obj,
                                    n_iq_constr=n_iq_constr,
                                    n_eq_constr=0,
                                    xl=self.lower,
                                    xu=self.upper,
                                    name=name)


class ZXHCF1(ZXHCF):
    def __init__(self, n_obj=3, n_var=None):
        super(ZXHCF1, self).__init__(name=self.__class__.__name__.upper(), n_iq_constr=1, n_obj=n_obj, n_var=n_var)

    def _fn(self, X):
        OptX = 0.2
        N, D = X.shape
        M = self.n_obj

        Sx = np.cumsum(X[:, :M][:, ::-1] ** 2, axis=1)[:, ::-1]

        # Step 2: Compute theta
        Sx_sqrt = np.sqrt(Sx[:, 1:])
        THETA = 2 / np.pi * np.arctan(Sx_sqrt / X[:, :M - 1])

        # Step 3: Calculate Sphere function
        h = np.sum((X[:, M:D] - OptX) ** 2, axis=1)

        # Step 4: Compute T_
        T = (1 - Sx[:, 0]) ** 2 + h

        # Step 5: Objectives (linear)
        G = np.column_stack((np.ones(N), np.cumprod(THETA, axis=1))) * np.column_stack((1 - THETA, np.ones(N)))
        f = G * np.tile((1 + T)[:, np.newaxis], (1, M))

        # Step 6: Constraints
        c = Sx[:, 0] + h - 1

        return np.column_stack([f, c])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("ZXHCF", fname))


class ZXHCF2(ZXHCF):
    def __init__(self, n_obj=3, n_var=None):
        super(ZXHCF2, self).__init__(name=self.__class__.__name__.upper(), n_iq_constr=2, n_obj=n_obj, n_var=n_var)

    def _fn(self, X):
        OptX = 0.2
        N, D = X.shape
        M = self.n_obj

        # Step 1: Compute cumsum
        Sx = np.cumsum(X[:, :M][:, ::-1] ** 2, axis=1)[:, ::-1]

        # Step 2: Compute theta
        THETA = (2 / np.pi) * np.arctan(np.sqrt(Sx[:, 1:]) / X[:, :M - 1])

        # Step 3: Calculate Rosenbrock function
        h = np.sum(100 * ((X[:, M:-1] - OptX) ** 2 - (X[:, M + 1:] - OptX) ** 2) +
                   (X[:, M:-1] - OptX) ** 2, axis=1)

        # Step 4: Compute T_
        T = (1 - Sx[:, 0]) ** 2 + h

        # Step 5: Objectives (concave)
        G = np.hstack([np.ones((N, 1)), np.cumprod(np.sin((np.pi / 2) * THETA), axis=1)]) * \
            np.hstack([np.cos((np.pi / 2) * THETA), np.ones((N, 1))])
        f = G * (1 + T)[:, np.newaxis]

        # Step 6: Constraints
        c = np.zeros((N, 2))
        c[:, 0] = Sx[:, 0] + h - 1
        c[:, 1] = -(Sx[:, 0] + h - 1 / 4)

        return np.column_stack([f, c])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("ZXHCF", fname))


class ZXHCF3(ZXHCF):
    def __init__(self, n_obj=3, n_var=None):
        if n_var is not None:
            if not n_var > n_obj:
                raise ValueError('Number of variables must be greater than number of objectives.')

        super(ZXHCF3, self).__init__(name=self.__class__.__name__.upper(), n_iq_constr=2, n_obj=n_obj, n_var=n_var)

    def _fn(self, X):
        OptX = 0.2
        N, D = X.shape
        M = self.n_obj

        # Step 1: Compute cumsum
        Sx = np.cumsum(X[:, :M][:, ::-1] ** 2, axis=1)[:, ::-1]

        # Step 2: Compute theta
        THETA = 2 / np.pi * np.arctan(np.sqrt(Sx[:, 1:]) / X[:, : M - 1])

        # Step 3: Calculate Ackley function
        h = 20 - 20 * np.exp(
            -0.2 * np.sqrt(np.sum((X[:, M:] - OptX) ** 2, axis=1) / (D - M))) + np.exp(
            1) - np.exp(np.sum(np.cos(2 * np.pi * (X[:, M:] - OptX)), axis=1) / (D - M))

        # Step 4: Compute T_
        T = (1 - Sx[:, 0]) ** 2 + h

        # Step 5: Objectives (convex)
        G = 1 - np.concatenate((np.ones((N, 1)), np.cumprod(np.sin(np.pi / 2 * THETA), axis=1)),
                               axis=1) * np.concatenate((np.cos(np.pi / 2 * THETA), np.ones((N, 1))), axis=1)
        f = G * (1 + T).reshape((-1, 1))

        # Step 6: Constraints
        c = np.zeros((N, 2))
        c[:, 0] = Sx[:, 0] + h - 1
        c[:, 1] = -(Sx[:, 0] + h - 1 / 2)

        return np.column_stack([f, c])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("ZXHCF", fname))


class ZXHCF4(ZXHCF):
    def __init__(self, n_obj=3, n_var=None):
        super(ZXHCF4, self).__init__(name=self.__class__.__name__.upper(), n_iq_constr=2, n_obj=n_obj, n_var=n_var)

    def _fn(self, X):
        OptX = 0.2
        N, D = X.shape
        M = self.n_obj

        # Step 1: Compute cumsum
        Sx = np.cumsum(X[:, :M][:, ::-1] ** 2, axis=1)[:, ::-1]

        # Step 2: Compute theta
        Sx_sqrt = np.sqrt(Sx[:, 1:])
        THETA = (2 / np.pi) * np.arctan(Sx_sqrt / X[:, :M - 1])

        # Step 3: Calculate Griewank function
        h = 5 * (np.sum((X[:, M:] - OptX) ** 2, axis=1) -
                 np.prod(np.cos(10 * np.pi * (X[:, M:] - OptX) / np.sqrt(np.arange(1, D - M + 1))), axis=1) + 1)

        # Step 4: Compute T_
        T = (1 - Sx[:, 0]) ** 2 + h

        # Step 5: Objectives (mixed)
        L = 2  # number of segments
        THETA_cos = np.cos(np.pi / 2 * THETA)
        G = 1 - np.hstack([np.ones((N, 1)), np.cumprod(np.sin(np.pi / 2 * THETA), axis=1)]) * np.hstack(
            [THETA_cos, np.ones((N, 1))])
        G[:, 0] = THETA[:, 0] - np.cos(2 * np.pi * L * THETA[:, 0] + np.pi / 2) / (2 * L * np.pi)
        f = G * np.tile((1 + T)[:, np.newaxis], (1, M))

        # Step 6: Constraints
        c = np.zeros((N, 2))
        c[:, 0] = Sx[:, 0] + h - 1
        c[:, 1] = -(Sx[:, 0] + h - 3 / 4)

        return np.column_stack([f, c])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("ZXHCF", fname))


class ZXHCF5(ZXHCF):
    def __init__(self, n_obj=3, n_var=None):
        super(ZXHCF5, self).__init__(name=self.__class__.__name__.upper(), n_iq_constr=1, k=True, n_obj=n_obj,
                                     n_var=n_var)

    def _fn(self, X):
        OptX = 0.2
        N, D = X.shape
        M = self.n_obj

        Sx = np.cumsum(X[:, :M][:, ::-1] ** 2, axis=1)[:, ::-1]

        # Step 2: Compute THETA
        Sx_sqrt = np.sqrt(Sx[:, 1:])
        PopDec_first_M_minus_1 = X[:, :M - 1]
        THETA = (2 / np.pi) * np.arctan(Sx_sqrt / PopDec_first_M_minus_1)

        # Step 3: Calculate Rosenbrock function
        h = np.sum(100 * ((X[:, M:M + M - 1] - OptX) ** 2 - (X[:, M + 1:M + M] - OptX) ** 2) + (
                X[:, M:M + M - 1] - OptX) ** 2, axis=1)

        # Step 4: Compute T
        T = (1 - Sx[:, 0]) ** 2 + h

        # Step 5: Compute G
        THETA_sin = np.sin(np.pi / 2 * THETA)
        G = 1 - np.hstack((np.ones((N, 1)), np.cumprod(THETA_sin, axis=1))) * np.hstack(
            (np.cos(np.pi / 2 * THETA), np.ones((N, 1))))
        f = G * np.tile((1 + T)[:, np.newaxis], (1, M))

        # Step 6: Constraints
        c = np.zeros((N, self.k + 1))
        c[:, 0] = Sx[:, 0] + h - 1
        for i in range(self.k):
            c[:, i + 1] = np.maximum(1 / 4 - THETA[:, i], THETA[:, i] - 3 / 4)

        return np.column_stack([f, c])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("ZXHCF", fname))


class ZXHCF6(ZXHCF):
    def __init__(self, n_obj=3, n_var=None):
        super(ZXHCF6, self).__init__(name=self.__class__.__name__.upper(), n_iq_constr=2, k=True, n_obj=n_obj,
                                     n_var=n_var)

    def _fn(self, X):
        OptX = 0.2
        N, D = X.shape
        M = self.n_obj

        # Step 1: Compute cumsum
        Sx = np.cumsum(X[:, :M][:, ::-1] ** 2, axis=1)[:, ::-1]

        # Step 2: Compute theta
        THETA = (2 / np.pi) * np.arctan(np.sqrt(Sx[:, 1:]) / X[:, :M - 1])

        # Step 3: Calculate Sphere function
        h = np.sum(np.power(X[:, M:D] - OptX, 2), axis=1)

        # Step 4: Compute T_
        T = np.power(1 - Sx[:, 0], 2) + h

        # Step 5: Objectives (mixed)
        A = 2  # number of segments
        G = 1 - np.column_stack([np.ones(N), np.cumprod(np.sin((np.pi / 2) * THETA), axis=1)]) * \
            np.column_stack([np.cos((np.pi / 2) * THETA), np.ones(N)])
        G[:, 0] = THETA[:, 0] - (np.cos(2 * np.pi * A * THETA[:, 0] + np.pi / 2) / (2 * A * np.pi))
        f = G * (1 + T).reshape(-1, 1)

        # Step 6: Constraints
        c = np.zeros((N, self.k + 2))
        c[:, 0] = Sx[:, 0] + h - 1
        c[:, 1] = -(Sx[:, 0] + h - 1 / 4)
        for i in range(self.k):
            c[:, i + 2] = np.maximum(1 / 4 - THETA[:, i], THETA[:, i] - 3 / 4)

        return np.column_stack([f, c])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("ZXHCF", fname))


class ZXHCF7(ZXHCF):
    def __init__(self, n_obj=3, n_var=None):
        super(ZXHCF7, self).__init__(name=self.__class__.__name__.upper(), n_iq_constr=2, k=True, n_obj=n_obj,
                                     n_var=n_var)

    def _fn(self, X):
        OptX = 0.2
        N, D = X.shape
        M = self.n_obj

        # Step 1: Compute cumsum
        Sx = np.cumsum(X[:, :M][:, ::-1] ** 2, axis=1)[:, ::-1]

        # Step 2: Compute theta
        THETA = 2 / np.pi * np.arctan(np.sqrt(Sx[:, 1:]) / X[:, :M - 1])

        # Step 3: Calculate Griewank function
        h = 5 * (np.sum((X[:, M:] - OptX) ** 2, axis=1) - np.prod(
            np.cos(10 * np.pi * (X[:, M:] - OptX) / np.sqrt(np.arange(1, D - M + 1))), axis=1) + 1)

        # Step 4: Compute T_
        T = (1 - Sx[:, 0]) ** 2 + h

        # Step 5: Objectives (linear)
        G = np.column_stack((np.ones(N), np.cumprod(THETA, axis=1))) * np.column_stack((1 - THETA, np.ones(N)))

        f = G * (1 + T)[:, np.newaxis]

        # Step 6: Constraints
        c = np.zeros((N, self.k + 2))
        c[:, 0] = Sx[:, 0] + h - 1
        c[:, 1] = -(Sx[:, 0] + h - 1 / 2)

        for i in range(self.k):
            c[:, i + 2] = np.maximum(1 / 4 - THETA[:, i], THETA[:, i] - 3 / 4)

        return np.column_stack([f, c])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("ZXHCF", fname))


class ZXHCF8(ZXHCF):
    def __init__(self, n_obj=3, n_var=None):
        if n_var is not None:
            if not n_var > n_obj:
                raise ValueError('Number of variables must be greater than number of objectives.')

        super(ZXHCF8, self).__init__(name=self.__class__.__name__.upper(), n_iq_constr=2, k=True, n_obj=n_obj,
                                     n_var=n_var)

    def _fn(self, X):
        OptX = 0.2
        N, D = X.shape
        M = self.n_obj

        # Step 1: Compute cumsum
        Sx = np.cumsum(X[:, :M][:, ::-1] ** 2, axis=1)[:, ::-1]

        # Step 2: Compute theta
        THETA = (2 / np.pi) * np.arctan(np.sqrt(Sx[:, 1:]) / X[:, :M - 1])

        # Step 3: Calculate Ackley function
        h = 20 - 20 * np.exp(-0.2 * np.sqrt(np.sum((X[:, M:] - OptX) ** 2, axis=1) / (D - M))) + np.exp(
            1) - np.exp(np.sum(np.cos(2 * np.pi * (X[:, M:] - OptX)), axis=1) / (D - M))

        # Step 4: Compute T_
        T = (1 - Sx[:, 0]) ** 2 + h

        # Step 5: Objectives (concave)
        G = np.cumprod(np.sin(np.pi / 2 * THETA), axis=1)
        G = np.column_stack((np.ones(N), G))
        G *= np.column_stack((np.cos(np.pi / 2 * THETA), np.ones(N)))
        f = G * (1 + T)[:, np.newaxis]

        # Step 6: Constraints
        c = np.zeros((N, self.k + 2))
        c[:, 0] = Sx[:, 0] + h - 1
        c[:, 1] = -(Sx[:, 0] + h - 3 / 4)

        for i in range(self.k):
            c[:, i + 2] = np.maximum(1 / 4 - THETA[:, i], THETA[:, i] - 3 / 4)

        return np.column_stack([f, c])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("ZXHCF", fname))


class ZXHCF9(ZXHCF):
    def __init__(self, n_obj=3, n_var=None):
        super(ZXHCF9, self).__init__(name=self.__class__.__name__.upper(), n_iq_constr=1, k=True, n_obj=n_obj,
                                     n_var=n_var)

    def _fn(self, X):
        OptX = 0.2
        N, D = X.shape
        M = self.n_obj

        # Step 1: Compute cumsum
        Sx = np.cumsum(X[:, :M][:, ::-1] ** 2, axis=1)[:, ::-1]

        # Step 2: Compute theta
        THETA = (2 / np.pi) * np.arctan(np.sqrt(Sx[:, 1:]) / X[:, :M - 1])

        # Step 3: Calculate Ackley function
        h = 20 - 20 * np.exp(-0.2 * np.sqrt(np.sum((X[:, M:] - OptX) ** 2, axis=1) / (D - M))) + np.exp(
            1) - np.exp(np.sum(np.cos(2 * np.pi * (X[:, M:] - OptX)) / (D - M)))

        # Step 4: Compute T_
        T = (1 - Sx[:, 0]) ** 2 + h

        # Step 5: Objectives (mixed)
        A = 2  # number of segments
        G = 1 - (np.concatenate([np.ones((N, 1)), np.cumprod(np.sin(np.pi / 2 * THETA), axis=1)], axis=1) *
                 np.concatenate([np.cos(np.pi / 2 * THETA), np.ones((N, 1))], axis=1))
        G[:, 0] = THETA[:, 0] - np.cos(2 * np.pi * A * THETA[:, 0] + np.pi / 2) / (2 * A * np.pi)
        f = G * np.tile((1 + T)[:, np.newaxis], (1, M))

        # Step 6: Constraints
        c = np.zeros((N, self.k + 1))
        c[:, 0] = Sx[:, 0] + h - 1
        for i in range(self.k):
            c[:, i + 1] = np.minimum(THETA[:, i] - 1 / 4, 3 / 4 - THETA[:, i])

        return np.column_stack([f, c])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("ZXHCF", fname))


class ZXHCF10(ZXHCF):
    def __init__(self, n_obj=3, n_var=None):
        super(ZXHCF10, self).__init__(name=self.__class__.__name__.upper(), n_iq_constr=2, k=True, n_obj=n_obj,
                                      n_var=n_var)

    def _fn(self, X):
        OptX = 0.2
        N, D = X.shape
        M = self.n_obj

        # Step 1: Compute cumsum
        Sx = np.cumsum(X[:, :M][:, ::-1] ** 2, axis=1)[:, ::-1]

        # Step 2: Compute theta
        THETA = (2 / np.pi) * np.arctan(np.sqrt(Sx[:, 1:]) / X[:, :M - 1])

        # Step 3: Calculate Griewank function
        h = 5 * (np.sum((X[:, M:] - OptX) ** 2, axis=1) -
                 np.prod(np.cos(10 * np.pi * (X[:, M:] - OptX) / np.sqrt(np.arange(1, D - M + 1))), axis=1) + 1)

        # Step 4: Compute T_
        T = (1 - Sx[:, 0]) ** 2 + h

        # Step 5: Objectives (convex)
        sin_theta = np.sin(np.pi / 2 * THETA)
        cos_theta = np.cos(np.pi / 2 * THETA)

        # Create the matrix G
        G = 1 - np.column_stack((np.ones(N), np.cumprod(sin_theta, axis=1))) * np.column_stack((cos_theta, np.ones(N)))
        f = G * (1 + T)[:, np.newaxis]

        # Step 6: Constraints
        c = np.zeros((N, self.k + 2))
        c[:, 0] = Sx[:, 0] + h - 1
        c[:, 1] = -(Sx[:, 0] + h - 1 / 4)

        for i in range(self.k):
            c[:, i + 2] = np.minimum(THETA[:, i] - 1 / 4, 3 / 4 - THETA[:, i])

        return np.column_stack([f, c])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("ZXHCF", fname))

class ZXHCF11(ZXHCF):
    def __init__(self, n_obj=3, n_var=None):
        super(ZXHCF11, self).__init__(name=self.__class__.__name__.upper(), n_iq_constr=2, k=True, n_obj=n_obj,
                                      n_var=n_var)

    def _fn(self, X):
        OptX = 0.2
        N, D = X.shape
        M = self.n_obj

        # Step 1: Compute cumsum
        Sx = np.cumsum(X[:, :M][:, ::-1] ** 2, axis=1)[:, ::-1]

        # Step 2: Compute theta
        THETA = (2 / np.pi) * np.arctan(np.sqrt(Sx[:, 1:]) / X[:, :M - 1])

        # Step 3: Calculate Sphere function
        h = np.sum((X[:, M:D] - OptX) ** 2, axis=1)

        # Step 4: Compute T_
        T = (1 - Sx[:, 0]) ** 2 + h

        # Step 5: Objectives (concave)
        G = np.column_stack([np.ones(N), np.cumprod(np.sin(np.pi / 2 * THETA), axis=1)])
        G *= np.column_stack([np.cos(np.pi / 2 * THETA), np.ones(N)])
        f = G * (1 + T)[:, None]

        # Step 6: Constraints
        c = np.zeros((N, self.k + 2))
        c[:, 0] = Sx[:, 0] + h - 1
        c[:, 1] = -(Sx[:, 0] + h - 1 / 2)
        for i in range(self.k):
            c[:, i + 2] = np.minimum(THETA[:, i] - 1 / 4, 3 / 4 - THETA[:, i])

        return np.column_stack([f, c])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("ZXHCF", fname))


class ZXHCF12(ZXHCF):
    def __init__(self, n_obj=3, n_var=None):
        super(ZXHCF12, self).__init__(name=self.__class__.__name__.upper(), n_iq_constr=2, k=True, n_obj=n_obj,
                                      n_var=n_var)

    def _fn(self, X):
        OptX = 0.2
        N, D = X.shape
        M = self.n_obj

        # Step 1: Compute cumsum
        Sx = np.cumsum(X[:, :M][:, ::-1] ** 2, axis=1)[:, ::-1]

        # Step 2: Compute theta
        THETA = (2 / np.pi) * np.arctan(np.sqrt(Sx[:, 1:]) / X[:, :M - 1])

        # Step 3: Calculate Rosenbrock function
        h = np.sum(100 * ((X[:, M + 1:-1] - OptX) ** 2 - (X[:, M + 2:] - OptX) ** 2) + (
                X[:, M + 1:-1] - OptX) ** 2, axis=1)

        # Step 4: Compute T_
        T = (1 - Sx[:, 0]) ** 2 + h

        # Step 5: Objectives (linear)
        G = np.hstack((np.ones((N, 1)), np.cumprod(THETA, axis=1))) * np.column_stack((1 - THETA, np.ones(N)))
        f = G * np.tile((1 + T)[:, np.newaxis], (1, M))

        # Step 6: Constraints
        c = np.zeros((N, self.k + 2))
        c[:, 0] = Sx[:, 0] + h - 1
        c[:, 1] = -(Sx[:, 0] + h - 3 / 4)

        for i in range(self.k):
            c[:, i + 2] = np.minimum(THETA[:, i] - 1 / 4, 3 / 4 - THETA[:, i])

        return np.column_stack([f, c])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("ZXHCF", fname))


class ZXHCF13(ZXHCF):
    def __init__(self, n_obj=3, n_var=None):
        super(ZXHCF13, self).__init__(name=self.__class__.__name__.upper(), n_iq_constr=1, k=True, n_obj=n_obj,
                                      n_var=n_var)

    def _fn(self, X):
        OptX = 0.2
        N, D = X.shape
        M = self.n_obj

        # Step 1: Compute cumsum
        Sx = np.cumsum(X[:, :M][:, ::-1] ** 2, axis=1)[:, ::-1]

        # Step 2: Compute theta
        THETA = (2 / np.pi) * np.arctan(np.sqrt(Sx[:, 1:]) / X[:, :M - 1])

        # Step 3: Calculate Griewank function
        h = 5 * (np.sum((X[:, M:] - OptX) ** 2, axis=1) -
                 np.prod(np.cos(10 * np.pi * (X[:, M:] - OptX) / np.sqrt(np.arange(1, D - M + 1))), axis=1) + 1)

        # Step 4: Compute T_
        T = (1 - Sx[:, 0]) ** 2 + h

        # Step 5: Objectives (concave)
        # Calculate the product of sin(pi/2*THETA) cumulatively
        sin_theta = np.sin(np.pi / 2 * THETA)
        cumulative_product = np.cumprod(sin_theta, axis=1)

        # Create arrays of ones for multiplication
        ones_N = np.ones(N)
        ones_N_col = ones_N[:, np.newaxis]

        # Calculate the final result by element-wise multiplication
        G = np.multiply(np.column_stack((ones_N, cumulative_product)),
                        np.column_stack((np.cos(np.pi / 2 * THETA), ones_N_col)))

        f = G * (1 + T)[:, np.newaxis]

        # Step 6: Constraints
        c = np.zeros((N, self.k + 1))
        c[:, 0] = Sx[:, 0] + h - 1
        for i in range(self.k):
            c[:, i + 1] = np.minimum(np.minimum(THETA[:, i] - 1 / 10, 4 / 5 - THETA[:, i]),
                                     np.maximum(2 / 5 - THETA[:, i], THETA[:, i] - 7 / 10))

        return np.column_stack([f, c])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("ZXHCF", fname))


class ZXHCF14(ZXHCF):
    def __init__(self, n_obj=3, n_var=None):
        if n_var is not None:
            if not n_var > n_obj:
                raise ValueError('Number of variables must be greater than number of objectives.')

        super(ZXHCF14, self).__init__(name=self.__class__.__name__.upper(), n_iq_constr=2, k=True, n_obj=n_obj,
                                      n_var=n_var)

    def _fn(self, X):
        OptX = 0.2
        N, D = X.shape
        M = self.n_obj

        # Step 1: Compute cumsum
        Sx = np.cumsum(X[:, :M][:, ::-1] ** 2, axis=1)[:, ::-1]

        # Step 2: Compute theta
        THETA = (2 / np.pi) * np.arctan(np.sqrt(Sx[:, 1:]) / X[:, :M - 1])

        # Step 3: Calculate Ackley function
        h = 20 - 20 * np.exp(-0.2 * np.sqrt(np.sum((X[:, M:] - OptX) ** 2, axis=1) / (D - M))) + np.exp(
            1) - np.exp(np.sum(np.cos(2 * np.pi * (X[:, M:] - OptX)), axis=1) / (D - M))

        # Step 4: Compute T_
        T = (1 - Sx[:, 0]) ** 2 + h

        # Step 5: Objectives (linear)
        G = np.hstack([np.ones((N, 1)), np.cumprod(THETA, axis=1)]) * np.column_stack((1 - THETA, np.ones(N)))
        f = G * (1 + T)[:, np.newaxis]

        # Step 6: Constraints
        c = np.zeros((N, self.k + 2))
        c[:, 0] = Sx[:, 0] + h - 1
        c[:, 1] = -(Sx[:, 0] + h - 1 / 4)
        for i in range(self.k):
            c[:, i + 2] = np.minimum(np.minimum(THETA[:, i] - 1 / 10, 4 / 5 - THETA[:, i]),
                                     np.maximum(2 / 5 - THETA[:, i], THETA[:, i] - 7 / 10))

        return np.column_stack([f, c])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("ZXHCF", fname))


class ZXHCF15(ZXHCF):
    def __init__(self, n_obj=3, n_var=None):
        super(ZXHCF15, self).__init__(name=self.__class__.__name__.upper(), n_iq_constr=2, k=True, n_obj=n_obj,
                                      n_var=n_var)

    def _fn(self, X):
        OptX = 0.2
        N, D = X.shape
        M = self.n_obj

        # Step 1: Compute cumsum
        Sx = np.cumsum(X[:, :M][:, ::-1] ** 2, axis=1)[:, ::-1]

        # Step 2: Compute theta
        THETA = (2 / np.pi) * np.arctan(np.sqrt(Sx[:, 1:]) / X[:, :M - 1])

        # Step 3: Calculate Rosenbrock function
        h = np.sum(100 * ((X[:, M + 1:-1] - OptX) ** 2 - (X[:, M + 2:] - OptX) ** 2) +
                   (X[:, M + 1:-1] - OptX) ** 2, axis=1)

        # Step 4: Compute T_
        T = (1 - Sx[:, 0]) ** 2 + h

        # Step 5: Objectives (mixed)
        A = 2  # number of segments
        G = 1 - np.hstack((np.ones((N, 1)), np.cumprod(np.sin(np.pi / 2 * THETA), axis=1))) * \
            np.hstack((np.cos(np.pi / 2 * THETA), np.ones((N, 1))))
        G[:, 0] = THETA[:, 0] - np.cos(2 * np.pi * A * THETA[:, 0] + np.pi / 2) / (2 * A * np.pi)
        f = G * np.tile((1 + T)[:, np.newaxis], (1, M))

        # Step 6: Constraints
        c = np.zeros((N, self.k + 2))
        c[:, 0] = Sx[:, 0] + h - 1
        c[:, 1] = -(Sx[:, 0] + h - 1 / 4)
        for i in range(self.k):
            c[:, i + 2] = np.minimum(np.minimum(THETA[:, i] - 1 / 10, 4 / 5 - THETA[:, i]),
                                     np.maximum(2 / 5 - THETA[:, i], THETA[:, i] - 7 / 10))

        return np.column_stack([f, c])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("ZXHCF", fname))


class ZXHCF16(ZXHCF):
    def __init__(self, n_obj=3, n_var=None):
        super(ZXHCF16, self).__init__(name=self.__class__.__name__.upper(), n_iq_constr=2, k=True, n_obj=n_obj,
                                      n_var=n_var)

    def _fn(self, X):
        OptX = 0.2
        N, D = X.shape
        M = self.n_obj

        # Step 1: Compute cumsum
        Sx = np.cumsum(X[:, :M][:, ::-1] ** 2, axis=1)[:, ::-1]

        # Step 2: Compute theta
        THETA = (2 / np.pi) * np.arctan(np.sqrt(Sx[:, 1:]) / X[:, :M - 1])

        # Step 3: Calculate Sphere function
        h = np.sum((X[:, M:D] - OptX) ** 2, axis=1)

        # Step 4: Compute T_
        T = (1 - Sx[:, 0]) ** 2 + h

        # Step 5: Objectives (convex)
        G = 1 - np.hstack([np.ones((N, 1)), np.cumprod(np.sin((np.pi / 2) * THETA), axis=1)]) * np.hstack(
            [np.cos((np.pi / 2) * THETA), np.ones((N, 1))])
        f = G * np.tile((1 + T)[:, np.newaxis], (1, M))

        # Step 6: Constraints
        c = np.zeros((N, self.k + 2))
        c[:, 0] = Sx[:, 0] + h - 1
        c[:, 1] = -(Sx[:, 0] + h - 1 / 4)

        for i in range(self.k):
            c[:, i + 2] = np.minimum(np.minimum(THETA[:, i] - 1 / 10, 4 / 5 - THETA[:, i]),
                                     np.maximum(2 / 5 - THETA[:, i], THETA[:, i] - 7 / 10))

        return np.column_stack([f, c])

    def _calc_pareto_front(self, **kwargs):
        fname = f"{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf"
        return load_pareto_front_from_file(os.path.join("ZXHCF", fname))

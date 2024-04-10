import numpy as np
from cmop.utils import CMOP, load_pareto_front_from_file


__all__ = ['CTP1', 'CTP2', 'CTP3', 'CTP4', 'CTP5', 'CTP6', 'CTP7', 'CTP8']


# DEAP convention #


def ctp(X, theta, a, b, c, d, e):
    """
    Generator for constrained multiobjective problems CTP.
    """

    X = np.atleast_2d(X)

    # Distance values
    h = X[:, 1:] ** 2 - 10 * np.cos(2 * np.pi * X[:, 1:])
    g = 1 + 10 * (X.shape[1] - 1) + np.sum(h, axis=1, keepdims=True)

    # Objective values
    f1 = X[:, [0]]
    f2 = g * (1 - np.sqrt(f1 / g))

    # Constraint values
    h1 = np.cos(theta) * (f2 - e) - np.sin(theta) * f1
    h2 = np.sin(theta) * (f2 - e) + np.cos(theta) * f1
    h3 = a * np.abs(np.sin(b * np.pi * h2 ** c)) ** d
    c = h3 - h1

    return np.column_stack([f1, f2, c])


def ctp1(X):
    """
    CTP1 constrained multiobjective optimization problem.
    """

    X = np.atleast_2d(X)

    # Distance values
    h = X[:, 1:] ** 2 - 10 * np.cos(2 * np.pi * X[:, 1:])
    g = 1 + 10 * (X.shape[1] - 1) + np.sum(h, axis=1, keepdims=True)

    # Objective values
    f1 = X[:, [0]]
    f2 = g * np.exp(-f1 / g)

    # Constraint values
    c1 = f2 - 0.858 * np.exp(-0.541 * f1)
    c2 = f2 - 0.728 * np.exp(-0.295 * f1)

    return np.column_stack([f1, f2, -c1, -c2])


def ctp2(X):
    """
    CTP2 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e = -0.2 * np.pi, 0.2, 10, 1, 6, 1
    return ctp(X, theta, a, b, c, d, e)


def ctp3(X):
    """
    CTP3 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e = -0.2 * np.pi, 0.1, 10, 1, 0.5, 1
    return ctp(X, theta, a, b, c, d, e)


def ctp4(X):
    """
    CTP4 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e = -0.2 * np.pi, 0.75, 10, 1, 0.5, 1
    return ctp(X, theta, a, b, c, d, e)


def ctp5(X):
    """
    CTP5 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e = -0.2 * np.pi, 0.1, 10, 2, 0.5, 1
    return ctp(X, theta, a, b, c, d, e)


def ctp6(X):
    """
    CTP6 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e = 0.1 * np.pi, 40, 0.5, 1, 2, -2
    return ctp(X, theta, a, b, c, d, e)


def ctp7(X):
    """
    CTP7 constrained multiobjective optimization problem.
    """
    theta, a, b, c, d, e = -0.05 * np.pi, 40, 5, 1, 6, 0
    return ctp(X, theta, a, b, c, d, e)


def ctp8(X):
    """
    CTP8 constrained multiobjective optimization problem.
    """
    theta1, a1, b1, c1, d1, e1 = -0.05 * np.pi, 40, 2, 1, 6, 0
    theta2, a2, b2, c2, d2, e2 = 0.1 * np.pi, 40, 0.5, 1, 2, -2
    Fk = ctp(X, theta1, a1, b1, c1, d1, e1)
    C = ctp(X, theta2, a2, b2, c2, d2, e2)
    return np.column_stack([Fk, C[:, [2]]])


# Generator #


class CTP(CMOP):
    """
    The CPT test suite generator of constrained multibjecitve problems CTP1-CTP8.

    Parameters
    ----------
    :param prob_id (int): CTP problem id.
    :param dim (int): Dimension of the decision space.
    :param scale_variable (bool, optional): Whether the decision variables are scaled to [0, 1].
    Default is False.

    Raise
    -----
    :raise ValueError: If prob_id is not in {1, ..., 8} or n_var is smaller than 2.

    References
    ----------
    [Deb2001] K. Deb, A. Pratap, T. Meyarivan, "Constrained Test Problems for Multi-objective
    Evolutionary Optimization," Evolutionary Multi-Criterion Optimization (EMO 2001), pp. 284-298,
    doi: 10.1007/3-540-44719-9_20.
    """

    _prob = {
        1: ctp1,
        2: ctp2,
        3: ctp3,
        4: ctp4,
        5: ctp5,
        6: ctp6,
        7: ctp7,
        8: ctp8
    }

    def __init__(self, prob_id, n_obj=2, n_var=2, scale_var=False, scale_obj=False):

        if prob_id not in set(range(1, 9)):
            raise ValueError("Please select a valid prob id.")

        # Set fn
        self.fn = self._prob[prob_id]

        # Set dim
        if n_var < 2:
            raise ValueError("Please select a larger value for n_var (>= 2).")

        # Set obj
        if n_obj != 2:
            raise ValueError("CTP suite can only be instantiated with 2 objectives.")

        n_iq_constr = 2 if prob_id in {1, 8} else 1

        # Set xl, xu
        xl = np.array([0.] + [-5.] * (n_var - 1))
        xu = np.array([1.] + [5.] * (n_var - 1))

        # Set name
        name = self.fn.__name__.upper()

        super(CTP, self).__init__(n_var=n_var,
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

    def _fn(self, X):
        return self.fn(X)


# Pymoo convention #

class CTP1(CTP):
    """
    CTP1 constrained multiobjective optimization problem.
    """

    def __init__(self, n_obj=2, n_var=5, scale_var=False, scale_obj=False):
        super(CTP1, self).__init__(prob_id=1, n_obj=n_obj, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)

    def _calc_pareto_front(self):
        return load_pareto_front_from_file(f"CTP/{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf")


class CTP2(CTP):
    """
    CTP2 constrained multiobjective optimization problem.
    """

    def __init__(self, n_obj=2, n_var=5, scale_var=False, scale_obj=False):
        super(CTP2, self).__init__(prob_id=2, n_obj=n_obj, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)

    def _calc_pareto_front(self):
        return load_pareto_front_from_file(f"CTP/{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf")


class CTP3(CTP):
    """
    CTP3 constrained multiobjective optimization problem.
    """

    def __init__(self, n_obj=2, n_var=5, scale_var=False, scale_obj=False):
        super(CTP3, self).__init__(prob_id=3, n_obj=n_obj, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)

    def _calc_pareto_front(self):
        return load_pareto_front_from_file(f"CTP/{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf")


class CTP4(CTP):
    """
    CTP4 constrained multiobjective optimization problem.
    """

    def __init__(self, n_obj=2, n_var=5, scale_var=False, scale_obj=False):
        super(CTP4, self).__init__(prob_id=4, n_obj=n_obj, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)

    def _calc_pareto_front(self):
        return load_pareto_front_from_file(f"CTP/{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf")


class CTP5(CTP):
    """
    CTP5 constrained multiobjective optimization problem.
    """

    def __init__(self, n_obj=2, n_var=5, scale_var=False, scale_obj=False):
        super(CTP5, self).__init__(prob_id=5, n_obj=n_obj, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)

    def _calc_pareto_front(self):
        return load_pareto_front_from_file(f"CTP/{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf")


class CTP6(CTP):
    """
    CTP6 constrained multiobjective optimization problem.
    """

    def __init__(self, n_obj=2, n_var=5, scale_var=False, scale_obj=False):
        super(CTP6, self).__init__(prob_id=6, n_obj=n_obj, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)

    def _calc_pareto_front(self):
        return load_pareto_front_from_file(f"CTP/{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf")


class CTP7(CTP):
    """
    CTP7 constrained multiobjective optimization problem.
    """

    def __init__(self, n_obj=2, n_var=5, scale_var=False, scale_obj=False):
        super(CTP7, self).__init__(prob_id=7, n_obj=n_obj, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)

    def _calc_pareto_front(self):
        return load_pareto_front_from_file(f"CTP/{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf")


class CTP8(CTP):
    """
    CTP8 constrained multiobjective optimization problem.
    """

    def __init__(self, n_obj=2, n_var=5, scale_var=False, scale_obj=False):
        super(CTP8, self).__init__(prob_id=8, n_obj=n_obj, n_var=n_var, scale_var=scale_var, scale_obj=scale_obj)

    def _calc_pareto_front(self):
        return load_pareto_front_from_file(f"CTP/{self.name.lower()}_M{self.n_obj}_D{self.n_var}.pf")

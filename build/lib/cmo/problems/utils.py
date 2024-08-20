import os
import numpy as np
from abc import abstractmethod
from pymoo.core.problem import Problem


class CMOP(Problem):

    def __init__(self, n_var, n_obj, n_iq_constr, n_eq_constr, xl, xu, scale_var=False,
                 scale_obj=False, nadir_point=None, ideal_point=None, constr_eq_eps=1e-4,
                 name=None, **kwargs):

        # No. of constraints
        self.n_constraints = n_iq_constr + n_eq_constr

        # Scale variables, objectives
        self.scale_var = scale_var
        self.scale_obj = scale_obj

        # Nadir and ideal points
        self._nadir_point = nadir_point
        self._ideal_point = ideal_point

        # Equality constraint relaxation
        self.constr_eq_eps = constr_eq_eps

        # Set CMOP name
        self.name = name

        # Create pareto front placeholder
        self._pareto_front = None

        super(CMOP, self).__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=self.n_constraints,
            xl=xl,
            xu=xu
        )

    def nadir_point(self):
        """
        Returns
        -------
        nadir_point : np.array
            The nadir point for a multi-objective problem.

        """
        # if the nadir point has not been calculated yet
        if self._nadir_point is None:

            # calculate the pareto front if not happened yet
            if self._pareto_front is None:
                self._pareto_front = self.pareto_front()

            # if already done, or it was successful - calculate the nadir point
            if self._pareto_front is not None:
                self._nadir_point = np.max(self._pareto_front, axis=0)

        return self._nadir_point

    def ideal_point(self):
        """
        Returns
        -------
        ideal_point : np.array
            The ideal point for a multi-objective problem.

        """
        # if the ideal point has not been calculated yet
        if self._ideal_point is None:

            # calculate the pareto front if not happened yet
            if self._pareto_front is None:
                self._pareto_front = self.pareto_front()

            # if already done, or it was successful - calculate the ideal point
            if self._pareto_front is not None:
                self._ideal_point = np.min(self._pareto_front, axis=0)

        return self._ideal_point

    def _evaluate(self, X, out, *args, **kwargs):

        X = np.atleast_2d(X)

        # If variables are scaled, before evaluation recalculate to normal
        if self.scale_var:
            X = self._unscale_var(X)

        # Evaluate X
        Y = self._fn(X)
        F, G = Y[:, :self.n_obj], Y[:, self.n_obj:]

        if self.scale_obj:
            F = (F - self._ideal_point) / (self._nadir_point - self._ideal_point)

        out["F"] = F
        out["G"] = G

    def _cv(self, x):
        return self.evaluate(x, return_values_of=["CV"])

    def cv(self, x):
        return self.evaluate(x, return_values_of=["CV"])

    def _unscale_var(self, X):
        return self.xl + X * (self.xu - self.xl)

    def unscale_var(self, X):
        return self.xl + X * (self.xu - self.xl)

    def _scale_var(self, X):
        return (X - self.xl) / (self.xu - self.xl)

    @abstractmethod
    def _fn(self, X):
        pass


def load_pareto_front_from_file(fname):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(current_dir, "pf", "%s" % fname)
    if os.path.isfile(fname):
        pf = np.loadtxt(fname)
        return pf[pf[:, 0].argsort()]

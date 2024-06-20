import numpy as np
from scipy.stats import uniform
from pymoo.core.individual import calc_cv

import pygmo as pg


def calculate_cv(G, H):
    """
    Calculate the constraint violation values for a given set of constraints.

    Parameters:
    - G (numpy.ndarray): Array containing the inequality constraint values.
    - H (numpy.ndarray): Array containing the equality constraint values.

    Returns:
    - list: List of constraint violation values.
    """
    return [cv for cv in calc_cv(G, H, config=None) if not np.isnan(cv)]


class ICMOP:
    """
    Class implementing the Indicator for Constrained Multi-Objective Problems (ICMOP).

    Parameters:
    - problem: An instance of a (constrained) multi-objective optimization problem.
    - tau_star (float): A threshold value used in the computation.
    - samples (int): Number of samples for normalizing values.
    - normalize (bool): Flag to determine whether to normalize values.

    Methods:
    - compute(F, CV): Compute the ICMOP value based on objectives (F) and constraint violations (CV).
    """

    def __init__(self, problem, tau_star=1, samples=100, normalize=True):
        """
        Initialize the ICMOP instance.

        Parameters:
        - problem: An instance of a (constrained) multi-objective optimization problem.
        - threshold (float): A threshold value used in the computation (tau).
        - samples (int): Number of samples for normalizing values.
        - normalize (bool): Flag to determine whether to normalize values.
        """
        self.problem = problem
        self.ideal = np.array(problem.ideal_point())
        self.nadir = np.array(problem.nadir_point())
        self.ref_point = np.ones_like(self.nadir)
        self.tau_star = tau_star
        self.samples = samples
        self.normalize = normalize
        self.dist_norm, self.cv_norm = None, None
        if normalize:
            self._get_normalised_values()

    def compute(self, F, CV):
        """
        Compute the ICMOP value based on objectives and constraint violations.

        Parameters:
        - F (numpy.ndarray): Array containing the objective values.
        - CV (numpy.ndarray): Array containing the constraint violation values.

        Returns:
        - float: Computed ICMOP value.
        """

        if len(F) != len(CV):
            raise ValueError('F and CV are not of the same lengths.')

        if 0 not in CV:
            if CV is None or len(CV) == 0:
                raise ValueError('Values for F and CV have not been provided.')

            min_cv = np.min(CV)
            if self.normalize:
                min_cv /= self.cv_norm
            return min_cv + self.tau_star

        F = self._filter_F(F, CV)

        if self._ref_point_domination_check(F):
            F_norm = self._normalise_F(F)
            F_norm = [point for point in F_norm if all(p <= r for p, r in zip(point, self.ref_point))]
            if len(F_norm) == 0:
                imop = 0
                print('imop error')
            else:
                hv = pg.hypervolume(F_norm).compute(self.ref_point)
                imop = -1 * hv

        else:
            distances = self._get_distances(F)
            imop = np.min(distances)
            if self.normalize:
                imop /= self.dist_norm

        return min(imop, self.tau_star)

    def _filter_F(self, F, CV):
        return [f for f, cv in zip(F, CV) if cv == 0.0]

    def _ref_point_domination_check(self, F):
        """
        Check if the reference point dominates the objective values.

        Parameters:
        - F (numpy.ndarray): Array containing the normalized objective values.

        Returns:
        - bool: True if the reference point dominates, False otherwise.
        """
        return any(all(self.nadir[i] >= f[i] for i in range(len(self.nadir))) for f in F)

    def _get_distances(self, F):
        """
        Calculate distances from the reference point to the normalized objective values.

        Parameters:
        - F (numpy.ndarray): Array containing the normalized objective values.

        Returns:
        - numpy.ndarray: Array containing the distances.
        """
        distances = []
        for f in F:
            is_ds = True
            for i in range(len(self.nadir)):
                if self.nadir[i] > f[i]:
                    is_ds = False

            if is_ds:
                distances.append(np.linalg.norm(f - self.nadir))
            else:
                D = [d for d in f - self.nadir if d > 0]
                distances.append(np.min(D) if len(D) > 0 else 0)

        distances = [d for d in distances if not np.isnan(d)]
        return np.array(distances)

    def _normalise_F(self, F):
        """
        Normalize the objective values.

        Parameters:
        - F (numpy.ndarray): Array containing the objective values.

        Returns:
        - numpy.ndarray: Array containing the normalized objective values.
        """
        denominator = 1e-10 if np.array_equal(self.nadir, self.ideal) else self.nadir - self.ideal
        return np.array([(f - self.ideal) / denominator for f in F])

    def _get_normalised_values(self):
        """
        Generate normalized values for distance and constraint violation.

        Returns:
        - None
        """
        lhs = uniform.rvs(size=(self.samples, self.problem.n_var))
        scaled_lhs = self.problem.xl + lhs * (self.problem.xu - self.problem.xl)
        F, G, H = self.problem.evaluate(scaled_lhs, return_values_of=['F', 'G', 'H'])

        CV = calculate_cv(G, H)
        D = self._get_distances(F)

        med_D, med_CV = np.median(D), np.median(CV)

        dist_norm = 1 if med_D == 0 or np.isnan(med_D) else 10 ** np.ceil(np.log(med_D))
        cv_norm = 1 if med_CV == 0 or np.isnan(med_CV) else 10 ** np.ceil(np.log(med_CV))

        self.dist_norm, self.cv_norm = dist_norm, cv_norm

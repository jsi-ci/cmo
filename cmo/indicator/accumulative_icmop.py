import numpy as np
from scipy.stats import uniform
from pymoo.core.individual import calc_cv
from moarchiving import BiobjectiveNondominatedSortedList as NDA
from decimal import *

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

class AccumulativeICMOP:
    """
    Class for implementing the Indicator for Constrained Multi-Objective Problems (ICMOP).

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
        - tau_star (float): A threshold value used in the computation.
        - samples (int): Number of samples for normalizing values.
        - normalize (bool): Flag to determine whether to normalize values.
        """
        if problem.n_obj > 2:
            raise ValueError('Class only works with biobjective problems, due to the use of moarchiving.')

        self.problem = problem
        self.ideal = np.array(problem.ideal_point())
        self.nadir = np.array(problem.nadir_point())
        self.ref_point = np.ones_like(self.nadir)
        self.tau_star = Decimal(tau_star)
        self.samples = samples
        self.normalize = normalize
        self.dist_norm, self.cv_norm = None, None
        self.nda = NDA(reference_point=self.ref_point)
        self.constraint_value = None
        self.min_dist = None
        if normalize:
            self._get_normalised_values()

    def min_dist_check(self, F):
        if len(self.nda) == 0:
            distances = self._get_distances(F)
            min_dist = np.min(distances)
            if self.normalize:
                min_dist /= self.dist_norm

            if self.min_dist is None or min_dist < self.min_dist:
                self.min_dist = min_dist

    def add(self, F, CV):
        """
        Add a single solution to the ICMOP instance.

        Parameters:
        - F (numpy.ndarray): Array containing the objective values for the solution.
        - CV (float): Constraint violation value for the solution.
        """
        if np.array(F).ndim > 1:
            raise ValueError('F should only represent one solution.')
        if isinstance(CV, (list, tuple, set, dict, np.ndarray)):
            raise ValueError('CV should be a single value.')

        CV = Decimal(CV)
        if CV != 0:
            if self.normalize:
                CV /= self.cv_norm
            cv_norm = CV + self.tau_star
            if self.constraint_value is None or cv_norm < self.constraint_value:
                self.constraint_value = cv_norm
            return

        self.constraint_value = self.tau_star
        self.nda.add(self._normalise_f(F))
        self.min_dist_check([F])

    def add_list(self, F, CV):
        """
        Add multiple solutions to the ICMOP instance.

        Parameters:
        - F (numpy.ndarray): Array containing the objective values.
        - CV (numpy.ndarray): Array containing the constraint violation values.
        """
        if len(F) != len(CV):
            raise ValueError('F and CV are not of the same lengths.')

        if 0 not in CV:
            if CV is None or len(CV) == 0:
                raise ValueError('Values for F and CV have not been provided.')

            min_cv = np.min(CV)
            min_cv = Decimal(min_cv)
            if self.normalize:
                min_cv /= self.cv_norm
            cv_norm = min_cv + self.tau_star
            if self.constraint_value is None or cv_norm < self.constraint_value:
                self.constraint_value = cv_norm
            return

        self.constraint_value = self.tau_star
        F = self._filter_F(F, CV)
        self.nda.add_list(self._normalise_F(F))
        self.min_dist_check(F)

    def compute(self):
        """
        Compute the ICMOP value.

        Returns:
        - float: The computed ICMOP value.
        """
        if self.constraint_value > self.tau_star:
            return self.constraint_value

        if len(self.nda) > 0:
            imop = -1 * self.nda.hypervolume
        else:
            imop = self.min_dist

        return min(imop, self.tau_star)

    def _filter_F(self, F, CV):
        """
        Filter objective values based on constraint violations.

        Parameters:
        - F (numpy.ndarray): Array containing the objective values.
        - CV (numpy.ndarray): Array containing the constraint violation values.

        Returns:
        - list: Filtered objective values.
        """
        return [f for f, cv in zip(F, CV) if cv == 0.0]

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

    def _normalise_f(self, f):
        """
        Normalize a single objective value.

        Parameters:
        - f (numpy.ndarray): Array containing the objective value.

        Returns:
        - numpy.ndarray: Normalized objective value.
        """
        denominator = 1e-10 if np.array_equal(self.nadir, self.ideal) else self.nadir - self.ideal
        return (f - self.ideal) / denominator

    def _normalise_F(self, F):
        """
        Normalize multiple objective values.

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

        self.dist_norm, self.cv_norm = dist_norm, Decimal(cv_norm)

import os
import numpy as np
from cmop.utils import CMOP, load_pareto_front_from_file

__all_ = ['CS1', 'CT1', 'CTS1', 'CTSE1', 'CTSEI1',
          'CS2', 'CT2', 'CTS2', 'CTSE2', 'CTSEI2',
          'CS3', 'CT3', 'CTS3', 'CTSE3', 'CTSEI3',
          'CS4', 'CT4', 'CTS4', 'CTSE4', 'CTSEI4']


class MODAct(CMOP):
    """Multi-Objective Design of Actuators

    MODAct is a framework for real-world constrained multi-objective optimization.
    Refer to the python package https://github.com/epfl-lamd/modact from requirements.

    Best-known Pareto fronts must be downloaded from here: https://doi.org/10.5281/zenodo.3824302

    Parameters
    ----------

    prob_id: str
        The name of the benchmark problem to use either as a string. Example values: cs1, cs3, ct2, ct4, cts3

    References:
    ----------
    C. Picard and J. Schiffmann, “Realistic Constrained Multi-Objective Optimization Benchmark Problems from Design,”
    IEEE Transactions on Evolutionary Computation, pp. 1–1, 2020.
    """

    def __init__(self, prob_id, **kwargs):
        try:
            import modact.problems as pb
        except ImportError:
            raise Exception("Please install the modact library: https://github.com/epfl-lamd/modact")

        self.prob_id = prob_id
        self.fct = pb.get_problem(prob_id)

        xl, xu = self.fct.bounds()
        n_var = len(xl)
        n_obj = len(self.fct.weights)
        n_iq_constr = len(self.fct.c_weights)

        self.weights = np.array(self.fct.weights)
        self.c_weights = np.array(self.fct.c_weights)

        super().__init__(n_var=n_var, n_obj=n_obj, n_iq_constr=n_iq_constr, n_eq_constr=0, xl=xl, xu=xu, vtype=float,
                         **kwargs)

    def _fn(self, X, *args, **kwargs):
        F, G = [], []
        for x in X:
            f, g = self.fct(x)
            F.append(np.array(f) * -1 * self.weights)
            G.append(np.array(g) * self.c_weights)
        return np.column_stack([F, G])

    def _calc_pareto_front(self, **kwargs):
        return load_pareto_front_from_file(os.path.join("MODACT", f"{self.prob_id}.pf"))


class CS1(MODAct):
    def __init__(self):
        super(CS1, self).__init__(prob_id='cs1')


class CS2(MODAct):
    def __init__(self):
        super(CS2, self).__init__(prob_id='cs2')


class CS3(MODAct):
    def __init__(self):
        super(CS3, self).__init__(prob_id='cs3')


class CS4(MODAct):
    def __init__(self):
        super(CS4, self).__init__(prob_id='cs4')


class CT1(MODAct):
    def __init__(self):
        super(CT1, self).__init__(prob_id='ct1')


class CTS1(MODAct):
    def __init__(self):
        super(CTS1, self).__init__(prob_id='cts1')


class CTSE1(MODAct):
    def __init__(self):
        super(CTSE1, self).__init__(prob_id='ctse1')


class CTSEI1(MODAct):
    def __init__(self):
        super(CTSEI1, self).__init__(prob_id='ctsei1')


class CT2(MODAct):
    def __init__(self):
        super(CT2, self).__init__(prob_id='ct2')


class CTS2(MODAct):
    def __init__(self):
        super(CTS2, self).__init__(prob_id='cts2')


class CTSE2(MODAct):
    def __init__(self):
        super(CTSE2, self).__init__(prob_id='ctse2')


class CTSEI2(MODAct):
    def __init__(self):
        super(CTSEI2, self).__init__(prob_id='ctsei2')


class CT3(MODAct):
    def __init__(self):
        super(CT3, self).__init__(prob_id='ct3')


class CTS3(MODAct):
    def __init__(self):
        super(CTS3, self).__init__(prob_id='cts3')


class CTSE3(MODAct):
    def __init__(self):
        super(CTSE3, self).__init__(prob_id='ctse3')


class CTSEI3(MODAct):
    def __init__(self):
        super(CTSEI3, self).__init__(prob_id='ctsei3')


class CT4(MODAct):
    def __init__(self):
        super(CT4, self).__init__(prob_id='ct4')


class CTS4(MODAct):
    def __init__(self):
        super(CTS4, self).__init__(prob_id='cts4')


class CTSE4(MODAct):
    def __init__(self):
        super(CTSE4, self).__init__(prob_id='ctse4')


class CTSEI4(MODAct):
    def __init__(self):
        super(CTSEI4, self).__init__(prob_id='ctsei4')

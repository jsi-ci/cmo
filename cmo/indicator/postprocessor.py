from cmo.indicator.erd import ERD
from cmo.indicator.icmop import ICMOP


class PostProcessor:
    """
    PostProcessor class for analyzing optimization runs, computing runtime distributions,
    and evaluating the performance of optimization algorithms using the ECDF (Empirical Cumulative Distribution Function)
    and ICMOP (Indicator for Constrained Multi-Objective Problems) metrics.

    Attributes:
    - runs (list): A list of optimization run data.
        Format: runs = [run, run, ...], where each run = [eval, eval, ...] and each eval = [f, cv].
    - pop_size (int): Population size used in the optimization algorithm.
    - problem: The optimization problem instance being solved.
    - tau_ref (float): Reference target value for ECDF computation.
    - epsilon (list of floats): Incremental values added to tau_ref for target computations in ECDF.
    - samples (int, optional): Number of samples for normalization in ICMOP. Defaults to 100.
    - normalize (bool, optional): Flag to indicate if normalization is to be applied in ICMOP. Defaults to True.
    - n_runs (int): Number of runs.
    - n_evals (int): Number of evaluations per run.

    Methods:
    - __init__: Constructor for initializing the PostProcessor instance with specific parameters for analysis.
    - _process_runs: Processes the runs to compute runtimes, initialize ECDF processor, and compute AUC for ECDF.
    - _get_runtime: Computes the runtime for a single run by evaluating ICMOP for each generation.
    - get_ecdf: Returns the ECDF computed from the runtimes.
    - get_auc_ecdf: Returns the area under the curve (AUC) of the ECDF.
    - get_runtime: Returns the collated runtime data.
    """

    def __init__(self, runs, pop_size, problem, tau_ref, epsilon, tau_star, samples=100, normalize=True):
        """
        Initializes the PostProcessor with the given parameters for computing initial runtimes and ECDF.

        Parameters:
        - runs (list): List of runs to be post-processed.
        - pop_size (int): Population size of the optimization algorithm.
        - problem: The optimization problem being solved.
        - tau_ref (float): Reference target value for ECDF computation.
        - epsilon (list of floats): Incremental values to be added to tau_ref for target computations.
        - tau_star (float): Target value for ICMOP computations.
        - samples (int, optional): Number of samples for ICMOP normalization. Defaults to 100.
        - normalize (bool, optional): Whether to apply normalization for ICMOP computations. Defaults to True.
        """
        # Input validation
        if not runs or pop_size <= 0:
            raise ValueError("Invalid 'runs' list or 'pop_size'")

        # Initialization of attributes
        self.runs = runs
        self.pop_size = pop_size
        self.problem = problem
        self.tau_ref = tau_ref
        self.epsilon = epsilon

        # Computing basic stats
        self.n_runs = len(runs)
        self.n_evals = len(runs[0]) if runs else 0

        self.icmop = ICMOP(problem, tau_star, samples, normalize)

        # Placeholder for computed attributes
        self.runtimes, self.ecdf_processor, self.collated_runtime, self.auc_ecdf = None, None, None, None

        # Process runs to compute metrics
        self._process_runs()

    def _process_runs(self):
        """
        Processes all runs to compute runtime distributions and ECDF metrics.
        """
        # Compute runtime for each run
        self.runtimes = [self._get_runtime(run) for run in self.runs]
        # Initialize ECDF processor with computed runtimes and target specifications
        self.ecdf_processor = ERD(self.runtimes, self.tau_ref, self.epsilon)
        # Collate runtime data
        self.collated_runtime = self.ecdf_processor.runtime
        # Compute AUC for ECDF
        self.auc_ecdf = self.ecdf_processor.compute_auc()

    def _get_runtime(self, run):
        """
        Computes the runtime for a given run by segmenting it into generations and computing ICMOP for each generation.

        Parameters:
        - run (list): The run data to process.

        Returns:
        - list: A list of computed runtimes for each generation in the run.
        """
        # Segment run into generations based on population size
        generations = [run[i:i + self.pop_size] for i in range(0, len(run), self.pop_size)]
        # Compute runtime for each generation using ICMOP
        return [self.icmop.compute([gen_eval[0] for gen_eval in gen], [gen_eval[1] for gen_eval in gen]) for gen in generations]

    def get_ecdf(self):
        """
        Returns the ECDF computed from the runtimes.

        Returns:
        - numpy.ndarray: The computed ECDF values.
        """
        return self.ecdf_processor.erd

    def get_auc_ecdf(self):
        """
        Returns the area under the curve (AUC) of the ECDF.

        Returns:
        - float: The AUC of the ECDF.
        """
        return self.auc_ecdf

    def get_runtime(self):
        """
        Returns the collated runtime data.

        Returns:
        - numpy.ndarray: The collated runtime data.
        """
        return self.collated_runtime

from pymoo.core.callback import Callback

class RuntimeCallback(Callback):
    """
    Callback class for tracking and recording runtime values during optimization.

    Parameters:
    - icmop: An instance of the ICMOP class for computing runtime scores.

    Methods:
    - notify(algorithm): Callback method to be called by the optimization algorithm.
    """

    def __init__(self, icmop) -> None:
        """
        Initialize the RuntimeCallback instance.

        Parameters:
        - icmop: An instance of the ICMOP class for computing runtime scores.
        """
        super().__init__()
        self.icmop = icmop
        self.runtime = []

    def notify(self, algorithm):
        """
        Callback method to be called by the optimization algorithm.

        Parameters:
        - algorithm: The optimization algorithm instance.

        Returns:
        - None
        """
        CV = algorithm.pop.get('CV')
        F = algorithm.pop.get('F')

        score = self.icmop.compute(F, CV)
        self.runtime.append(score)

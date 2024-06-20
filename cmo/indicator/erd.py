import numpy as np
import matplotlib.pyplot as plt


class ERD:
    """
    Empirical Cumulative Distribution Function (ERD) analysis for runtime data.

    Parameters:
    - runtimes (numpy.ndarray): 2D Array where each row contains runtime data for a specific configuration.
    - tau_ref (float): Reference value for the starting target computation.
    - epsilons (list of floats): Incremental values to be added to tau_ref for target computations.

    Methods:
    - compute_auc(): Compute the Area Under the ERD Curve (AUC).
    - visualise(title=None, show=True, save=False, path=None): Visualize the runtime and ERD curves.
    """

    def __init__(self, runtimes, tau_ref, eps):
        """
        Initialize the ERD instance with runtime data and target computation parameters.

        Parameters:
        - runtimes (numpy.ndarray): 2D Array where each row contains runtime data for a specific configuration.
        - tau_ref (float): Reference value for the starting target computation.
        - epsilons (list of floats): Incremental values to be added to tau_ref for target computations.
        """
        self.runtimes = runtimes
        self.eps = eps
        self.tau_ref = tau_ref

        self.runtime, self.erd = None, None
        self._aggregate_runtimes()
        self._compute_erd()

    def _aggregate_runtimes(self):
        """
        Aggregate runtimes by computing the mean of each configuration's runtimes.

        This method updates the 'runtime' attribute with the aggregated runtime data.
        """
        self.runtime = np.array([np.mean(lst) + self.tau_ref for lst in np.transpose(self.runtimes)])

    def _compute_erd(self):
        """
        Compute the Empirical Cumulative Distribution Function (ERD) values based on the aggregated runtimes and targets.

        This method updates the 'erd' attribute with the computed ERD values.
        """
        erd = [0]
        for r in self.runtime:
            current = erd[-1]
            for t in self.eps[erd[-1]:]:
                if r <= t:
                    current += 1
                else:
                    break
            erd.append(current)
        erd_norm = [val / len(self.eps) for val in erd]
        self.erd = erd_norm

    def compute_auc(self):
        """
        Compute the Area Under the ERD Curve (AUC).

        Returns:
        - float: The computed AUC value.
        """
        x = np.linspace(0, len(self.runtime), num=len(self.runtime) + 1) / len(self.runtime)
        return np.trapz(self.erd, x)

    def visualise(self, title=None, show=True, save=False, path=None):
        """
        Visualize the runtime and ERD curves with options to show and/or save the plot.

        Parameters:
        - title (str, optional): Title for the plot.
        - show (bool, optional): If True, display the plot.
        - save (bool, optional): If True, save the plot to the specified path.
        - path (str, optional): Path to save the plot (required if save is True).
        """
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xscale('log')  # Set x-axis to log scale for better visualization
        ax1.set_xlabel('log(Generations)')
        ax1.set_ylabel('Runtime', color=color)
        ax1.plot(self.runtime, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # Create another y-axis for the ERD
        color = 'tab:blue'
        ax2.set_ylabel('ERD', color=color)
        ax2.plot(self.erd, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axhline(y=0.5, color='green', linestyle='--', label='ERD = 0.5')

        if title:
            plt.suptitle(title)
        fig.tight_layout()

        if save:
            if not path:
                raise ValueError('Please provide a path for the visualisation.')
            plt.savefig(path)

        if show:
            plt.show()

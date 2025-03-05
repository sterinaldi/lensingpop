import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from tqdm import tqdm
import dill
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


class funInterpolator:
    """
    A unified interpolator that handles both 1D and 2D function interpolation.

    This class uses `RegularGridInterpolator` to interpolate functions defined on 1D or 2D grids.
    
    Attributes:
        I (RegularGridInterpolator): The interpolator instance.
        dim (int): The dimensionality of the input data (1D or 2D).
    """
    def __init__(self, points, f):
        """
        Initialize the interpolator.

        Parameters:
            points (tuple or np.ndarray): Grid points for interpolation.
                - If 1D, provide a single array.
                - If 2D, provide a tuple of two arrays.
            f (np.ndarray): Function values at the grid points.
        """
        # Determine the dimensionality of the interpolation
        self.dim = 1 if isinstance(points, np.ndarray) else len(points)
        
        # Ensure that points are wrapped correctly in a tuple for RegularGridInterpolator
        if self.dim == 1:
            points = (points,)  # Convert 1D array into a tuple
        
        # Initialize the interpolator
        self.I = RegularGridInterpolator(points, f, fill_value=0., bounds_error=False)

    def __call__(self, x):
        """
        Allows the instance to be called as a function to evaluate the interpolation.

        Parameters:
            x (np.ndarray): Input values (1D or 2D).

        Returns:
            np.ndarray: Interpolated function values.
        """
        return self.pdf(x)
    
    def pdf(self, x):
        """
        Evaluate the interpolated function at given points.

        Parameters:
            x (np.ndarray): Input points (1D or 2D array).

        Returns:
            np.ndarray: Interpolated values at given points.
        """
        if self.dim == 1:
            return self.I((x,))  # Ensure single-variable input is wrapped correctly
        return self.I(x)  # Directly use for 2D inputs

class KDEEstimator:
    """
    A Kernel Density Estimator (KDE) for estimating probability density functions from sample data.

    This class uses Gaussian Kernel Density Estimation (KDE) for probability density estimation.

    Attributes:
        I (gaussian_kde or None): The KDE instance if fitted, otherwise None.
    """
    def __init__(self, samples=None):
        """
        Initialize the KDEEstimator with optional samples.

        Parameters:
            samples (np.ndarray, optional): The data samples to fit the KDE.
        """
        if samples is not None:
            self.fit(samples)
        else:
            self.I = None  # KDE is not initialized yet

    def fit(self, samples):
        """
        Fit the KDE using the provided samples.

        Parameters:
            samples (np.ndarray): A 1D or 2D array of samples.
                - If 1D, it will be reshaped to (n_samples, 1).
                - If 2D, it should be of shape (n_samples, n_features).
        """
        samples = np.atleast_2d(samples)  # Ensure 2D format
        if samples.shape[0] < samples.shape[1]:  # Ensure samples are (n_samples, n_features)
            samples = samples.T  # Transpose if necessary
        self.I = gaussian_kde(samples.T)  # Fit KDE

    def pdf(self, x):
        """
        Evaluate the probability density function (PDF) at given points.

        Parameters:
            x (np.ndarray): A 1D or 2D array of points where to evaluate the PDF.

        Returns:
            np.ndarray: The PDF values at the given points.
        """
        x = np.atleast_2d(x)  # Ensure 2D format
        return self.I(x.T)  # Evaluate KDE PDF

    def save(self, filename):
        """
        Save the KDE estimator to a file using dill.

        Parameters:
            filename (str): The file path to save the estimator.
        """
        with open(filename, "wb") as f:
            dill.dump(self, f)
        print(f"KDE estimator saved to {filename}")

    @staticmethod
    def load(filename):
        """
        Load a KDE estimator from a file.

        Parameters:
            filename (str): The file path to load the estimator from.

        Returns:
            KDEEstimator: The loaded KDE estimator instance.
        """
        with open(filename, "rb") as f:
            estimator = dill.load(f)
        print(f"KDE estimator loaded from {filename}")
        return estimator

def roc(lensing, nlensing, bench=False):
    x = 10**np.linspace(-3, 2, 300)
    blu_l, dblu_l = lensing.T
    blu_nl, dblu_nl = nlensing.T    

    if bench:
        # Benchmark mode
        pos = []
        neg = []

        for th in x:
            tp = (blu_l > th).sum()
            fn = (blu_l <= th).sum()
            fp = (blu_nl > th).sum()
            tn = (blu_nl <= th).sum()
            pos.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            neg.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

        return pos, neg

    # Monte Carlo mode
    n_iter = 100
    pos = np.zeros((n_iter, len(x)))
    neg = np.zeros((n_iter, len(x)))

    for j in range(n_iter):
        dx = np.random.normal(0, dblu_l)  # Noise for lensing
        dy = np.random.normal(0, dblu_nl)  # Noise for non-lensing

        for i, th in enumerate(x):
            tp = ((blu_l + dx) > th).sum()
            fn = ((blu_l + dx) <= th).sum()
            fp = ((blu_nl + dy) > th).sum()
            tn = ((blu_nl + dy) <= th).sum()

            pos[j, i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            neg[j, i] = fp / (fp + tn) if (fp + tn) > 0 else 0

    return pos, neg

def interp_roc(ax, xs, ys):
    
    mean_x_axis = 10**np.linspace(-4,0,300)
    ys_interp = np.zeros((len(xs),len(mean_x_axis)))
    for i in range(len(xs)):
        f = interp1d( xs[i], ys[i],fill_value=1.0,bounds_error=False)
        ys_interp[i] = f(mean_x_axis)
    max_y_axis = np.max(ys_interp, axis=0)
    min_y_axis = np.min(ys_interp, axis=0)

    return mean_x_axis, max_y_axis, min_y_axis

# Define function for generating the ROC plot
def roc_plot(ax, blu_l, blu_ul, label, color, fill=False, alpha=0.5, lw=2, ls=None):
    pos, neg = roc(blu_l, blu_ul)
    if fill:
        x, y1, y2 = interp_roc(ax, neg, pos)
        ax.fill_between(x, y1, y2, label=label, color=color, alpha=alpha)
    else:
        ax.plot(neg[0], pos[0], label=label, color=color, lw=lw, ls=ls)


def marginal1d_plot(
    column, ax, draws, dim, true_dist, 
    wrong_dist=None, labels=None, colors=None, figsize=7, 
    levels=[0.5, 0.68, 0.9], n_pts=200, xlim=None
):
    """
    Plot the marginal probability distribution for a selected column variable.

    Parameters:
        column (int): The column index for marginalization.
        ax (matplotlib.axes.Axes): The axis to plot the results.
        draws (iterable): List of distribution draws (e.g., samples from models).
        dim (int): Dimensionality of the distribution.
        true_dist (function): True distribution function for comparison.
        wrong_dist (iterable or None): Alternative model or distribution for comparison.
        labels (list or None): Labels for the different distributions.
        colors (list or None): Colors for the distributions.
        figsize (float): Size of the figure.
        levels (list): Credible region levels.
        n_pts (numpy.ndarray): Number of points for the grid.
        xlim (list or None): Limits for the x-axis.

    Returns:
        None
    """
    levels = np.atleast_1d(levels)
    mlim = [5, 180]   # Mass limits
    if xlim is None:
        xlim = mlim
    
    #m = np.linspace(mlim[0], mlim[1], n_pts[0]+2)[1:-1]
    
    m = np.linspace(xlim[0], xlim[1], n_pts)
    dm = m[1] - m[0]
    # Plot true distribution
    probs = true_dist.pdf(m).reshape(n_pts)
    probs[np.isnan(probs)] = 0
    norm = probs.sum() * dm
    ax.plot(m, probs / norm, lw=1.5, color='black', label=r'$\mathrm{Benchmark}$')
    
    # Plot sampled distributions if provided
    if np.iterable(draws):
        probs = np.array([(d.pdf(m).reshape(n_pts)) for d in tqdm(draws, total=len(draws), desc='dist')])
        #probs = np.array([m.sum(axis=tuple(dims)) * np.prod([dm[k] for k in dims]) for m in astro_dists])
        
        # Compute credible regions
        percentiles = [50, 5, 16, 84, 95]
        p = {perc: np.percentile(probs, perc, axis=0) for perc in percentiles}
        norm = p[50].sum() * dm
        for perc in percentiles:
            p[perc] = p[perc] / norm
        
        # Fill credible regions and plot median
        ax.fill_between(m, p[95], p[5], color='mediumturquoise', alpha=0.5)
        ax.fill_between(m, p[84], p[16], color='darkturquoise', alpha=0.5)
        ax.plot(m, p[50], lw=0.7, color=colors[0], label=labels[0]) 
    
    # Plot wrong distribution if provided
    if wrong_dist is not None:
        if np.iterable(wrong_dist):
            probs = np.array([(d.pdf(m).reshape(n_pts)) for d in tqdm(wrong_dist, total=len(draws), desc='dist')])
            #probs = np.array([m.sum(axis=tuple(dims)) * np.prod([dm[k] for k in dims]) for m in astro_dists])
            p = {perc: np.percentile(probs, perc, axis=0) for perc in percentiles}
            norm = p[50].sum() * dm
            for perc in percentiles:
                p[perc] = p[perc] / norm
            
            ax.fill_between(m, p[95], p[5], color='papayawhip', alpha=0.5)
            ax.fill_between(m, p[84], p[16], color='moccasin', alpha=0.5)
            ax.plot(m, p[50], lw=0.7, color=colors[1], label=labels[1])
        else:
            probs = wrong_dist.pdf(m) * np.ones(m.shape)
            norm = (m[-1] - m[0])
            ax.plot(m, probs, lw=1.5, color=colors[2], label=labels[1])
    
    # Customize plot appearance
    ax.set_ylim(0.001, 0.021)
    ax.set_yticks([])
    ax.yaxis.set_major_locator(MaxNLocator(3)) 
    ax.set_ylabel(r'$p(m_1^z)$', fontsize=18)
    ax.set_xlim(xlim[0], xlim[1])
    ax.xaxis.set_minor_locator(MultipleLocator(5)) 
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=18)



def plotting_diff(m1, d, d1, d2, model, true_model, wrong_model, marker_color, colors, labels, 
                  ylabel, xlim=None, ylim=None, yscale='log', ticks=[20, 40, 60, 80, 100]):
    """
    Compare two datasets (benchmark vs. model) and plot marginal differences.
    
    Parameters:
        m1 (array-like): The x-axis values (e.g., primary mass values).
        d1 (array-like): Benchmark data as a 2D array [value, error].
        d2 (array-like): Model data as a 2D array [value, error].
        model (object): The sampled/model distribution object.
        true_model (object): The true reference distribution/model.
        wrong_model (object): Alternative model for comparison.
        colors (list): Colors for plotting benchmark and model data.
        labels (list): Labels for benchmark and model datasets.
        ylabel (str): Label for the y-axis.
        xlim (tuple or None): Limits for the x-axis as (min, max).
        ylim (tuple or None): Limits for the y-axis as (min, max).
        yscale (str): Scale for the y-axis ('linear' or 'log').
        ticks (list): Ticks for the x-axis.

    Returns:
        tuple: A tuple containing the matplotlib `Figure` and `Axes` objects.
    """
    # Create subplots with shared x-axis
    fig, axes = plt.subplots(
        nrows=2, sharex=True, figsize=(10, 10), 
        gridspec_kw={'height_ratios': [1, 3]}
    )
    fig.subplots_adjust(0, 0, 1, 1, 0, 0)
    
    x = (d1[:, 0]-d[:, 0])/d[:, 0] *100
    a = d1[:, 0]
    b = d[:, 0]
    sigma_a = d1[:, 1]
    sigma_b = d[:, 1]
    dx = np.sqrt((sigma_a / b)**2 + ((-a / b**2 + 1 / b) * sigma_b)**2) *100
    #dx = (d1[:, 0]/d[:, 0])*np.sqrt( (d[:, 1]/d[:, 0])**2 + (d1[:, 1]/d1[:, 0])**2 ) *100

    # Plot benchmark data
    axes[1].errorbar(
        m1, x, dx, solid_capstyle='projecting', 
        color=marker_color[0], alpha=0.8, capsize=5, fmt='o', label=labels[0]
    )
    
    
    x = (d2[:, 0]-d[:, 0])/d[:, 0] *100
    a = d2[:, 0]
    b = d[:, 0]
    sigma_a = d2[:, 1]
    sigma_b = d[:, 1]
    dx = np.sqrt((sigma_a / b)**2 + ((-a / b**2 + 1 / b) * sigma_b)**2) *100
    #dx = (d2[:, 0]/d[:, 0])*np.sqrt( (d[:, 1]/d[:, 0])**2 + (d2[:, 1]/d2[:, 0])**2 ) *100

    # Plot model data
    axes[1].errorbar(
        m1, x, dx, solid_capstyle='projecting', 
        color=marker_color[1], alpha=0.8, capsize=5, fmt='o', label=labels[1]
    )
    
    # Plot marginal distribution on the top subplot
    marginal1d_plot(0, axes[0], model, 2, true_model, wrong_model, labels=labels, colors=colors)

    # Customize the bottom subplot
    ax = axes[1]
    ax.set_ylabel(r'' + ylabel, fontsize=20)
    ax.set_xlabel(r'$m_1^z [M_{\odot}]$', fontsize=20)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlim(xlim[0], xlim[1])
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    #plt.yscale(yscale)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(fontsize=18)
    
    return fig, ax

def select_idx(x, interval = 10):
    # Sort the array and keep track of indices
    sorted_indices = np.argsort(x)
    sorted_m1z = np.array(x)[sorted_indices]

    max_val = np.max(x)
    interval_indices = []

    # Pick one value per interval
    for start in range(0, int(max_val) + 1, interval):
        # Find elements in the current interval
        in_interval = np.where((sorted_m1z >= start) & (sorted_m1z < start + interval))[0]
        if len(in_interval) > 0:
            # Pick the first one and get its original index
            interval_indices.append(sorted_indices[in_interval[0]])
    return interval_indices


def select_idx_2d(m1z, q, dm1z=10, dq=0.1):
    """
    Select one sample from each 2D grid defined by dm1z and dq.
    
    Args:
        m1z (array-like): Primary mass values.
        q (array-like): Mass ratio values.
        dm1z (float): Interval size for m1z grid.
        dq (float): Interval size for q grid.
    
    Returns:
        list: Indices of selected samples from each 2D grid cell.
    """
    # Ensure inputs are numpy arrays
    m1z = np.array(m1z)
    q = np.array(q)
    
    # Sort by m1z and q while keeping track of indices
    sorted_indices = np.lexsort((q, m1z))
    sorted_m1z = m1z[sorted_indices]
    sorted_q = q[sorted_indices]
    
    # Determine grid bounds
    max_m1z = np.max(m1z)
    max_q = np.max(q)
    
    selected_indices = []
    
    # Iterate over 2D grid cells
    for m1z_start in range(0, int(max_m1z) + 1, dm1z):
        for q_start in np.arange(0, max_q + dq, dq):
            # Find elements in the current grid cell
            in_grid = np.where(
                (sorted_m1z >= m1z_start) & (sorted_m1z < m1z_start + dm1z) &
                (sorted_q >= q_start) & (sorted_q < q_start + dq)
            )[0]
            
            if len(in_grid) > 0:
                # Pick the first one in the grid cell
                selected_indices.append(sorted_indices[in_grid[0]])
    
    return selected_indices

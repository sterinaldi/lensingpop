import numpy as np
from scipy.stats import skewnorm
from scipy.special import erf
import dill
from ..population import simulated_universe as universe
from ..population import sampler 

###########################################################################################################
""" 
This script generates the chirp mass posterior by following the procedure in 
Appendix A of https://iopscience.iop.org/article/10.3847/2041-8213/ab77c9/pdf. 
The mass ratio (q) is sampled from a skew-normal distribution to account for 
measurement uncertainties in gravitational wave observations.
"""

################# Uncertainty parameters ##################
# SNR threshold for event detection, based on 70 O3 gravitational wave events
snr_threshold  = 8

# Standard deviations for chirp mass and mass ratio uncertainty, scaled with SNR
sigma_mass = 0.11 * snr_threshold  # Uncertainty in chirp mass
sigma_q = 0.22 * snr_threshold  # Uncertainty in mass ratio

# Skewness parameter for mass ratio distribution
skew_alpha = -0.0066


def compute_loc_from_mode(x_true, alpha, scale):
    """
    Computes the location parameter (loc) for a skew-normal distribution given the mode.

    Parameters:
    - x_true (float): The mode of the skew-normal distribution.
    - alpha (float): The skewness parameter.
    - scale (float): The scale (spread) of the distribution.

    Returns:
    - float: The computed location parameter (xi) for the skew-normal distribution.
    """
    delta = alpha / np.sqrt(1 + alpha**2)  # Skewness-related adjustment
    loc = x_true - scale * delta  # Compute location parameter
    return loc


def truncated_skewnorm(alpha, scale, x_true, lower_bound, upper_bound, size=1000):
    """
    Generates samples from a truncated skew-normal distribution.

    Parameters:
    - alpha (float): Skewness parameter of the skew-normal distribution.
    - scale (float): Scale (spread) of the distribution.
    - x_true (float): Mode of the distribution.
    - lower_bound (float): Lower truncation limit.
    - upper_bound (float): Upper truncation limit.
    - size (int): Number of samples to generate (default: 1000).

    Returns:
    - np.array: Array of samples from the truncated skew-normal distribution.
    """
    loc = compute_loc_from_mode(x_true, alpha, scale)  # Compute location parameter
    samples = []
    
    while len(samples) < size:
        sample = skewnorm.rvs(alpha, loc=loc, scale=scale)  # Generate sample
        
        # Accept the sample only if it falls within bounds
        if lower_bound <= sample <= upper_bound:
            samples.append(sample)
    
    return np.array(samples)


def genPosterior(Mc_z, q, z, snr_obs, alpha=-0.0066, n=1000):
    """
    Generates a set of posterior samples incorporating measurement uncertainty 
    for observed binary systems.

    Parameters:
    - Mc_z (float): Redshifted chirp mass.
    - q (float): True mass ratio.
    - z (float): Redshift.
    - snr_obs (float): Observed signal-to-noise ratio (SNR).
    - alpha (float, optional): Skewness parameter for mass ratio distribution (default: -0.0066).
    - n (int, optional): Number of samples to generate (default: 1000).

    Returns:
    - tuple (np.array, np.array, np.array, np.array):
        - m1z_obs: Observed primary mass (redshifted).
        - q_obs: Observed mass ratio.
        - Mc_obs: Observed chirp mass.
        - symratio_obs: Symmetric mass ratio.
    """
    
    # Add Gaussian uncertainty to chirp mass based on SNR
    Mc_center = Mc_z * np.exp(np.random.normal(0, sigma_mass / snr_obs, 1))
    Mc_obs = Mc_center * np.exp(np.random.normal(0, sigma_mass / snr_obs, n))

    # Sample mass ratio with skew-normal distribution
    q_center = truncated_skewnorm(alpha, sigma_q / snr_obs, q, 0, 1, size=1)
    q_obs = truncated_skewnorm(alpha, sigma_q / snr_obs, q_center, 0, 1, size=n)
    
    # Compute symmetric mass ratio
    symratio_obs = q_obs / (1 + q_obs)**2
    
    # Compute total mass and observed primary mass
    M = Mc_obs / symratio_obs**(3./5.)
    m1z_obs = 0.5 * M * (1 + np.sqrt(1 - 4 * symratio_obs))
    
    # Convert to source-frame primary mass
    m1s = m1z_obs / (1 + z)

    # Ensure the sampled masses fall within the expected mass range
    while True:
        indices = np.where((m1s > universe.m_max) + (m1s < universe.m_min))[0]  # Find out-of-bounds samples
        n_out = indices.size
        
        if n_out == 0:  # If all samples are valid, exit loop
            break
        # Resample out-of-bounds chirp masses
        _Mc_obs = Mc_center * np.exp(np.random.normal(0, sigma_mass / snr_obs, n_out))
        Mc_obs[indices] = _Mc_obs
        
        # Resample out-of-bounds mass ratios
        _q_obs = truncated_skewnorm(alpha, sigma_q / snr_obs, q_center, 0, 1, size=n_out)
        q_obs[indices] = _q_obs
        
        # Recompute symmetric mass ratio and primary mass
        _symratio_obs = _q_obs / (1 + _q_obs)**2
        M = _Mc_obs / _symratio_obs**(3./5.)
        _m1z_obs = 0.5 * M * (1 + np.sqrt(1 - 4 * _symratio_obs))
        
        # Update out-of-bounds values
        m1z_obs[indices] = _m1z_obs
        m1s[indices] = _m1z_obs / (1 + z)

    return m1z_obs, q_obs, Mc_obs, symratio_obs



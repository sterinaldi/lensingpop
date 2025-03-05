import numpy as np
from figaro.cosmology import CosmologicalParameters   
from numba import jit  

# Parameters for redshift distribution
alpha_z = 2.7  # Controls the shape of the redshift distribution
beta_z = 2.9  # Secondary shape parameter
z_p = 1.9  # Pivot point for redshift
z_min = 0.02  # Minimum redshift value
z_max = 1.3  # Maximum redshift value

# Parameters for the mass distribution (Power-law model)
m_min = 5  # Minimum mass
m_max = 100  # Maximum mass
alpha = -1.8  # Power-law index for the mass distribution
mu_0 = 55  # Mean of the Gaussian component in the mass distribution
sigma = 4 # Sigma of the Gaussian component in the mass distribution
dmudz = 20  # Rate of change of mean with redshift
w_0 = 0.15 # Relative weight for combining mass distribution components

# Parameters for the magnification distribution
gamma = -3  # Power-law index for magnification
mag_min = 4 # Minimum primary magnification value
mag_max = 30  # Maximum primary magnification value
mag2_min = -30  # Minimum secondary magnification value
mag2_max = -1  # Maximum secondary magnification value
rel_sd = 0.15  # Relative standard deviation for secondary magnification

# Cosmological parameters (Hubble constant, matter density, dark energy density)
h = 0.676
om = 0.315
ol = 0.685

# Initialize cosmology and calculate the maximum comoving volume
omega = CosmologicalParameters(h, om, ol, -1, 0, 0)  # w_0 = -1, w_a = 0, curvature = 0
vol_max = omega.ComovingVolume(np.array([z_max]))

# Distribution functions with JIT optimization for speed
@jit(nopython=False)
def PL_distribution(m, z):
    """Combined power-law and Gaussian mass distribution."""
    return (1 - weight(z)) * (m ** alpha * (1 + alpha) / (m_max ** (1 + alpha) - m_min ** (1 + alpha))) + weight(z) * norm(m, z)

@jit(nopython=False)
def mu(z):
    """Calculates the mean of the Gaussian component as a function of redshift."""
    return mu_0 + dmudz * (z / z_max)

@jit(nopython=False)
def norm(m, z):
    """Normalized Gaussian distribution for mass."""
    return np.exp(-(m - mu(z)) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)

@jit(nopython=False)
def redshift_distribution(z):
    """Distribution for redshift."""
    return (1 + (1 + z_p) ** (-alpha_z - beta_z)) * (1 + z) ** alpha_z / (1 + ((1 + z) / (1 + z_p)) ** (alpha_z + beta_z))

@jit(nopython=False)
def weight(z):
    """Weight function for mixing the power-law and Gaussian components."""
    return w_0 * (1 + (z / z_max))

@jit(nopython=False)
def mass_distribution(m, z):
    """Distribution function for mass at a given redshift."""
    return (1 - weight(z)) * (m ** alpha * (1 + alpha) / (m_max ** (1 + alpha) - m_min ** (1 + alpha))) + weight(z) * norm(m, z)

def magnification_distribution(mag):
    """Power-law distribution for magnification."""
    return (mag ** gamma * (1 + gamma) / (mag_max ** (1 + gamma) - mag_min ** (1 + gamma)))

def magnification2_distribution(mag2, mag1):
    """Gaussian distribution for a secondary magnification given a primary magnification. (Using SIS lensing model)"""
    mag2_sis = 2-mag1
    sigma_mag2 = rel_sd * np.abs(mag2_sis)
    return np.exp(-(mag2 - mag2_sis) ** 2 / (2 * sigma_mag2 ** 2)) / (np.sqrt(2 * np.pi) * sigma_mag2)

import numpy as np
import matplotlib.pyplot as plt
from corner import corner

from figaro.utils import rejection_sampler
from figaro.cosmology import CosmologicalParameters

from tqdm import tqdm
from numba import jit

# Parameters

# Redshift distribution
alpha_z = 2.7
beta_z  = 2.9
z_p     = 1.9
z_min = 0.02 #0.2
z_max = 1.3
# Mass distribution
# PowerLaw
m_min = 5 # default = 1
m_max = 100
alpha = -2.0   #default -1.35
#alpha = -1.35
mu_0  = 45  #default = 35
sigma = 5 #2.0
dmudz = 20

# Relative weight
w_0 = 0.12 # default = 0.3
# Magnification distribution
gamma   = -3
mag_min = 1
mag_max = 100
rel_sd = 0.15 
# Cosmology

h = 0.674
om = 0.315
ol = 0.685

omega = CosmologicalParameters(h, om, ol, -1, 0)
vol_max = omega.ComovingVolume(np.array([z_max]))

@jit
def PL_distribution(m, z):
    return (1-weight(z))*(m**alpha * (1+alpha)/(m_max**(1+alpha) - m_min**(1+alpha))) + weight(z)*norm(m, z)

@jit
def mu(z):
    return mu_0 + dmudz*(z/z_max)
@jit
def norm(m, z):
    return np.exp(-(m-mu(z))**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
@jit
def redshift_distribution(z):
    return (1 + (1+z_p)**(-alpha_z-beta_z)) * (1+z)**alpha_z / (1+((1+z)/(1+z_p))**(alpha_z+beta_z))

def lensed_redshift_distribution(z):
    return (omega.ComovingVolume(z)/ vol_max ) * redshift_distribution(z)
@jit
def weight(z):
    return w_0*(1+(z/z_max))
@jit
def mass_distribution(m, z):
    return (1-weight(z))*(m**alpha * (1+alpha)/(m_max**(1+alpha) - m_min**(1+alpha))) + weight(z)*norm(m, z)
@jit
def magnification_distribution(mag):
    return (mag**gamma * (1+gamma)/(mag_max**(1+gamma) - mag_min**(1+gamma)))
@jit
def magnification2_distribution(mag2,mag1):
    sigma_mag2 = rel_sd * mag1
    return np.exp(-(mag2-mag1)**2/(2*sigma_mag2**2))/(np.sqrt(2*np.pi)*sigma_mag2)

if __name__ == '__main__':

    n_draws = 10000
    z = rejection_sampler(n_draws, redshift_distribution, [0,z_max])
    mag = rejection_sampler(n_draws, magnification_distribution, [mag_min, mag_max])
    m1 = []
    m2 = []
    for zi in tqdm(z, desc = 'm'):
        masses = rejection_sampler(2, lambda m: mass_distribution(m, zi), [m_min, m_max])
        m1.append(np.max(masses))
        m2.append(np.min(masses))

    samples = np.array([m1, m2, z, mag]).T

    c = corner(samples, labels = ['m1', 'm2', 'z', 'mu'])
    plt.savefig('m1m2z.pdf')

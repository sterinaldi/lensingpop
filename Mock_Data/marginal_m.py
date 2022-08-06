import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from simulated_universe import *
from numba import jit

m_pts = 1000
z_pts = 1000

m  = np.linspace(m_min, m_max, m_pts)
z  = np.linspace(0, z_max, z_pts)
dz = z[1]-z[0]

# True distribution
f_m = []

for mi in tqdm(m, desc = 'True'):
    f_m.append(np.sum(mass_distribution(mi,z)*redshift_distribution(z)*(1+z)*dz))

np.savetxt('true_mass_dist.txt', np.array([m, f_m]).T, header = 'm p')

plt.plot(m, f_m, lw = 0.7, label = '$True\ distribution$')

# No evolution
@jit
def mass_distribution_noevol(m, z):
    return (1-weight_noevol(z))*(m**alpha * (1+alpha)/(m_max**(1+alpha) - m_min**(1+alpha))) + weight_noevol(z)*norm_noevol(m, z)
@jit
def mu_noevol(z):
    return mu_0
@jit
def norm_noevol(m, z):
    return np.exp(-(m-mu_noevol(z))**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
@jit
def weight_noevol(z):
    return w_0
    
f_m = []

for mi in tqdm(m, desc = 'No evolution'):
    f_m.append(np.sum(mass_distribution_noevol(mi,z)*redshift_distribution(z)*(1+z)*dz))

np.savetxt('mass_dist_noevol.txt', np.array([m, f_m]).T, header = 'm p')

plt.plot(m, f_m, lw = 0.7, label = '$No\ evolution$')

# PL
@jit
def mass_distribution_PL(m, z):
    return (m**alpha * (1+alpha)/(m_max**(1+alpha) - m_min**(1+alpha)))
f_m = []

for mi in tqdm(m, desc = 'PowerLaw'):
    f_m.append(np.sum(mass_distribution_PL(mi,z)*redshift_distribution(z)*(1+z)*dz))

np.savetxt('mass_dist_pl.txt', np.array([m, f_m]).T, header = 'm p')

plt.plot(m, f_m, lw = 0.7, label = '$PowerLaw$')


plt.xlabel('$M\ [M_\\odot]$')
plt.ylabel('$p(M)$')
plt.legend(loc = 0, frameon = False)
plt.savefig('mass_dist.pdf', bbox_inches = 'tight')
        

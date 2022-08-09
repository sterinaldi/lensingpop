import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from simulated_universe import *
from numba import jit

catfile = 'PowerlawplusPeakplusDelta100000Samples.npz' #'Catalog_Unlensed_1087.npz'

mass_samples = np.load(catfile)['m1']# np.concatenate((np.load(catfile)['m1'],np.load(catfile)['m2']))
redshift_samples = np.load(catfile)['redshift']# np.concatenate((np.load(catfile)['redshift'],np.load(catfile)['redshift']))
plt.hist(mass_samples*(1+redshift_samples), bins = int(np.sqrt(len(mass_samples))), histtype = 'step', density = True)

m_pts = 1000
z_pts = 1000

m  = np.linspace(m_min, m_max*(1+z_max), m_pts)
m_sf = np.linspace(m_min, m_max, m_pts)
z  = np.linspace(0, z_max, z_pts)
dz = z[1]-z[0]
dm = m[1]-m[0]

# True distribution
f_m = []

def m_z_dist(m, z):
    marg_m2 = np.array([np.sum(mass_distribution(m_sf[m_sf < mi/(1+zi)], zi))*dm for zi in z])
    f = mass_distribution(mi/(1+z),z)*redshift_distribution(z)*dz*marg_m2/(1+z)
    f[mi/(1+z) < m_min] = 0
    f[mi/(1+z) > m_max] = 0
    return f

for mi in tqdm(m, desc = 'True'):
    f_m.append(np.sum(m_z_dist(mi, z)))
f_m = np.array(f_m)/(np.sum(f_m)*dm)

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

def m_z_dist_noevol(m, z):
    marg_m2 = np.array([np.sum(mass_distribution_noevol(m_sf[m_sf < mi/(1+zi)], zi))*dm for zi in z])
    f = mass_distribution_noevol(mi/(1+z),z)*redshift_distribution(z)*dz*marg_m2/(1+z)
    f[mi/(1+z) < m_min] = 0
    f[mi/(1+z) > m_max] = 0
    return f

for mi in tqdm(m, desc = 'No evolution'):
    f_m.append(np.sum(m_z_dist_noevol(mi,z)))
f_m = np.array(f_m)/(np.sum(f_m)*dm)

np.savetxt('mass_dist_noevol.txt', np.array([m, f_m]).T, header = 'm p')

plt.plot(m, f_m, lw = 0.7, label = '$No\ evolution$')

# PL
@jit
def mass_distribution_PL(m, z):
    return (m**alpha * (1+alpha)/(m_max**(1+alpha) - m_min**(1+alpha)))
f_m = []

def m_z_dist_PL(m, z):
    marg_m2 = np.array([np.sum(mass_distribution_PL(m_sf[m_sf < mi/(1+zi)], zi))*dm for zi in z])
    f = mass_distribution_PL(mi/(1+z),z)*redshift_distribution(z)*dz*marg_m2/(1+z)
    f[mi/(1+z) < m_min] = 0
    f[mi/(1+z) > m_max] = 0
    return f

for mi in tqdm(m, desc = 'PowerLaw'):
    f_m.append(np.sum(m_z_dist_PL(mi,z)))
f_m = np.array(f_m)/(np.sum(f_m)*dm)

np.savetxt('mass_dist_pl.txt', np.array([m, f_m]).T, header = 'm p')

plt.plot(m, f_m, lw = 0.7, label = '$PowerLaw$')


plt.xlabel('$M\ [M_\\odot]$')
plt.ylabel('$p(M)$')
plt.legend(loc = 0, frameon = False)
plt.savefig('mass_dist.pdf', bbox_inches = 'tight')
        

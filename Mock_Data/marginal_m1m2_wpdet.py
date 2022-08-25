import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from simulated_universe import *
from numba import jit
from scipy.interpolate import RegularGridInterpolator
import dill 

catfile = 'Catalog_100000Samples_afterSelection.npz' 

mass_samples = np.load(catfile)['m1']# np.concatenate((np.load(catfile)['m1'],np.load(catfile)['m2']))
redshift_samples = np.load(catfile)['redshift']# np.concatenate((np.load(catfile)['redshift'],np.load(catfile)['redshift']))
plt.hist(mass_samples*(1+redshift_samples), bins = int(np.sqrt(len(mass_samples))), histtype = 'step', density = True)



with open('./gwdet_default_interpolator.pkl', 'rb') as f:
    pdet = dill.load(f)

m_pts = 400
z_pts = 400
z_max = 2.3
m  = np.linspace(m_min, m_max*(1+z_max), m_pts)
z  = np.linspace(0, z_max, z_pts)
dz = z[1]-z[0]
dm = m[1]-m[0]

m1z_grid, m2z_grid = np.meshgrid(m,m,indexing='ij')
f_m = []

def pdet_m1zm2z(m1z,m2z,z):
    p = np.array([pdet(np.array([m1z/(1+zi),m2z/(1+zi),zi]).T)*mass_distribution(m1z/(1+zi),zi)*mass_distribution(m2z/(1+zi),zi)*redshift_distribution(zi)*dz/(1+zi)**2 for zi in z])
    m1 = m1z/(1+z)
    m2 = m2z/(1+z)
    p[m1 < m_min] = 0
    p[m1 > m_max] = 0
    p[m2 < m_min] = 0
    p[m2 > m_max] = 0   
    return np.sum(p)

f_m = []
for m1zi,m2zi in tqdm(zip(m1z_grid.flatten(), m2z_grid.flatten() ), total = len(m1z_grid.flatten()), desc = 'Real\ distribution'):
    if m1zi>=m2zi:
        f_m.append(pdet_m1zm2z(m1zi,m2zi,z))
    else:
        f_m.append(0)

f_m = np.reshape(f_m, (m_pts,m_pts))
interp = RegularGridInterpolator((m,m), f_m, bounds_error=False, fill_value=0)

with open('./real_dist_wpdet.pkl', 'wb') as file:
    dill.dump(interp, file)

f_m = np.sum(f_m, axis=1)*dm

norm = np.sum(f_m)*dm
plt.plot(m, f_m/norm, lw = 0.7, label = '$Real\ distribution\ x \ p_{\mathrm{det}}$')


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
    
def m_z_dist_noevol(m1z, m2z, z):
    p = np.array([pdet(np.array([m1z/(1+zi),m2z/(1+zi),zi]).T)*mass_distribution_noevol(m1z/(1+zi),zi)*mass_distribution_noevol(m2z/(1+zi),zi)*redshift_distribution(zi)*dz/(1+zi)**2 for zi in z])
    m1 = m1z/(1+z)
    m2 = m2z/(1+z)
    p[m1 < m_min] = 0
    p[m1 > m_max] = 0
    p[m2 < m_min] = 0
    p[m2 > m_max] = 0
    return np.sum(p)

f_m = []
for m1zi,m2zi in tqdm(zip(m1z_grid.flatten(), m2z_grid.flatten() ), total = len(m1z_grid.flatten()), desc = 'No evolution '):
    if m1zi>=m2zi:
        f_m.append(m_z_dist_noevol(m1zi,m2zi,z))
    else:
        f_m.append(0)

f_m = np.reshape(f_m, (m_pts,m_pts))
interp = RegularGridInterpolator((m,m), f_m, bounds_error=False, fill_value=0)

with open('./noevol_wpdet.pkl', 'wb') as file:
    dill.dump(interp, file)

f_m = np.sum(f_m, axis=1)*dm
norm = np.sum(f_m)*dm
plt.plot(m, f_m/norm, lw = 0.7, label = '$No\ evolution\ x \ p_{\mathrm{det}}$')

# PL
@jit
def mass_distribution_PL(m, z):
    return (m**alpha * (1+alpha)/(m_max**(1+alpha) - m_min**(1+alpha)))
f_m = []

def m_z_dist_PL(m1z, m2z, z):
    p = np.array([pdet(np.array([m1z/(1+zi),m2z/(1+zi),zi]).T)*mass_distribution_PL(m1z/(1+zi),zi)*mass_distribution_PL(m2z/(1+zi),zi)*redshift_distribution(zi)*dz/(1+zi)**2 for zi in z])
    m1 = m1z/(1+z)
    m2 = m2z/(1+z)
    p[m1 < m_min] = 0
    p[m1 > m_max] = 0
    p[m2 < m_min] = 0
    p[m2 > m_max] = 0
    return np.sum(p)

f_m = []
for m1zi,m2zi in tqdm(zip(m1z_grid.flatten(), m2z_grid.flatten() ), total = len(m1z_grid.flatten()), desc = 'Power law'):
    if m1zi>=m2zi:
        f_m.append(m_z_dist_noevol(m1zi,m2zi,z))
    else:
        f_m.append(0)

f_m = np.reshape(f_m, (m_pts,m_pts))
interp = RegularGridInterpolator((m,m), f_m, bounds_error=False, fill_value=0)

with open('./PL_wpdet.pkl', 'wb') as file:
    dill.dump(interp, file)

f_m = np.sum(f_m, axis=1)*dm
norm = np.sum(f_m)*dm
plt.plot(m, f_m/norm, lw = 0.7, label = '$Powerlaw\ x \ p_{\mathrm{det}}$')
plt.xlim(0,250)
plt.title('After selection')
plt.xlabel('$m_1^z\ [M_\\odot]$')
plt.ylabel('$p(m_1^z)$')
plt.legend(loc = 0, frameon = False)
plt.savefig('dist_wpdet.pdf', bbox_inches = 'tight')
        

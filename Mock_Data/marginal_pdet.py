import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from simulated_universe import *
from numba import jit
from scipy.interpolate import RegularGridInterpolator
import dill 

with open('./gwdet_default_interpolator.pkl', 'rb') as f:
    pdet = dill.load(f)

m_pts = 500
z_pts = 500
z_max = 2.3
m  = np.linspace(m_min, m_max*(1+z_max), m_pts)
z  = np.linspace(0, z_max, z_pts)
dz = z[1]-z[0]
dm = m[1]-m[0]

m1z_grid, m2z_grid = np.meshgrid(m,m,indexing='ij')
f_m = []
def pdet_m1zm2z(m1z,m2z,z):
    
    p = np.array([pdet(np.array([m1z/(1+zi),m2z/(1+zi),zi]).T)*dz/(1+zi)**2 for zi in z])
    m1 = m1z/(1+z)
    m2 = m2z/(1+z)
    p[m1 < m_min] = 0
    p[m1 > m_max] = 0
    p[m2 < m_min] = 0
    p[m2 > m_max] = 0
    p[m1 < m1] = 0
    return np.sum(p)

f_m = []
for m1zi,m2zi in tqdm(zip(m1z_grid.flatten(), m2z_grid.flatten() ), total = len(m1z_grid.flatten()), desc = 'pdet'):
    if m1zi>=m2zi:
        f_m.append(pdet_m1zm2z(m1zi,m2zi,z))
    else:
        f_m.append(0)

f_m = np.reshape(f_m, (m_pts,m_pts))
interp = RegularGridInterpolator((m,m), f_m, bounds_error=False, fill_value=0)

with open('./pdet_m1zm2z.pkl', 'wb') as file:
    dill.dump(interp, file)

f_m = np.sum(f_m, axis=1)*dm

norm = np.sum(f_m)*dm
plt.plot(m, f_m/norm, lw = 0.7, label = '$Real\ distribution$')


plt.plot(m, f_m, lw = 0.7, label = '$PowerLaw$')


plt.xlabel('$m_1^z\ [M_\\odot]$')
plt.ylabel('$p_{\mathrm{pdet}}(m_1^z)$')
plt.legend(loc = 0, frameon = False)
plt.savefig('pdet.pdf', bbox_inches = 'tight')
        

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from simulated_universe import *
from numba import jit
from scipy.interpolate import RegularGridInterpolator
import dill 
from matplotlib.ticker import MaxNLocator
plt.style.use('./plotrc.mplstyle')
import matplotlib as mpl

mpl.rcParams['lines.linewidth'] = 2
#mpl.rcParams['lines.linestyle'] = '--'

mpl.rcParams['xtick.major.width'] = 0.6  # default 0.8
mpl.rcParams['ytick.major.width'] = 0.6  # default 0.8
mpl.rcParams['axes.linewidth'] = 0.6  # default 0.8 
mpl.rcParams['lines.linewidth'] = 0.6  # default 1.5 
mpl.rcParams['lines.markeredgewidth'] = 0.6  # default 1
# The magic sauce
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = ''.join([r'\usepackage[T1]{fontenc}'
                                           r'\usepackage{cmbright}'])
catfile = './catalog/PowerlawplusPeakplusDelta30000Samples.npz' #'Catalog_Unlensed_1087.npz'
mass_samples = np.load(catfile)['m1']# np.concatenate((np.load(catfile)['m1'],np.load(catfile)['m2']))
redshift_samples = np.load(catfile)['redshift']# np.concatenate((np.load(catfile)['redshift'],np.load(catfile)['redshift']))



fig, ax = plt.subplots(figsize=(8,6))
ax.hist(mass_samples*(1+redshift_samples), bins = int(np.sqrt(len(mass_samples))), histtype = 'step', density = True, label='Astri dist')

m_pts = 200
z_pts = 200
dim=2
dims = list(np.arange(dim, dtype = int))
dims.remove(0)
n_pts  = np.array([m_pts,z_pts])
#q_bds  = [0.2, 1.]
#q  = np.linspace(q_bds[0], q_bds[1], n_pts[1]+2)[1:-1]
#m = np.linspace(lim[0], lim[1], n_pts[0]+2)[1:-1]
#dm = m[1]-m[0]
#chieff = np.linspace(-1.,1., n_pts[2]+2)[1:-1]
#dgrid = [m[1]-m[0], m[1]-m[0], chieff[1]-chieff[0]]
"""
m  = np.linspace(m_min, m_max*(1+z_max), m_pts)
q  = np.linspace(0, 1, m_pts)
z  = np.linspace(0, z_max, z_pts)
grid  = np.zeros(shape = (np.prod(n_pts), 2))
dgrid = [m[1]-m[0], q[1]-q[0]]
dz = z[1]-z[0]
for i, m1i in enumerate(m):
    for j, qi in enumerate(q):
            grid[i*n_pts[1] + j] = [m1i, m1i*qi]
"""

m  = np.linspace(m_min, m_max*(1+z_max), m_pts)
q  = np.linspace(0, 1, m_pts)
z  = np.linspace(0, z_max, z_pts)
grid  = np.zeros(shape = (np.prod(n_pts), 2))
dgrid = [m[1]-m[0], m[1]-m[0]]
dz = z[1]-z[0]
for i, m1i in enumerate(m):
    for j, m2i in enumerate(m):
            grid[i*n_pts[1] + j] = [m1i, m2i]

# True distribution

def m_z_dist(m1z,m2z, z):
    p = np.array([mass_distribution(m1z/(1+zi),zi)*mass_distribution(m2z/(1+zi),zi)*redshift_distribution(zi)*dz/((1+zi)**2) for zi in z])
    m1 = m1z/(1+z)
    m2 = m2z/(1+z)
    p[m1 < m_min] = 0
    p[m1 > m_max] = 0
    p[m2 < m_min] = 0
    p[m2 > m_max] = 0
    
    return np.sum(p)

f_m = []
for m1zi,m2zi in tqdm(grid, total = len(grid), desc = 'Real distribution'):
    if m1zi>=m2zi:
        f_m.append(m_z_dist(m1zi,m2zi,z))
    else:
        f_m.append(0)

norm = np.sum(f_m)*(m[1]-m[0])**2 /2
f_m = np.reshape(f_m, n_pts)

print(norm)
interp = RegularGridInterpolator((m,m), f_m/norm, bounds_error=False, fill_value=0)

with open('./real_distv3.pkl', 'wb') as file:
    dill.dump(interp, file)

probs = np.array([f_m.sum(axis = tuple(dims))*np.prod([dgrid[k] for k in dims])]).reshape(m.shape)
norm = probs.sum()*dgrid[0] 

ax.plot(m, probs/norm, lw = 0.7, label = 'Benchmark',color='black')
"""
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
    p=np.array([mass_distribution_noevol(m1z/(1+zi),zi)*mass_distribution_noevol(m2z/(1+zi),zi)*redshift_distribution(zi)*dz/(1+zi)**2 for zi in z])
    m1 = m1z/(1+z)
    m2 = m2z/(1+z)
    p[m1 < m_min] = 0
    p[m1 > m_max] = 0
    p[m2 < m_min] = 0
    p[m2 > m_max] = 0
    return np.sum(p)

f_m = []
for m1zi,m2zi in tqdm(zip(m1z_grid.flatten(), m2z_grid.flatten() ), total = len(m1z_grid.flatten()), desc = 'No evolution'):
    if m1zi>=m2zi:
        f_m.append(m_z_dist_noevol(m1zi,m2zi,z))
    else:
        f_m.append(0)

f_m = np.reshape(f_m, (m_pts,m_pts))
interp = RegularGridInterpolator((m,m), f_m, bounds_error=False, fill_value=0)

with open('./noevol_pdf.pkl', 'wb') as file:
    dill.dump(interp, file)

f_m = np.sum(f_m, axis=1)*dm
norm = np.sum(f_m)*dm
plt.plot(m, f_m/norm, lw = 0.7, linestyle='dashed', label = '$No\ evolution$')
"""
# PL
#@jit
#def mass_distribution_PL(m,z):
#    return (m**alpha * (1+alpha)/(m_max**(1+alpha) - m_min**(1+alpha))) 
@jit
def mass_distribution_PL(mm1,mm2, z):
    beta = 0.2
    return (mm1**alpha * (1+alpha)/(m_max**(1+alpha) - m_min**(1+alpha))) *  ((mm2/mm1)**beta * (1+beta)/(1**(1+beta) - 0.2**(1+beta)))  
    #if mm1 <= m_min: return 0
    #return (mm1**alpha * (1+alpha)/(m_max**(1+alpha) - m_min**(1+alpha))) /( mm1-m_min)
f_m = []

def m_z_dist_PL(m1z, m2z, z):
    #p = np.array([mass_distribution_PL(m1z/(1+zi),zi)*mass_distribution_PL(m2z/(1+zi),zi)*redshift_distribution(zi)*dz/(1+zi)**2 for zi in z])
    p = np.array([mass_distribution_PL(m1z/(1+zi),m2z/(1+zi),zi)*redshift_distribution(zi)*dz/((1+zi)**2)/(m1z/(1+zi)) for zi in z])
    #p = np.array([mass_distribution_PL(m1z/(1+zi),zi)/(m_max-m_min)*redshift_distribution(zi)*dz/((1+zi)**2) for zi in z])
    m1 = m1z/(1+z)
    m2 = m2z/(1+z)
    p[m1 < m_min] = 0
    p[m1 > m_max] = 0
    p[m2 < m_min] = 0
    p[m2 > m_max] = 0
    return np.sum(p)

f_m = []
for m1zi,m2zi in tqdm(grid, total = len(grid), desc = 'Power law'):
    if m1zi>=m2zi:
        f_m.append(m_z_dist_PL(m1zi,m2zi,z))
    else:
        f_m.append(0)

norm = np.sum(f_m)*(m[1]-m[0])**2 /2
f_m = np.reshape(f_m, n_pts)

interp = RegularGridInterpolator((m,m), f_m/norm, bounds_error=False, fill_value=0)
print(norm)
with open('../result/pop_prior/PL_pdfv3.pkl', 'wb') as file:
    dill.dump(interp, file)

probs = np.array([f_m.sum(axis = tuple(dims))*np.prod([dgrid[k] for k in dims])]).reshape(m.shape)
norm = probs.sum()*dgrid[0]
ax.plot(m, probs/norm, lw = 0.7, label = 'PL',color='orange')
        
    
    
ax.set_ylim(0,0.04)
ax.set_ylabel(r'$p(m_1^z)$',fontsize=18)
ax.set_xlabel('$m_1^z\ [M_\\odot]$')
[l.set_rotation(45) for l in ax.get_xticklabels()]
ticks = [20,50,80,110,140,170,200,230,260]
ax.set_xticks(ticks)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.legend(loc = 0, frameon = False)
fig.savefig('./mass_distv2.pdf', bbox_inches = 'tight')
ax.set_xlim(15.15,225)
fig.savefig('./mass_distv2_xlim.pdf', bbox_inches = 'tight')

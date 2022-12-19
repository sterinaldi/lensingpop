import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from simulated_universe import *
from numba import jit
from scipy.interpolate import RegularGridInterpolator
import dill 
from xeff_pop import *
plt.style.use('./plotrc.mplstyle')
import matplotlib as mpl

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 0.6  # default 0.8
mpl.rcParams['ytick.major.width'] = 0.6  # default 0.8
mpl.rcParams['axes.linewidth'] = 0.6  # default 0.8 
mpl.rcParams['lines.linewidth'] = 0.6  # default 1.5 
mpl.rcParams['lines.markeredgewidth'] = 0.6  # default 1
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = ''.join([r'\usepackage[T1]{fontenc}'
                                           r'\usepackage{cmbright}'])

class AstroDist:
    def __init__(self, points, f, wide=False):
        self.I = RegularGridInterpolator(points, f, fill_value = 0., bounds_error = False) # FIXME: sistemare
        self.spin = Gaussian_spin_distribution(wide=wide, **spin_pars)
    def __call__(self, x):
        return self.pdf(x)
    def mq_pdf(self, x):
        m1  = x[:,0]
        q  = x[:,1]
        return self.I((m1, q)) # Jacobian
    
    def pdf(self, x):
        m1  = x[:,0]
        m2  = x[:,1]
        chi = x[:,2]
        return self.spin.prob(chi)*self.I((m1, m2/m1))/m1 # Jacobian

    
z_bds  = [0.01,1.3]
m1_bds = [15*(1+z_bds[0]), 98*(1+z_bds[1])]
# Plot Astrophysical distribution of the catalog
catfile = './catalog/PowerlawplusPeakplusDelta30000Samples.npz'
mass_samples = np.load(catfile)['m1']
redshift_samples = np.load(catfile)['redshift']

fig, ax = plt.subplots(figsize=(8,6))
samples = mass_samples*(1+redshift_samples)
samples = samples[samples>m1_bds[0]]
ax.hist(samples, color='blue',
        bins = int(np.sqrt(len(mass_samples))), histtype = 'step', density = True, label='Astri dist')


# Initialize the (m1, q, chi) grid
z_pts = 200
n_pts = 200
n_pts  = np.array([n_pts, n_pts])


q_bds  = [0.2, 1.]#[5.,240.]
bounds    = np.array([m1_bds, q_bds])
m = np.linspace(m1_bds[0], m1_bds[1], n_pts[0]+2)[1:-1]
q  = np.linspace(q_bds[0], q_bds[1], n_pts[1]+2)[1:-1]
z  = np.linspace(z_bds[0], z_bds[1], z_pts+2)[1:-1]

dz = z[1]-z[0]
dgrid = [m[1]-m[0], q[1]-q[0]]
grid  = np.zeros(shape = (np.prod(n_pts), 2))

# Grid (m1, q, chi)
for i, m1i in tqdm(enumerate(m), desc = 'Grid', total = n_pts[0]):
    for j, qi in enumerate(q):
        grid[i*n_pts[1] + j] = [m1i, qi*m1i]
            
        
            
            
# Benchmark (m1 m2 z) distribution
def m_z_dist(m1z,m2z, z):
    p = np.array([mass_distribution(m1z/(1+zi),zi)*mass_distribution(m2z/(1+zi),zi)*redshift_distribution(zi)*dz*(m1z)/((1+zi)**3) for zi in z])
    m1 = m1z/(1+z)
    m2 = m2z/(1+z)
    p[m1 < m_min] = 0
    p[m1 > m_max] = 0
    p[m2 < m_min] = 0
    p[m2 > m_max] = 0
    
    return np.sum(p)

f_m = []
for m1zi,m2zi in tqdm(grid[:,:2], total = len(grid), desc = 'Real distribution'):
    if m1zi>=m2zi:
        f_m.append(m_z_dist(m1zi,m2zi,z))
    else:
        f_m.append(0)

#det_jacobian = (1/grid[:,0])
#f_m = f_m / det_jacobian
        
f_m = np.reshape(f_m, n_pts)

# Normalization of p(m1z, m2z, chi)
# samples from (m1z,q) uniform distribution
mc_samples = np.random.uniform(low=bounds[:,0], high=bounds[:,1], size=(int(1e6),2))
interpolator = AstroDist((m, q), f_m)
norm = np.sum(interpolator.mq_pdf(mc_samples))*np.prod(dgrid)
f_m = f_m / norm
print(norm)


# Saving model
print('Saving benchmark interpolators...')
benchmark_dist = AstroDist((m, q), f_m)
with open('../result/pop_prior/benchmark_dist.pkl', 'wb') as f:
    dill.dump(benchmark_dist, f)
    


# PL
def mass_distribution_PL(mm1,mm2, z):
    beta = 0.2
    return (mm1**alpha * (1+alpha)/(m_max**(1+alpha) - m_min**(1+alpha))) *  ((mm2/mm1)**beta / (1+beta)/(1**(1+beta) - q_bds[0]**(1+beta)))  
    #if mm1 <= m_min: return 0
    #return (mm1**alpha * (1+alpha)/(m_max**(1+alpha) - m_min**(1+alpha))) /( mm1-m_min)
f_m = []

def m_z_dist_PL(m1z, m2z, z):
    #p = np.array([mass_distribution_PL(m1z/(1+zi),zi)*mass_distribution_PL(m2z/(1+zi),zi)*redshift_distribution(zi)*dz/(1+zi)**2 for zi in z])
    p = np.array([mass_distribution_PL(m1z/(1+zi),m2z/(1+zi),zi)*redshift_distribution(zi)*dz/((1+zi)) for zi in z])
    #p = np.array([mass_distribution_PL(m1z/(1+zi),zi)/(m_max-m_min)*redshift_distribution(zi)*dz/((1+zi)**2) for zi in z])
    m1 = m1z/(1+z)
    m2 = m2z/(1+z)
    p[m1 < m_min] = 0
    p[m1 > m_max] = 0
    p[m2 < m_min] = 0
    p[m2 > m_max] = 0
    return np.sum(p)

f_m = []
for m1zi,m2zi in tqdm(grid[:,:2], total = len(grid), desc = 'Power law'):
    if m1zi>=m2zi:
        f_m.append(m_z_dist_PL(m1zi,m2zi,z))
    else:
        f_m.append(0)
        
#det_jacobian = (1/grid[:,0])
#f_m = f_m / det_jacobian
f_m = np.reshape(f_m, n_pts)

        
# Normalization of p(m1z, m2z, chi)
# samples from (m1z,q) uniform distribution
mc_samples = np.random.uniform(low=bounds[:,0], high=bounds[:,1], size=(int(1e6),2))

interpolator = AstroDist((m, q), f_m, wide=True)
norm = np.sum(interpolator.mq_pdf(mc_samples))*np.prod(dgrid)
f_m = f_m / norm
print(norm)
    
# Saving model
print('Saving interpolators...')
PL_dist = AstroDist((m, q), f_m, wide=True)
with open('../result/pop_prior/PL_dist.pkl', 'wb') as f:
    dill.dump(PL_dist, f)
    
      
    
# Plot marginalize m1z plot with (m1, m2, chi) grid
dim=3
dims = list(np.arange(dim, dtype = int))
# marginalize to axis = 0 for plotting result
dims.remove(0)
# Grid (m1z, m2z, chi)
m1_bds = [15*(1+z_bds[0]), 98*(1+z_bds[1])]
n_pts  = np.array([75,75,75])
m = np.linspace(m1_bds[0], m1_bds[1], n_pts[0]+2)[1:-1]
dm = m[1]-m[0]
chieff = np.linspace(-1.,1., n_pts[2]+2)[1:-1]
dgrid = [m[1]-m[0], m[1]-m[0], chieff[1]-chieff[0]]

dims = list(np.arange(dim, dtype = int))
dims.remove(0)
grid  = np.zeros(shape = (np.prod(n_pts), 3))

for i, m1i in enumerate(m):
    for j, m2i in enumerate(m):
            for l, xi in enumerate(chieff):
                grid[i*(n_pts[1]*n_pts[2]) + j*n_pts[2] + l] = [m1i, m2i, xi]

        
# plot benchmark m1z
probs = benchmark_dist(grid).reshape(n_pts)
probs[np.isnan(probs)] = 0
probs = np.array([probs.sum(axis = tuple(dims))*np.prod([dgrid[k] for k in dims])]).reshape(m.shape)
norm = probs.sum()*dgrid[0]
ax.plot(m, probs/norm, lw = 1.5, color='black', label='Benchmark dist')
print('benchmark norm',norm)

probs = PL_dist(grid).reshape(n_pts)
probs[np.isnan(probs)] = 0
probs = np.array([probs.sum(axis = tuple(dims))*np.prod([dgrid[k] for k in dims])]).reshape(m.shape)
norm = probs.sum()*dgrid[0]
ax.plot(m, probs/norm, lw = 1.5, color='orange', label='PL dist')
print('PL norm',norm)

    
ax.set_ylim(0,0.04)
ax.set_ylabel(r'$p(m_1^z)$',fontsize=18)
ax.set_xlabel('$m_1^z\ [M_\\odot]$')
[l.set_rotation(45) for l in ax.get_xticklabels()]
ticks = [20,50,80,110,140,170,200,230,260]
ax.set_xticks(ticks)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.legend(loc = 0, frameon = False)
fig.savefig('./mass_dist_m1qchi.pdf', bbox_inches = 'tight')
ax.set_xlim(15.15,225)
fig.savefig('./mass_dist_m1qchi_xlim.pdf', bbox_inches = 'tight')

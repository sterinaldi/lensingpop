import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from simulated_universe import *
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import dill 
plt.style.use('../result/plotrc.mplstyle')
import matplotlib as mpl

class AstroDist:
    def __init__(self, points, f, wide=False):
        self.I = RegularGridInterpolator(points, f, fill_value = 0., bounds_error = False) # FIXME: sistemare
       
    def __call__(self, x):
        return self.pdf(x)
    
    def pdf(self, x):
        m1  = x[:,0]
        q  = x[:,1]
        return self.I((m1, q)) 

    
z_bds  = [z_min,z_max]
#m1_bds = [m_min*(1+z_bds[0]), m_max*(1+z_bds[1])]
m1_bds = [5,200]
q_bds  = [0.1, 1.]#[5.,240.]
Vol = (m1_bds[1]-m1_bds[0])*0.9
##### Plot Astrophysical distribution of the catalog
catfile = './catalog/PowerlawplusPeakplusDelta24000Samples_unlensed.npz'
m1 = np.load(catfile)['m1']
m2 = np.load(catfile)['m2']
redshift_samples = np.load(catfile)['redshift']

fig, ax = plt.subplots(figsize=(8,6))
q = m2/m1
samples = m1*(1+redshift_samples)
ax.hist(samples, color='blue',
        bins = int(np.sqrt(len(samples))), histtype = 'step', density = True, label='Astro dist')


####################################

# Initialize the (m1, q, z, chi) grid
z_pts = 300
n_pts = 300
n_pts  = np.array([n_pts, n_pts])
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
        grid[i*n_pts[1] + j] = [m1i, qi]


####################################
# Benchmark distribution
def m_z_dist(m1z,m2z, z):
    p = np.array([mass_distribution(m1z/(1+zi),zi)*mass_distribution(m2z/(1+zi),zi)*redshift_distribution(zi)*dz*m1z/((1+zi)**2) for zi in z])
    m1 = m1z/(1+z)
    m2 = m2z/(1+z)
    p[(m1 < m_min)+(m1 > m_max)+(m2 > m1)+(m2 < m_min)] = 0
    return np.sum(p)

f_m = []
for m1zi,qi in tqdm(grid, total = len(grid), desc = 'Real distribution'):
    f_m.append(m_z_dist(m1zi,m1zi*qi,z))

f_m = np.reshape(f_m, n_pts)



#################################### Normalized to 2D (m1z, q)
# Normalization of p(m1z, m2z, chi)
# samples from (m1z,q) uniform distribution
bounds    = np.array([m1_bds, q_bds])
mc_samples = np.random.uniform(low=bounds[:,0], high=bounds[:,1], size=(int(2e6),2))

interpolator = AstroDist((m, q), f_m)
norm = np.mean(interpolator.pdf(mc_samples)) * Vol

f_m = f_m / norm
####################

# Saving model
print('Saving benchmark interpolators...')
benchmark_dist = AstroDist((m, q), f_m)
with open('../result/pop_prior/bench_m1q.pkl', 'wb') as f:
    dill.dump(benchmark_dist, f)
    
    
    
#################################### Normalized to 1D (m1z)


class AstroDist1d:
    def __init__(self, points, f, wide=False):
        self.I = RegularGridInterpolator(points, f, fill_value = 0., bounds_error = False) # FIXME: sistemare
       
    def __call__(self, x):
        return self.pdf(x)
    
    def pdf(self, x):
        return self.I((x)) 



f_m = np.sum(f_m, axis=1) * dgrid[1]

# Saving model
print('Saving benchmark interpolators...')
benchmark_dist = AstroDist1d((m,), f_m)
with open('../result/pop_prior/bench_m1.pkl', 'wb') as f:
    dill.dump(benchmark_dist, f)
    
    
################################# Plot p(m1z) ######################
print('check normalization=',np.sum(f_m*dgrid[0]))

ax.plot(m, f_m, label='Benchmark pdf', color='black')
ax.set_ylabel(r'$p(m_1^z)$',fontsize=18)
ax.set_xlabel('$m_1^z\ [M_\odot]$',fontsize=18)
[l.set_rotation(45) for l in ax.get_xticklabels()]
#ticks = [20,50,80,110,140,170,200,230,260]
#ax.set_xticks(ticks)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.legend(loc = 0, frameon = False,fontsize=18)
ax.set_xlim(m1_bds[0],m1_bds[1])
fig.savefig('./benchmark_mass_dist.pdf', bbox_inches = 'tight')

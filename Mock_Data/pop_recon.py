import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from simulated_universe import *
from numba import jit
from scipy.interpolate import RegularGridInterpolator
import dill 
from figaro.mixture import DPGMM

catfile = 'PowerlawplusPeakplusDelta100000Samples.npz' #'Catalog_Unlensed_1087.npz'

mass_samples = np.load(catfile)['m1']# np.concatenate((np.load(catfile)['m1'],np.load(catfile)['m2']))
redshift_samples = np.load(catfile)['redshift']# np.concatenate((np.load(catfile)['redshift'],np.load(catfile)['redshift'])
plt.hist(mass_samples*(1+redshift_samples), bins = int(np.sqrt(len(mass_samples))),label='before pdet', histtype = 'step', density = True)

catfile = 'Catalog_100000Samples_afterSelection.npz'

mass_samples = np.load(catfile)['m1']# np.concatenate((np.load(catfile)['m1'],np.load(catfile)['m2']))
redshift_samples = np.load(catfile)['redshift']# np.concatenate((np.load(catfile)['redshift'],np.load(catfile)['redshift']))
plt.hist(mass_samples*(1+redshift_samples), bins = int(np.sqrt(len(mass_samples))),label ='after pdet', histtype = 'step', density = True)



m_pts = 200
z_pts = 200
z_max = 2.1

m  = np.linspace(m_min, m_max*(1+z_max), m_pts)
m_src = np.linspace(m_min, m_max, m_pts)
z  = np.linspace(0, z_max, z_pts)
dz = z[1]-z[0]
dm = m[1]-m[0]
dm_src = m_src[1]-m_src[0]
m1_grid, m2_grid, z_grid = np.meshgrid(m_src,m_src,z,indexing='ij')
# True distribution
m1_grid, m2_grid, z_grid = m1_grid.flatten(), m2_grid.flatten(), z_grid.flatten()

bounds = [[0,100],[0,100],[0,2.1]]

mix = DPGMM(bounds)
pop_model =[]

print(np.load(catfile)['redshift'].max())
pop_model.append(mix.density_from_samples(np.array([np.load(catfile)['m1'],np.load(catfile)['m2'],np.load(catfile)['redshift']]).T))
with open('./p_obs.pkl', 'wb') as f:
    dill.dump(pop_model,f)
    
print('done dpgmm')


f_m =  pop_model[0].pdf(np.array([m1_grid, m2_grid, z_grid]).T)
f_m[m1_grid<m2_grid] = 0

with open('./gwdet_default_interpolator.pkl', 'rb') as f:
    pdet = dill.load(f)



p = pdet(np.array([m1_grid, m2_grid, z_grid]).T)
#print(np.sum(p))
#p /= np.sum(p)

eps = 5e-5
f_m[p!=0] /= p[p!=0]
#print(p[(p<eps)*(p!=0)])
f_m[np.isnan(f_m)] = 0

f_m[p<eps] = 0



f_m = np.reshape(f_m, (m_pts,m_pts,z_pts))


interp = RegularGridInterpolator((m_src,m_src,z), f_m, bounds_error=False, fill_value=0)
print('recon')

def pop_recon(m1z,m2z,z):
    p = np.array([interp(np.array([m1z/(1+zi),m2z/(1+zi),zi]).T)*dz/(1+zi)**2 for zi in z])
    m1 = m1z/(1+z)
    m2 = m2z/(1+z)
    p[m1 < m_min] = 0
    p[m1 > m_max] = 0
    p[m2 < m_min] = 0
    p[m2 > m_max] = 0
    p[m1<m2] = 0
    return np.sum(p)



m1z_grid, m2z_grid = np.meshgrid(m,m,indexing='ij')
f_m = []
for m1zi,m2zi in tqdm(zip(m1z_grid.flatten(), m2z_grid.flatten() ), total = len(m1z_grid.flatten()), desc = 'Recon distribution'):
    f_m.append(pop_recon(m1zi,m2zi,z))
    

f_m = np.reshape(f_m, (m_pts,m_pts))

interp = RegularGridInterpolator((m,m), f_m, bounds_error=False, fill_value=0)
with open('./recons_dist.pkl', 'wb') as file:
    dill.dump(interp, file)

f_m = np.sum(f_m, axis=1)*dm

norm = np.sum(f_m)*dm
plt.plot(m, f_m/norm, lw = 0.7, label = '$Recon\ distribution$')

#plt.xlim(0,250)
plt.xlabel('$m_1^z\ [M_\\odot]$')
plt.ylabel('$p(m_1^z)$')
plt.legend(loc = 0, frameon = False)
plt.savefig('recon.pdf', bbox_inches = 'tight')

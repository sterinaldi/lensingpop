import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from simulated_universe import mass_distribution, redshift_distribution, m_min, m_max, z_max

m_pts = 1000
z_pts = 1000

m  = np.linspace(m_min, m_max, m_pts)
z  = np.linspace(0, z_max, z_pts)
dz = z[1]-z[0]

f_m = []

for mi in tqdm(m):
    f_m.append(np.sum(mass_distribution(mi,z)*redshift_distribution(z)*(1+z)*dz))

np.savetxt('true_mass_dist.txt', np.array([m, f_m]).T, header = 'm p')

plt.plot(m, f_m, lw = 0.7)
plt.xlabel('$M\ [M_\\odot]$')
plt.ylabel('$p(M)$')
plt.savefig('true_mass_dist.pdf', bbox_inches = 'tight')
        

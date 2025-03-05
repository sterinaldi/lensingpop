import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares
from lal import antenna
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import dill 
import time
from lensingpop.population import simulated_universe as universe

t0 = time.time()

H0 = universe.h * 100  # Hubble constant in km/s/Mpc
# Luminosity distance function
def d1(x):
    return ((1 + x)**3 * 0.31 + 0.69)**-0.5  # Cosmological parameters

def fr(rs):  # Returns luminosity distance for a redshift
    return 3 * 10**5 * (1 + rs) * quad(d1, 0, rs)[0] / H0

fr = np.vectorize(fr)
# Antenna patterns and angle grids
n = 15  # Number of points for the 4D angle grid

alpha = np.linspace(0, 2 * np.pi, num=n)
psi = np.linspace(0, 2 * np.pi, num=n)
delta = np.arccos(np.linspace(-1, 1, num=n))
io = np.linspace(-1, 1, num=n)

# Antenna responses for Livingston, Hanford, and Virgo detectors
f = antenna.AntennaResponse('L1', alpha, delta, psi=psi, times=1000000000).plus
fx = antenna.AntennaResponse('L1', alpha, delta, psi=psi, times=1000000000).cross
anl = np.outer(f**2, (0.5 * (1 + io**2))**2) + np.outer(fx**2, io**2)
anl = np.reshape(anl, (n, n, n, n))

fh = antenna.AntennaResponse('H1', alpha, delta, psi=psi, times=1000000000).plus
fxh = antenna.AntennaResponse('H1', alpha, delta, psi=psi, times=1000000000).cross
anh = np.outer(fh**2, (0.5 * (1 + io**2))**2) + np.outer(fxh**2, io**2)
anh = np.reshape(anh, (n, n, n, n))

fv = antenna.AntennaResponse('V1', alpha, delta, psi=psi, times=1000000000).plus
fxv = antenna.AntennaResponse('V1', alpha, delta, psi=psi, times=1000000000).cross
anv = np.outer(fv**2, (0.5 * (1 + io**2))**2) + np.outer(fxv**2, io**2)
anv = np.reshape(anv, (n, n, n, n))

# Load precomputed SNRs for each detector
o3l = np.load('./detectability/SNRdata/SNRo3L1.npy')
o3h = np.load('./detectability/SNRdata/SNRo3H1.npy')
o3v = np.load('./detectability/SNRdata/SNRo3V1.npy')

# Precompute alpha(m2/m1) for mass ratio q
al = np.zeros(100)
for j in range(10, 100):
    a = np.zeros(j)
    for i in range(10, j):
        a[i] = o3l[int(100/j*i)][i]
    def ff(x):
        return (np.linspace(10, j, len(a)-10)**x[0] * x[1] - a[10:])
    res_1 = least_squares(ff, np.array([1, a[-1] / j**1]))
    al[j] = res_1.x[0]

    
nsize = 100
# Arrays for m1, q, and z
m1_array = np.linspace(5, 99, nsize)  # Mass m1 from 5 to 100 solar masses
q_array = np.linspace(0.1, 1, nsize)   # Mass ratio q from 0.1 to 1
z_array = np.linspace(1e-6, 2, nsize)   # Redshift from 0 to 3.5

# 3D grid to store detection probabilities
pdet_grid = np.zeros((nsize, nsize, nsize))
snr_grid = np.zeros((nsize, nsize, nsize))

# Detection probability function, generalized to take m1, q (mass ratio), and z (redshift)
def pdet_m1qz(m1, q, z):
    m2 = q * m1  # Compute m2 from the mass ratio
    if m2 > m1:  # If m2 exceeds precomputed SNR range
        return 0
    # Calculate the luminosity distance for the given redshift
    r =  universe.omega.LuminosityDistance(z)
    # Calculate the redshifted SNR for each detector
    alpha_factor = (1 + z)**al[int(np.round(100 * q)) - 1] / r**2
    combined_snr = np.sqrt(1000**2*np.outer(alpha_factor, (o3l[int(m1), int(m2)] * anl + o3h[int(m1), int(m2)] * anh + o3v[int(m1), int(m2)] * anv)))
    
    snr_median = np.median(combined_snr) 
    p_sum =  np.sum((combined_snr > 8)) / (n**4)
    return p_sum, snr_median

# Calculate pdet for every combination of m1, q, z
for i, m1 in tqdm(enumerate(m1_array), total=len(m1_array), desc='Progress'):
    for j, q in enumerate(q_array):
        for k, z in enumerate(z_array):
            pdet_grid[i, j, k], snr_grid[i, j, k] = pdet_m1qz(m1, q, z)
    

# Save file in the same directory as the script
output_path = "./detectability/selfunc_m1qz_source.pkl" 
interp = RegularGridInterpolator(points = (m1_array, q_array, z_array), values = pdet_grid, method = 'linear', bounds_error = False, fill_value = 0.)
with open(output_path, 'wb') as f:
    dill.dump(interp, f)  

print("Selection function saved to {}.".format(output_path))    
   
output_path = "./detectability/snr_m1qz_source.pkl"
interp = RegularGridInterpolator(points = (m1_array, q_array, z_array), values = snr_grid, method = 'linear', bounds_error = False, fill_value = 0.)
with open(output_path, 'wb') as f:
    dill.dump(interp, f)  

print("SNR function saved to {}.".format(output_path)) 

print("Done. Time used = {} s".format(time.time()-t0))

#!/users/hoi-tim.cheung/.conda/envs/py38/bin/python

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import truncnorm
import scipy.integrate as integrate
from astropy.cosmology import Planck15, z_at_value
cosmo = Planck15
import astropy.units as u
import gwdet 
import time 

pdetfunction = gwdet.detectability()
kappa = 1.0
zmin = 0
zmax = 15.0
rdir = '/users/hoi-tim.cheung/dmm'
norm = integrate.quad(lambda x: (1+x)**(kappa-1)*cosmo.differential_comoving_volume(x).to(u.Gpc**3/u.sr).value, zmin, zmax)[0]
def LuminosityDistance(redshift):
    dL = cosmo.luminosity_distance(redshift).value
    return dL

def inverse_transform_sampling(bins, pdf, nSamples=1):
    cumValue = np.zeros(bins.shape)
    cumValue[1:] = np.cumsum(pdf[1:] * np.diff(bins))
    cumValue /= cumValue.max()
    inv_cdf = interp1d(cumValue, bins)
    r = np.random.rand(nSamples)
    sample = inv_cdf(r)
    return inv_cdf(r)


def powerlaw_pdf(mu, alpha, mu_min, mu_max):

    if -alpha != -1:
        norm_mass = (mu_max **(1+alpha) - mu_min**(1+alpha))  /  (1+alpha)
    else:
        norm_mass = np.log(mu_max ) - np.log(mu_min)

    p_mu = mu**alpha / norm_mass
    p_mu[(mu<mu_min) + (mu>mu_max)] = 0
    return p_mu

def TruncNormSampler(clip_a, clip_b, mean, std, Nsamples):

    a, b = (clip_a - mean) / std, (clip_b - mean) / std

    return truncnorm.rvs(a,b,size=Nsamples ) * std + mean

print('load data')

z_grid = np.linspace(zmin,zmax, 400)
z_eval = interp1d(LuminosityDistance(z_grid), z_grid)
data = np.load(rdir+'/lens_dat.npz')
m1= data['m1']
m2 = data['m2']
redshift= data['z']
zeff1 = data['zeff1']
zeff2 = data['zeff2']
dl1 = data['dl1']
dl2 = data['dl2']



SNR_threshold = 8
Npos = 1000
sigma_mass = 0.08 * SNR_threshold
sigma_symratio = 0.022 * SNR_threshold
sigma_theta = 0.21 * SNR_threshold
###############################################
Np = m1.size

snr1 = pdetfunction.snr(m1,m2,zeff1)
snr2 = pdetfunction.snr(m1,m2,zeff2)
snr_obs1 = np.zeros(snr1.shape)
snr_obs2 = np.zeros(snr2.shape)


print('calculating snr obs')

for i in range(snr1.size):
    snr_obs1[i] = snr1[i] + TruncNormSampler( -snr1[i],np.inf, 0.0, 1.0, 1)
    snr_obs2[i] = snr2[i] + TruncNormSampler( -snr2[i],np.inf, 0.0, 1.0, 1)


################## Compute chrip mass and symmetry ratio ##################
Mc = (1+ redshift) * (m1*m2) ** (3./5.) / (m1+m2)** (1./5.)
sym_mass_ratio = (m1*m2)  / (m1+m2)** 2

m1p1 = np.zeros((Np,Npos))
m2p1 = np.zeros((Np,Npos))
zp1 = np.zeros((Np,Npos))
m1p2 = np.zeros((Np,Npos))
m2p2 = np.zeros((Np,Npos))
zp2 = np.zeros((Np,Npos))

def measurement_uncertainty(Mc_z, smr, dl, zeff, snr_opt, snr_obs, N = 1000):
    Mc_center = Mc_z * np.exp( np.random.normal(0, sigma_mass / snr_obs, 1) )
    Mc_obs = Mc_center * np.exp( np.random.normal(0, sigma_mass / snr_obs, N) )
    ################## generate symmetry ratio noise by using truncated normal distribution ##################
    symratio_obs = TruncNormSampler( 0.0, 0.25, smr, sigma_symratio / snr_obs, N)

    ################## compute redshifted m1 and m2 ##################
    M = Mc_obs / symratio_obs ** (3./5.)

    m1_obsz = 0.5 * M * (1 + np.sqrt(1 - 4 * symratio_obs) )
    m2_obsz = 0.5 * M * (1 - np.sqrt(1 - 4 * symratio_obs) )
    m1s = m1_obsz / (1 + zeff )
    m2s = m2_obsz / (1 + zeff )
    dl_obs = dl * pdetfunction.snr(m1s,m2s,np.repeat(zeff,1000)) /snr_opt

    z_obs = z_eval(dl_obs)

    m1_obs = m1_obsz / (1 + z_obs )
    m2_obs = m2_obsz / (1 + z_obs )


    return m1_obs, m2_obs, z_obs

print('generating posterior')
import time
for i in range(m1.size):
    ################## chrip mass noise ##################
    t1 = time.time()
    m1p1[i], m2p1[i], zp1[i] = measurement_uncertainty(Mc[i], sym_mass_ratio[i], dl1[i], zeff1[i], snr1[i], snr_obs1[i], Npos)
    m1p2[i], m2p2[i], zp2[i] = measurement_uncertainty(Mc[i], sym_mass_ratio[i], dl2[i], zeff2[i], snr2[i], snr_obs1[i], Npos)
    
    print(time.time()-t1)
np.savez(rdir+'/lensed_posterior.npz',m1p1 = m1p1,m1p2= m1p2,m2p1 = m2p1,m2p2 = m2p2,zp1 = zp1,zp2 = zp2)



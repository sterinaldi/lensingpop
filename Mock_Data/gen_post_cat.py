import numpy as np
import argparse
from scipy.stats import truncnorm   
from scipy.interpolate import interp1d
from simulated_universe import *
from figaro.load import _find_redshift
from pathlib import Path
import os
import sys
import gwdet
import time

np.random.seed(0)

pdetfunction = gwdet.detectability()

############################################################################################################
""" 
Here, we generate the mass posterior by following the procedure in 
Appendix A of https://iopscience.iop.org/article/10.3847/2041-8213/ab77c9/pdf 
"""



parser = argparse.ArgumentParser(description='Generate population and posterior samples.')
parser.add_argument('-i', dest = 'file', type=str, help = 'input file')
parser.add_argument('--Npos',type=int,help='number of posterior samples per event',default=1000)
args = parser.parse_args()
file = Path(args.file)
Npos = int(args.Npos) # nunber of posterior samples per event

def TruncNormSampler(clip_a, clip_b, mean, std, Nsamples):
    a, b = (clip_a - mean) / std, (clip_b - mean) / std
    return truncnorm.rvs(a,b,size=Nsamples ) * std + mean


################# Uncertainty parameters ##################
SNR_threshold  = 8
sigma_mass     = 0.08  * SNR_threshold
sigma_symratio = 0.022 * SNR_threshold
sigma_theta    = 0.21  * SNR_threshold


### function to generate measurement uncertainty ##### 

def measurement_uncertainty(Mc_z, smr, dl, z, snr_opt, snr_obs, N = 1000):
    Mc_center = Mc_z * np.exp( np.random.normal(0, sigma_mass / snr_obs, 1) )
    Mc_obs = Mc_center * np.exp( np.random.normal(0, sigma_mass / snr_obs, N) )
    ################## generate symmetry ratio noise by using truncated normal distribution ##################
    symratio_obs = TruncNormSampler( 0.0, 0.25, smr, sigma_symratio / snr_obs, N)

    ################## compute redshifted m1 and m2 ##################
    M = Mc_obs / symratio_obs ** (3./5.)

    m1_obsz = 0.5 * M * (1 + np.sqrt(1 - 4 * symratio_obs) )
    m2_obsz = 0.5 * M * (1 - np.sqrt(1 - 4 * symratio_obs) )
    m1s = m1_obsz / (1 + z )
    m2s = m2_obsz / (1 + z )
    dl_obs = dl * pdetfunction.snr(m1s,m2s,np.repeat(z_eval(dl),1000)) /snr_opt
    z_obs = z_eval(dl_obs)
    
    return m1_obsz, m2_obsz, z_obs



################## Read m1, m2, z, snr from mock catalog ##################
data = np.load(file)
m1 = data['m1']
m2 = data['m2']
redshift = data['redshift']
dl = data['DL']
snr = data['snr']
del data
# make a interpolator to evaluate redshift from luminosity distance
z_grid = np.linspace(0.01,z_max, 400)
z_eval = interp1d(omega.LuminosityDistance(z_grid), z_grid)

################## Generate Gaussian noise for SNR ##################
snr_obs = np.zeros(snr.shape)
for i in range(snr.size):
    snr_obs[i] = snr[i] + TruncNormSampler( -snr[i],np.inf, 0.0, 1.0, 1)

################## Compute chrip mass and symmetry ratio ##################
Mz             = (1+redshift) * (m1*m2)**(3./5.) / (m1+m2)**(1./5.)
sym_mass_ratio = (m1*m2)  / (m1+m2)** 2  

################## generate posterior sample for m1 and m2 ##################
m1_posterior = np.zeros((Mz.size,Npos))
m2_posterior = np.zeros((Mz.size,Npos))
z_posterior = np.zeros((Mz.size,Npos))

for i in tqdm(range(0,Mz.size), desc = 'Posteriors'):
    m1_posterior[i], m2_posterior[i], z_posterior[i] = measurement_uncertainty(Mz[i], sym_mass_ratio[i], dl[i], redshift[i], snr[i], snr_obs[i], Npos)
    
################## Save the posterior ##################
"""
Warning: the 'redshift' entry does not match with 'z_posterior' since the latter is computed from the luminosity distance (which is affected by the presence of the lens).
"""
np.savez(Path('./m1m2z_posterior_PPD_afterSelection.npz'), m1=m1, m2=m2, redshift=redshift, m1_posterior = m1_posterior, m2_posterior = m2_posterior, z_posterior = z_posterior)

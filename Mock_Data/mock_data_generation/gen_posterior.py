import numpy as np
import argparse
from scipy.stats import truncnorm   
np.random.seed(0)


parser = argparse.ArgumentParser(description='Generate population and posterior samples.')
parser.add_argument('--N',type=int,help='number of events in the catalog',default=1000000)
parser.add_argument('--Npos',type=int,help='number of posterior samples per event',default=1000)
args = parser.parse_args()
N = int(args.N) # Sunber of events 
Npos = int(args.Npos) # nunber of posterior samples per event

def TruncNormSampler(clip_a, clip_b, mean, std, Nsamples):

    a, b = (clip_a - mean) / std, (clip_b - mean) / std
    return truncnorm.rvs(a,b,size=Nsamples ) * std + mean


################################## Generate posterior samples ##################################


################## Read m1, m2, z, snr from mock catalog ##################
data = np.load('PowerlawplusPeakplusDelta{:.0f}Samples_afterSelection.npz'.format(N))
m1 = data['m1']
m2 = data['m2']
redshift = data['redshift']
snr = data['snr']
del data 

############################################################################################################
""" 
Here, we generate the mass posterior by following the procedure in 
Appendix A of https://iopscience.iop.org/article/10.3847/2041-8213/ab77c9/pdf 
"""
################## Parameters ##################
SNR_threshold = 8
sigma_mass = 0.08 * SNR_threshold
sigma_symratio = 0.022 * SNR_threshold
sigma_theta = 0.21 * SNR_threshold
################## Generate Gaussian noise for SNR ##################
SNR_obs = np.zeros(snr.shape)
for i in range(snr.size):
    SNR_obs[i] = snr[i] + TruncNormSampler( -snr[i],np.inf, 0.0, 1.0, 1)




################## Compute chrip mass and symmetry ratio ##################
Mz = (1+ redshift) * (m1*m2) ** (3./5.) / (m1+m2)** (1./5.)  
sym_mass_ratio = (m1*m2)  / (m1+m2)** 2  

################## generate posterior sample for m1 and m2 ##################
m1_posterior = np.zeros((Mz.size,Npos))
m2_posterior = np.zeros((Mz.size,Npos))
print('generating posterior')
for i in range(0,Mz.size):
    ################## chrip mass noise ##################
    Mz_obs = Mz[i] * np.exp( np.random.normal(0, sigma_mass / SNR_obs[i], Npos) )

    ################## generate symmetry ratio noise by using truncated normal distribution ##################
    symratio_obs = TruncNormSampler( 0.0, 0.25, sym_mass_ratio[i], sigma_symratio / SNR_obs[i], Npos)

    ################## compute redshifted m1 and m2 ##################
    M = Mz_obs / symratio_obs ** (3./5.)

    m1_obsz = 0.5 * M * (1 + np.sqrt(1 - 4 * symratio_obs) )
    m2_obsz = 0.5 * M * (1 - np.sqrt(1 - 4 * symratio_obs) )

    m1_posterior[i] = m1_obsz / (1 + redshift[i] )
    m2_posterior[i] = m2_obsz / (1 + redshift[i] )
    
################## Save the posterior ##################
np.savez('m1m2posterior_PPD_afterSelection{:.0f}.npz'.format(N), m1=m1, m2=m2, redshift=redshift, m1_posterior = m1_posterior, m2_posterior = m2_posterior)




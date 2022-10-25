import numpy as np
import argparse
from scipy.stats import truncnorm   
from simulated_universe import *
from figaro.load import _find_redshift
from pathlib import Path
import time
import dill
np.random.seed(0)
# Here come all the definitions used in this script
###########################################################################################################


""" 
Here, we generate the mass posterior by following the procedure in 
Appendix A of https://iopscience.iop.org/article/10.3847/2041-8213/ab77c9/pdf 
"""

t0 = time.time

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

    while True:
        index = np.where((m1s>m_max) + (m1s<m_min) + (m2s>m_max) + (m2s<m_min)   )[0]
        n_out = index.size
        if not n_out>0: break
        Mc_obs = Mc_center * np.exp( np.random.normal(0, sigma_mass / snr_obs, n_out) )
   ################## generate symmetry ratio noise by using truncated normal distribution ##################
        symratio_obs = TruncNormSampler( 0.0, 0.25, smr, sigma_symratio / snr_obs, n_out)

    ################## compute redshifted m1 and m2 ##################
        M = Mc_obs / symratio_obs ** (3./5.)
        temp1 = 0.5 * M * (1 + np.sqrt(1 - 4 * symratio_obs) )
        temp2 = 0.5 * M * (1 - np.sqrt(1 - 4 * symratio_obs) )
        m1s[index] = temp1 / (1 + z )
        m2s[index] = temp2 / (1 + z )
        m1_obsz[index] = temp1
        m2_obsz[index] = temp2
    z_obs = np.zeros(N)
    snrr = f_snr(np.array([m1s,m2s,np.repeat(z,N)]).T)/dl
    dl_obs = dl * snrr / snr_opt
    for index in range(N):
        z_obs[index] = _find_redshift(omega,dl_obs[index])

    return m1_obsz, m2_obsz, z_obs

##################Load SNR interpolator ###############3   

with open('snr_m1m2z.pkl','rb') as f:
    f_snr = dill.load(f)


################## Read m1, m2, z, snr from mock catalog ##################
data = np.load(file)
m1 = data['m1']
m2 = data['m2']
redshift = data['redshift']
dl = data['DL']
snr = data['snr']

################## Filter the event with mass ratio m1/m2 >6.5 ################################
index = np.where((snr>0) *(m1/m2<6.5))[0]
m1 = m1[index]
m2 = m2[index]
redshift = redshift[index]
dl = dl[index]
snr = snr[index]
del data
################## Generate Gaussian noise for SNR ##################
snr_obs = np.zeros(snr.shape)
for ii in range(snr.size):
    snr_obs[ii] = snr[ii] + TruncNormSampler( -snr[ii],np.inf, 0.0, 1.0, 1)

################## Compute chrip mass and symmetry ratio ##################
Mz             = (1+redshift) * (m1*m2)**(3./5.) / (m1+m2)**(1./5.)
sym_mass_ratio = (m1*m2)  / (m1+m2)** 2  


################## generate posterior sample for m1 and m2 ##################
m1_posterior = np.zeros((Mz.size,Npos))
m2_posterior = np.zeros((Mz.size,Npos))
z_posterior = np.zeros((Mz.size,Npos))

for i in tqdm(range(0,Mz.size), desc = 'Posteriors'):
    m1_posterior[i], m2_posterior[i], z_posterior[i] = measurement_uncertainty(Mz[i], sym_mass_ratio[i], dl[i], redshift[i], snr[i], snr_obs[i], Npos)
#print(time.time()-t0)
################## Save the posterior ##################
"""
Warning: the 'redshift' entry does not match with 'z_posterior' since the latter is computed from the luminosity distance (which is affected by the presence of the lens).
"""
np.savez(Path('./m1m2z_posterior_PPD_afterSelection_'+str(m1.size)+'_unlensed.npz'), m1=m1, m2=m2, redshift=redshift, m1_posterior = m1_posterior, m2_posterior = m2_posterior, z_posterior = z_posterior)

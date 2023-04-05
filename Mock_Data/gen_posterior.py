import numpy as np
import argparse
from scipy.stats import truncnorm   
from simulated_universe import *
from xeff_pop import *
from figaro.load import _find_redshift
from pathlib import Path
import time
import dill

###########################################################################################################
""" 
Here, we generate the mass posterior by following the procedure in 
Appendix A of https://iopscience.iop.org/article/10.3847/2041-8213/ab77c9/pdf 
"""

def TruncNormSampler(clip_a, clip_b, mean, std, Nsamples):
    a, b = (clip_a - mean) / std, (clip_b - mean) / std
    return truncnorm.rvs(a,b,size=Nsamples ) * std + mean

### function to generate measurement uncertainty ##### 

def measurement_uncertainty(Mc_z, mr, dl, z, snr_opt, snr_obs, N = 1000):
    Mc_center = Mc_z * np.exp( np.random.normal(0, sigma_mass / snr_obs, 1) )
    Mc_obs = Mc_center * np.exp( np.random.normal(0, sigma_mass / snr_obs, N) )
   ################## generate symmetry ratio noise by using truncated normal distribution ##################
    mr_center = 1./(1+np.exp(-np.random.normal(loc = np.log(mr/(1-mr)), scale = sigma_q, size = 1)))
    ratio_obs = 1./(1+np.exp(-np.random.normal(loc = np.log(mr_center/(1-mr_center)), scale = sigma_q, size = N)))
    symratio_obs = ratio_obs/(1+ratio_obs)**2

    ################## compute redshifted m1 and m2 ##################
    M = Mc_obs / symratio_obs ** (3./5.)
    m1z_obs = 0.5 * M * (1 + np.sqrt(1 - 4 * symratio_obs) )
    m2z_obs = 0.5 * M * (1 - np.sqrt(1 - 4 * symratio_obs) )
    m1s = m1z_obs / (1 + z )
    m2s = m2z_obs / (1 + z )

    while True:
        index = np.where((m1s>m_max) + (m1s<m_min) + (m2s>m1s) + (m2s<m_min)   )[0]
        n_out = index.size
        if not n_out>0: break
        Mc_obs = Mc_center * np.exp( np.random.normal(0, sigma_mass / snr_obs, n_out) )
   ################## generate symmetry ratio noise by using truncated normal distribution ##################
        ratio_obs = 1./(1+np.exp(-np.random.normal(loc = np.log(mr_center/(1-mr_center)), scale = sigma_q, size = n_out)))
        symratio_obs = ratio_obs/(1+ratio_obs)**2

    ################## compute redshifted m1 and m2 ##################
        M = Mc_obs / symratio_obs ** (3./5.)
        temp1 = 0.5 * M * (1 + np.sqrt(1 - 4 * symratio_obs) )
        temp2 = 0.5 * M * (1 - np.sqrt(1 - 4 * symratio_obs) )
        m1s[index] = temp1 / (1 + z )
        m2s[index] = temp2 / (1 + z )
        m1z_obs[index] = temp1
        m2z_obs[index] = temp2
    
    snrr = f_snr(np.array([m1s,m2s,np.repeat(z,N)]).T)/dl
    dl_obs = dl * snrr / snr_obs 
    # compute the redshift 
    z_obs = np.array([_find_redshift(omega,dl) for dl in dl_obs])

    
    return m1z_obs, m2z_obs, z_obs



np.random.seed(0)
t0 = time.time

parser = argparse.ArgumentParser(description='Generate population and posterior samples.')
parser.add_argument('-i', dest = 'file', type=str, help = 'input file')
parser.add_argument('--Npos',type=int,help='number of posterior samples per event',default=1000)
parser.add_argument("-L", dest = "L", action = 'store_true', help = "Generate lensed population", default = False)

args = parser.parse_args()
file = Path(args.file)
Npos = int(args.Npos) # nunber of posterior samples per event
snr_file =  Path("./selection_functions/snr_m1m2z.pkl").resolve()
################# Uncertainty parameters ##################
SNR_threshold  = 8
sigma_mass     = 0.08  * SNR_threshold
sigma_symratio = 0.022 * SNR_threshold
sigma_theta    = 0.21  * SNR_threshold
sigma_q        = 1.0671


##################Load SNR interpolator ###############3   

with open(snr_file,'rb') as f:
    f_snr = dill.load(f)


################## Read m1, m2, z, snr from mock catalog ##################
data = np.load(file)
m1 = data['m1']
m2 = data['m2']
redshift = data['redshift']
dl = data['DL']
snr = data['snr']
#print(snr[snr<=0])
#q= m2/m1
#print(q[1/q>6.5])


################## Generate Gaussian noise for SNR ##################
snr_obs = np.zeros(snr.shape)
for ii in range(snr.size):
    snr_obs[ii] = snr[ii] + np.random.normal( loc= 0.0, scale = 1.0, size = 1)
#    snr_obs[ii] = snr[ii] + TruncNormSampler( -np.inf,np.inf, 0.0, 1.0, 1)


    
    
################## Filter the event with mass ratio m1/m2 >10.0 ################################
index = np.where(snr_obs>=8)[0]
m1 = m1[index]
m2 = m2[index]
redshift = redshift[index]
dl = dl[index]
snr = snr[index]
del data
################## Compute chrip mass and symmetry ratio ##################
Mz             = (1+redshift) * (m1*m2)**(3./5.) / (m1+m2)**(1./5.)
mass_ratio     = m2/m1


################## generate posterior sample for m1z, m2z and z ##################
m1z_posterior = np.zeros((Mz.size,Npos))
m2z_posterior = np.zeros((Mz.size,Npos))
z_posterior = np.zeros((Mz.size,Npos))

for i in tqdm(range(0,Mz.size), desc = 'Posteriors'):
    m1z_posterior[i], m2z_posterior[i], z_posterior[i] = measurement_uncertainty(Mz[i], mass_ratio[i], dl[i], redshift[i], snr[i], snr_obs[i], Npos)



# Initialize the spin population model
spin_pop = Gaussian_spin_distribution(**spin_pars)


# Generate the spin catalog

Nspin = int(m1.size/2) if args.L else m1.size
chi_eff = spin_pop.sample(Nsamples=Nspin)

# generate the posterior for spin

if args.L: # 
    chi_eff = np.concatenate([chi_eff, chi_eff])
    eff_posterior = spin_posterior_samples(chi_eff, Nspin*2)
else:            
    eff_posterior = spin_posterior_samples(chi_eff, Nspin)    

################## Save the posterior ##################
"""
Warning: the 'redshift' entry does not match with 'z_posterior' since the latter is computed from the luminosity distance (which is affected by the presence of the lens).
"""
if args.L:
    output_file=Path('./catalog/m1m2zxeff_posterior_PPD_afterSelection_lensed'+str(m1.size)+'.npz')
else:
    output_file=Path('./catalog/m1m2zxeff_posterior_PPD_afterSelection_unlensed'+str(m1.size)+'.npz')


np.savez(output_file, m1=m1, m2=m2, redshift=redshift, xeff=chi_eff,
         m1_posterior = m1z_posterior, m2_posterior = m2z_posterior,
         z_posterior = z_posterior, xeff_posterior=eff_posterior)

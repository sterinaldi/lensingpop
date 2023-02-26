import numpy as np
import os 
import time
import scipy.stats as ss
import argparse
import sys

def TruncNormSampler(clip_a, clip_b, mean, std, Nsamples):
    a, b = (clip_a - mean) / std, (clip_b - mean) / std
    return ss.truncnorm.rvs(a,b,size=Nsamples ) * std + mean

def TruncNormPdf(x, clip_a, clip_b, mean, std):
    a, b = (clip_a - mean) / std, (clip_b - mean) / std
    return ss.truncnorm.pdf(x,a,b) 

class Gaussian_spin_distribution():
    def __init__(self, wide = False, **model_pars):
        self.mu_eff = model_pars['mu_eff']
        self.sigma_eff = model_pars['sigma_eff2'] if wide else model_pars['sigma_eff']    
        self.eff_bounds=[-1,1]
    
    def sample(self, Nsamples):
        s = TruncNormSampler(self.eff_bounds[0],self.eff_bounds[1],self.mu_eff,self.sigma_eff,Nsamples)
        return s

    def prob(self, x):
        return TruncNormPdf(x,self.eff_bounds[0],self.eff_bounds[1],self.mu_eff,self.sigma_eff)    
       
def spin_posterior_samples(chi_eff, Nevent, posterior_effsigma = 0.2, Npos = 1000):
    x_eff = []
    for i in range(Nevent):
        c1 = TruncNormSampler(-1,1,chi_eff[i],posterior_effsigma,1) 
        x_eff.append(TruncNormSampler(-1,1,c1,posterior_effsigma,Npos))
    return np.array(x_eff).reshape(Nevent, Npos)


# parameters for our spin (effective spin,  effective precessing spin) population, which is a multivariate normal distribution + peak 
spin_pars = {'mu_eff': 0.06, # mean of effective spin
              'sigma_eff': 0.12, # standard deviation of effective spin
              'sigma_eff2':0.5}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate population and posterior samples.')
    parser.add_argument("-L", dest = "L", action = 'store_true', help = "Generate lensed population", default = False) 
    # True to turn on the lensing mechanism
    parser.add_argument("-N", type = int, help = "number of population events", default=1986) 
    args = parser.parse_args()    
    N = args.N # number of binary events

    cdir = os.path.dirname(os.path.dirname(sys.path[0]))
    np.random.seed(0)
    start = time.time()

    
    if args.L:
        savedir = './catalog/spin_data_'+str(N)+'_lensed.npz'    
    else:
        savedir = './catalog/spin_data_'+str(N)+'_unlensed.npz'

    # Initialize the spin population model
    spin_pop = Gaussian_spin_distribution(**spin_pars)

    # Generate the spin catalog
    chi_eff = spin_pop.sample(Nsamples=N)

    # generate the posterior
    if args.L: # 
        chi_eff = np.concatenate([chi_eff, chi_eff])
        eff_posterior = spin_posterior_samples(chi_eff, N*2)
    else:            
        eff_posterior = spin_posterior_samples(chi_eff, N)

    np.savez(savedir,chi_eff=chi_eff,eff_posterior=eff_posterior)
    



import numpy as np
import os 
import matplotlib.pyplot as plt
import time
import scipy.stats as ss
import os 
import sys
cdir = os.path.dirname(os.path.dirname(sys.path[0]))

np.random.seed(0)
start = time.time()

spin_pars = {'mu_eff': 0.06,
  'sigma_eff': 0.12,
  'mu_p': 0.21,
  'sigma_p': 0.09,
  'rho': 0.12}

def TruncNormSampler(clip_a, clip_b, mean, std, Nsamples):
	a, b = (clip_a - mean) / std, (clip_b - mean) / std
	return ss.truncnorm.rvs(a,b,size=Nsamples ) * std + mean

class Gaussian_spin_distribution():
    def __init__(self, **model_pars):
        mu_eff = model_pars['mu_eff']
        mu_p = model_pars['mu_p']
        sigma_eff = model_pars['sigma_eff']
        sigma_p = model_pars['sigma_p']
        rho = model_pars['rho']
        mean = [mu_eff, mu_p]
        cov = [[sigma_eff**2, rho*sigma_eff*sigma_p],[rho*sigma_eff*sigma_p,sigma_p**2]]
        self.model=ss.multivariate_normal(mean=mean, cov=cov)
        self.normalization = self.GetNormalization()
        
    def GetNormalization(self,Nbins=80):
        eff_grid = np.linspace(-1,1,Nbins) # Redshifted mass 1
        p_grid = np.linspace(0,1,Nbins) # Redshifted mass 1
        grids=[eff_grid,p_grid]
        x,y = np.meshgrid(eff_grid, p_grid, indexing='ij')
        return 2*np.mean(self.model.pdf(np.array([x.flatten(),y.flatten()]).T))
        

    def sample(self, Nsamples):
        chi_eff, chi_p =  self.model.rvs(size=Nsamples).T
        while True:
            index = np.where((chi_eff>1) + (chi_eff<-1) + (chi_p>1) + (chi_p<0)   )[0]
            n_out = index.size
            if not n_out>0: break
            chi_eff[index], chi_p[index] = self.model.rvs(size=n_out).T
            
        return chi_eff,chi_p
    def prob(self, chi_eff, chi_p):
        return self.model.pdf(np.array([chi_eff,chi_p]).T) / self.normalization




spin_pop = Gaussian_spin_distribution(spin_pars)

N = 1986
chi_eff, chi_p = spin_pop.sample(Nsamples=N)

posterior_effsigma = 0.2
posterior_psigma = 0.2
Npos = 1000
eff_posterior = np.zeros((N,Npos))
p_posterior = eff_posterior.copy()

for i in range(N):

	eff_posterior[i] = TruncNormSampler(-1,1,chi_eff[i],posterior_effsigma,Npos)
	p_posterior[i] = TruncNormSampler(0,1,chi_p[i],posterior_psigma,Npos)


np.savez(cdir+'/Mock_Data/spin_data.npz',chi_eff=chi_eff,chi_p=chi_p,eff_posterior=eff_posterior,p_posterior=p_posterior)

  
# posterior











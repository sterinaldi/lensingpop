import numpy as np
import os 
import time
import scipy.stats as ss
import os 
import argparse
import sys
cdir = os.path.dirname(os.path.dirname(sys.path[0]))

np.random.seed(0)
start = time.time()
# parameters for our spin (effective spin,  effective precessing spin) population, which is a multivariate normal distribution + peak 
spin_pars = {'mu_eff': 0.06, # mean of effective spin
  'sigma_eff': 0.12, # standard deviation of effective spin
  'mu_p': 0.21, # mean of effective precessing spin
  'sigma_p': 0.09, # standard deviation of effective precessing spin 
  'rho': 0.12, # parameter in for the covariance matrix
  'lambda':0.2, # fraction of the peak 
  'peak1_mean':0.3,
  'peak2_mean':0.35,
  'peak_sd':0.03}
def TruncNormSampler(clip_a, clip_b, mean, std, Nsamples):
    a, b = (clip_a - mean) / std, (clip_b - mean) / std
    return ss.truncnorm.rvs(a,b,size=Nsamples ) * std + mean

class Gaussian_spin_distribution():
    def __init__(self, peak = False, **model_pars):
        mu_eff = model_pars['mu_eff']
        mu_p = model_pars['mu_p']
        sigma_eff = model_pars['sigma_eff']
        sigma_p = model_pars['sigma_p']
        rho = model_pars['rho']
        self.peak = peak
        self.lambda_peak= model_pars['lambda']
        self.peak1_mean = model_pars['peak1_mean']
        self.peak2_mean = model_pars['peak2_mean']
        self.peak_sd = model_pars['peak_sd']
        self.eff_bounds=[-1,1]
        self.p_bounds=[0,1]
        

        self.mean = np.array([mu_eff, mu_p])
        self.cov = np.array([[sigma_eff**2, rho*sigma_eff*sigma_p],[rho*sigma_eff*sigma_p,sigma_p**2]])

        self.model=ss.multivariate_normal(mean=self.mean, cov=self.cov)
        self.normalization = [1,1]
#        self.normalization = self.GetNormalization()
#        print(self.normalization)
    
    def GetGrid(self, n_pts=2000):
        xeff  = np.linspace(self.eff_bounds[0], self.eff_bounds[1], n_pts)
        xp = np.linspace(self.p_bounds[0],self.p_bounds[1],n_pts)
        xeff_grid, xp_grid = np.meshgrid(xeff, xp,indexing='ij')
        return xeff_grid.flatten(), xp_grid.flatten(), xeff, xp
    """   
    def GetNormalization(self):
        x,y = self.GetGrid()[:2]
        if self.peak:
            return [2*np.mean(self.model.pdf(np.array([x,y]).T)),2*np.mean(self.narrow_peak_pdf(np.array([x,y])))]
        else:
            return [2*np.mean(self.model.pdf(np.array([x,y]).T))]
    """
    def sample(self, Nsamples):
    
        if self.peak:
            Npeak = int(Nsamples * self.lambda_peak)
            Nnormal = Nsamples - Npeak
            chi_eff, chi_p = self.model.rvs(size=Nnormal).T
            while True:
                index = np.where((chi_eff>1) + (chi_eff<-1) + (chi_p>1) + (chi_p<0)   )[0]
                n_out = index.size
                if not n_out>0: break
                chi_eff[index], chi_p[index] = self.model.rvs(size=n_out).T
            chi_eff = np.append(chi_eff, self.narrow_peak_rvs(self.peak1_mean, self.peak_sd, size=Npeak) )
            chi_p = np.append(chi_p, self.narrow_peak_rvs(self.peak2_mean, self.peak_sd, size=Npeak) )

        else: 
            Nnormal = Nsamples
            chi_eff, chi_p =  self.model.rvs(size=Nnormal).T

            while True:
                index = np.where((chi_eff>1) + (chi_eff<-1) + (chi_p>1) + (chi_p<0)   )[0]
                n_out = index.size
                if not n_out>0: break
                chi_eff[index], chi_p[index] = self.model.rvs(size=n_out).T
            
        return chi_eff,chi_p
    def narrow_peak_pdf(self, x):
        xeff, xp = x
        p_xeff = np.exp(-(xeff-self.peak1_mean)**2/(2*self.peak_sd**2))/(np.sqrt(2*np.pi)*self.peak_sd)
        p_xp = np.exp(-(xp-self.peak2_mean)**2/(2*self.peak_sd**2))/(np.sqrt(2*np.pi)*self.peak_sd)
        #print(p_xeff, p_xp)
        return p_xeff * p_xp
    def narrow_peak_rvs(self, mean, sd, size): 
        return ss.norm.rvs(mean,sd,size)
    """
    def pdf(self, chi_eff, chi_p):
        if self.peak:
            f = self.lambda_peak
            return  (1-f)*self.model.pdf(np.array([chi_eff,chi_p]).T)/self.normalization[0] + f*self.narrow_peak_pdf(np.array([chi_eff,chi_p]) ) / self.normalization[1]  
        else:
            return self.model.pdf(np.array([chi_eff,chi_p]).T) / self.normalization
    """
    def pdf(self, chi_eff, chi_p,x=True):
        if self.peak:
            if x:
                f = self.lambda_peak
            else:
                f = 0
            return  (1-f)*self.model.pdf(np.array([chi_eff,chi_p]).T)/self.normalization[0] + f*self.narrow_peak_pdf(np.array([chi_eff,chi_p]) ) / self.normalization[1]  
        else:
            return self.model.pdf(np.array([chi_eff,chi_p]).T) / self.normalization

        
    def marginal_pxeff(self,x):
        f = self.lambda_peak if self.peak else 0
        p1 = np.exp(-(x-self.mean[0])**2/(2*self.cov[0,0]**2))/(np.sqrt(2*np.pi)*self.cov[0,0])
        p2 = np.exp(-(x-self.peak1_mean)**2/(2*self.peak_sd**2))/(np.sqrt(2*np.pi)*self.peak_sd)
        
        return (1-f)*p1 + f*p2
    
    def marginal_pxp(self):
        f = self.lambda_peak if self.peak else 0
        p1 = np.exp(-(x-self.mean[1])**2/(2*self.cov[1,1]**2))/(np.sqrt(2*np.pi)*self.cov[1,1])
        p2 = np.exp(-(x-self.peak1_mean)**2/(2*self.peak_sd**2))/(np.sqrt(2*np.pi)*self.peak_sd)
        
        return (1-f)*p1 + f*p2
       
def spin_posterior(chi_eff, chi_p, Nevent, posterior_effsigma = 0.2, posterior_psigma = 0.2, Npos = 1000):
    x_eff = []
    x_p = []
    for i in range(Nevent):
        c1 = TruncNormSampler(-1,1,chi_eff[i],posterior_effsigma,1) 
        c2 = TruncNormSampler(0,1,chi_p[i],posterior_effsigma,1)
        x_eff.append(TruncNormSampler(-1,1,c1,posterior_effsigma,Npos))
        x_p.append(TruncNormSampler(0,1,c2,posterior_psigma,Npos))
    return np.array(x_eff).reshape(Nevent, Npos), np.array(x_p).reshape(Nevent, Npos)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate population and posterior samples.')
    parser.add_argument("-L", dest = "L", action = 'store_true', help = "Generate lensed population", default = False) # True to turn on the lensing mechanism
    parser.add_argument("-N", type = int, help = "number of population events", default=1986) 
    
    args = parser.parse_args()
    N = args.N # number of binary events
    

    if args.L:
        savedir = './spin_data_lensed_'+str(N)+'.npz'    
    else:
        savedir = './spin_data_unlensed_'+str(N)+'.npz'

    # Initialize the spin population model
    spin_pop = Gaussian_spin_distribution(**spin_pars)

    # Generate the spin catalog
    chi_eff, chi_p = spin_pop.sample(Nsamples=N)

    # generate the posterior
    if args.L: # 
        chi_eff = np.concatenate([chi_eff, chi_eff])
        chi_p = np.concatenate([chi_p, chi_p])
        eff_posterior, p_posterior = spin_posterior(chi_eff, chi_p, N*2)
    else:            
        eff_posterior, p_posterior = spin_posterior(chi_eff, chi_p, N)

    #np.savez(cdir+'/Mock_Data/spin_data.npz',chi_eff=chi_eff,chi_p=chi_p,eff_posterior=eff_posterior,p_posterior=p_posterior)
    np.savez(savedir,chi_eff=chi_eff,chi_p=chi_p,eff_posterior=eff_posterior,p_posterior=p_posterior)
    



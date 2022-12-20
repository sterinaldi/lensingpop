import numpy as np
import matplotlib.pyplot as plt
import cpnest
import cpnest.model
from numpy.lib import recfunctions as rfn
from pathlib import Path
from corner import corner
from figaro.plot import plot_1d_dist
import dill

mmin = 15.15
mmax = 225.4

sqrt2pi = np.sqrt(2*np.pi)

def pl_mass(x, a, xmin, xmax):
    return x**-a * (1-a)/(xmax**(1-a) - xmin**(1-a))
    
def pl_q(x, a):
    return x**-a * (1-a)

def gaussian(x, mu, sigma):
    return np.exp(-(x-mu)**2/(2*sigma**2))/(sqrt2pi*sigma)

def par_model(s, a, b, mu, sigma):
    return pl_mass(s[:,0], a, mmin, mmax)*pl_q(s[:,1], b)*gaussian(s[:,2], mu, sigma)

class par_dist:
    def __init__(self, pars):
        self.pars = pars
        
    def pdf(self, x):
        return par_model(np.atleast_2d(x), *self.pars)

class parametric_inference(cpnest.model.Model):
    def __init__(self, samples, selection_function):
        self.samples = samples
        self.names   = ['a', 'b', 'mu', 's']
        self.bounds  = [[1,5],[0,5],[-1,1],[0,1]]
        
        
        self.par_bounds   = np.array([[mmin, mmax], [0.,1.], [-1.,1.]])
        self.norm_samples = np.random.uniform(low = self.par_bounds[:,0], high = self.par_bounds[:,1], size = (10000, len(self.par_bounds)))
        self.log_volume   = np.log(np.prod(np.diff(self.par_bounds)))
    
        self.selfunc_samples = np.log(selection_function((samples[:,0], samples[:,1]*samples[:,0])))
        self.selfunc_norm    = selection_function((self.norm_samples[:,0], self.norm_samples[:,1]*self.norm_samples[:,0]))

    def log_prior(self, x):
        logP = super(parametric_inference, self).log_prior(x)
        if np.isfinite(logP):
            if x['b'] == 1 or x['b'] == -1:
                return -np.inf
            return 0.
        return logP
    
    def log_likelihood(self, x):
        log_norm = np.log(np.mean(par_model(self.norm_samples, *x.values)*self.selfunc_norm)) + self.log_volume
        return np.sum(np.log(par_model(samples, *x.values)) + self.selfunc_samples - log_norm)

if __name__ == '__main__':
    
    file = Path('./catalog/m1m2zxeffxp_posterior_PPD_afterSelection_unlensed3303.npz')
    out_folder   = Path('par_inf')
    selfunc_file = Path('./production/selfunc_m1m2_detector.pkl')
    with open(selfunc_file, 'rb') as f:
        selfunc = dill.load(f)

    data = np.load(file)
    samples = np.array([data['m1'][:1500]*(1+data['redshift'][:1500]), data['m2'][:1500]/data['m1'][:1500], data['xeff'][:1500]]).T
    
    c = corner(samples, labels = ['$M_1$','$q$','$\\chi_{eff}$'])
    c.savefig(Path(out_folder, 'samples.pdf'), bbox_inches = 'tight')
    
    postprocessing = True
    
    if not postprocessing:
        M = parametric_inference(samples, selfunc)
        work = cpnest.CPNest(M,
                             verbose   = 2,
                             nlive     = 1000,
                             maxmcmc   = 5000,
                             nensemble = 1,
                             output    = out_folder
                             )
        work.run()
        post = rfn.structured_to_unstructured(work.posterior_samples.ravel())[:,:4]
        np.savetxt(Path(out_folder, 'PE_samples.txt'), post, header = 'a b mu sigma')
        
        draws = [par_dist(p) for p in post]
        with open(Path(out_folder, 'parametric_draws.pkl'), 'wb') as f:
            dill.dump(draws, f)
    
    else:
        post = np.genfromtxt(Path(out_folder, 'PE_samples.txt'))
    
    # PLOTS
    c = corner(post, labels = ['$\\alpha$', '$\\beta$', '$\\mu$', '$\\sigma$'])
    c.savefig(Path(out_folder, 'joint_posteriors.pdf'), bbox_inches = 'tight')
    
    M  = np.linspace(samples[:,0].min(), samples[:,0].max(), 1000)
    mf = [pl_mass(M, ai, mmin, mmax) for ai in post[:,0]]
    plot_1d_dist(M, mf, samples = samples[:,0], out_folder = out_folder, name = 'm1_wrong_dist', label = 'M_1', unit = 'M_{\\odot}', median_label = '\\mathrm{PL}')

    q  = np.linspace(samples[:,1].min(), samples[:,1].max(), 1000)
    qf = [pl_q(q, bi) for bi in post[:,1]]
    plot_1d_dist(q, qf, samples = samples[:,1], out_folder = out_folder, name = 'q_wrong_dist', label = 'q', median_label = '\\mathrm{PL}')

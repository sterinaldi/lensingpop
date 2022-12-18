import numpy as np
import matplotlib.pyplot as plt
import cpnest
import cpnest.model
from numpy.lib import recfunctions as rfn
from pathlib import Path
import dill

class par_dist:
    def __init__(self, pars):
        self.pars = pars
        
    def pdf(self, x):
        return par_model(x, *self.pars)

class parametric_inference(cpnest.model.Model):
    def __init__(self, samples):
        self.samples = samples
        self.names   = [''] # FIXME: Update names with Damon
        self.bounds  = [[]] # FIXME: Update bounds with Damon

    def log_prior(self, x):
        logP = super(parametric_inference, self).log_prior(x)
        if np.isfinite(logP):
            return 0.
        return logP
    
    def log_likelihood(self, x):
        return np.sum(np.log(par_model(samples, *x.values)))
    

if __name__ == '__main__':
    
    file = Path('file/with/unfiltered/catalog')
    out_folder = Path('par_inf')
    samples = np.genfromtxt(file)
    
    M = parametric_inference(samples)
    work = cpnest.CPNest(M,
                         verbose   = 2,
                         nlive     = 1000,
                         maxmcmc   = 5000,
                         nensemble = 1,
                         output    = out_folder
                         )
    work.run()
    post = rfn.structured_to_unstructured(work.posterior_samples.ravel())
    np.savetxt(Path(out_folder, 'PE_samples.txt'), post)
    
    draws = [par_dist(p) for p in post]
    with open(Path(out_folder, 'parametric_draws.pkl'), 'wb') as f:
        dill.dump(draws, f)

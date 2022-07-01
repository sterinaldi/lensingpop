import numpy as np
from figaro.montecarlo import MC_integral
from figaro.exceptions import FIGAROException
import dill 
import math

class OddsRatio():

    def __init__(self, selection=False, error=True, N_image=2,Nevent=1000, Lensing_rate=0.001, Nmc=int(1e5)):
        self.Nmc = Nmc 
        self.selection = selection
        self.error = error
        self.Nevent = Nevent
        self.R_L = Lensing_rate
        self.N_L = Nevent = Nevent*Lensing_rate
        self.N_U = Nevent - N_L 
        self.N_i = N_i
        #self.population = self.InitPopulationPrior()
    
    def __call__(self, event1, event2, population, PEuniform=True):
        if PEuniform:
            #return self.OddsPrior() * self.BayesFactor_PEuniform(event1, event2, population) 
            if self.error:
                blu = self.BayesFactor_PEuniform(event1, event2, population)
                return (self.OddsPrior()*blu[0],blu[1]) 
            else:
                return self.OddsPrior()*self.BayesFactor_PEuniform(event1, event2, population) 
    #maybe we can reconstruct pop prior directly  
    def InitPopulationPrior(self):
        # List is expected for computing the uncertainty from population inference
        #if self.selection:
        # Reconstruct population prior with selection function 
        return 1.0 

    def OddsPrior(self):
        return math.factorial(self.N_i) * self.N_L / self.Nevent**self.N_i
 
    def BayesFactor_PEuniform(self, event1, event2, population):
            
        #population = self.population
        Poverlap = self.MC_integral_3terms(event1,event2,population, n_draws=self.Nmc,error=self.error)
        Pastro_event1 = MC_integral(population, event1, n_draws=self.Nmc,error=self.error)
        Pastro_event2 = MC_integral(population, event2, n_draws=self.Nmc,error=self.error)
        if self.error:
            #Compute propagation error of BLU =   term1 / (term2 * term3)
            blu = Poverlap[0] / Pastro_event1[0] / Pastro_event2[0]
            prop_error = blu * np.sqrt((Poverlap[1]/Poverlap[0])**2 + (Pastro_event1[1]/Pastro_event1[0])**2 + (Pastro_event2[1]/Pastro_event2[0])**2)
            return (blu, prop_error)
        else:
            return Poverlap / Pastro_event1 / Pastro_event2 
        
    def MC_integral_3terms(self,p, q, r, n_draws = 1e4, error = True):
        """
        Monte Carlo integration using FIGARO reconstructions.
            ∫p(x)q(x)r(x)dx ~ ∑q(x_i)r(x_i)/N with x_i ~ p(x)
    
        p(x) must have a rvs() method and q(x) and r(x) must have a rvs() method.
        Lists of r is also accepted.
    
        Arguments:
            :class instance p: the probability density to evaluate. Must have a rvs() method.
            :class instance q: the probability density to sample from. Must have a pdf() method.
            :list or class instance r: the probability density to sample from. Must have a pdf() method.

            :int n_draws:              number of MC draws
            :bool error:               whether to return the uncertainty on the integral value or not.
    
        Return:
            :double: integral value
            :double: uncertainty (if error = True)
           """
           # Number of q draws and methods check
        if np.iterable(r):
            if not np.alltrue([hasattr(ri, 'rvs') for ri in r]):
                raise FIGAROException("r must have rvs method")
            n_r = len(r)
            np.random.shuffle(r)
            iter_r = True
        else:
            if not hasattr(r, 'rvs'):
                raise FIGAROException("r must have rvs method")
            iter_r = False

        n_draws = int(n_draws)
        if iter_r:
            samples = p.rvs(n_draws)
            probabilities = np.array([ri.pdf(samples)*q.pdf(samples) for ri in r])
        else:
            samples = p.rvs(n_draws)
            probabilities = np.atleast_2d(r.pdf(samples)*q.pdf(samples))
    
        means = probabilities.mean(axis = 1)
        I = means.mean()
        if not error:
            return I
        mc_error = (probabilities.var(axis = 1)/n_draws).mean()
        figaro_error = means.var()/len(means)
    
        return I, np.sqrt(mc_error + figaro_error)


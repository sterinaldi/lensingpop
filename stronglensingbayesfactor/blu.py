import numpy as np
from figaro.exceptions import FIGAROException
import dill
from figaro.marginal import marginalise
from figaro.credible_regions import ConfidenceArea
from matplotlib import rcParams
from matplotlib import axes



"""
Here is the example of using this class.

# Importing packages:

import numpy as np
from blu import *
import dill 
from simulated_universe import *



# Inferred HDPGMM model from observations
with open('/Users/damon/Desktop/blu_upload/git_download/hier/test_output/posteriors_hier.pkl', 'rb') as f:
    pop_model=dill.load(f)
    
# Importing the stored theoretical pdf and make it usable (has a pdf function) for computing BLU.
class true_dist():
    def __init__(self):
        with open('./real_dist.pkl', 'rb') as f: self.pop_pdf = dill.load(f)
        
    def __call__(self, x):
        return self.pdf(x)
        
    def pdf(self, x):
        return self.pop_pdf(x)
        
# Wrong population model (power law)
with open('./pop_power.pkl', 'rb') as f:
    pop_power=dill.load(f)
    
# Uniform population model
with open('./pop_uniform.pkl', 'rb') as f:
    pop_uni=dill.load(f)

# Import the posterior of lensing pair (DPGMM model) 
with open('./img1_det.pkl', 'rb') as f:
    img1=dill.load(f)
with open('./img2_det.pkl', 'rb') as f:
    img2=dill.load(f)


pop_true=true_dist() #Initialize the theoretical pdf 


N = 1e4 # Number of Monte Carlo samples
OLU_true = OddsRatio(gw_pop=pop_true,Nmc=N) # BLU class by using the theoretical pdf 


blu_true = []

for i in range(len(img1)):
    print(i,'-th pair')
    blu_true.append(OLU_true.BayesFactor_PEuniform(img1[i],img2[i]))

def log_blu(data,error=True):
    
    if error:
        # for y = log(x)
        #    dy = dx/x 
        blu, error = data
        error /= blu 
        blu = np.log(blu)
        return (blu,error)
    else:
        return np.log(data)
    

true_ans = np.array(true_ans)


"""



class OddsRatio():

    def __init__(self, gw_pop=None, error=True, N_image=2,Nevent=1000, Lensing_rate=0.001, Nmc=int(1e5)):
        
        self.Nmc = Nmc 
        self.error = error  # compute error or not
        self.Nevent = Nevent # total number of events we observed 
        self.R_L = Lensing_rate # 
        self.N_L = Nevent = Nevent*Lensing_rate
        self.N_U = Nevent - self.N_L 
        self.N_i = N_image
        self.population = gw_pop
    
    def __call__(self, event1, event2, PEuniform=True):
        if PEuniform:
            if self.error:
                plu = self.OddsPrior()
                blu = self.BayesFactor_PEuniform(event1, event2)
                return (plu*blu[0],plu*blu[1]) 
            else:
                return self.OddsPrior()*self.BayesFactor_PEuniform(event1, event2) 
    
    def OddsPrior(self):
        return np.math.factorial(self.N_i) * self.N_L / self.Nevent**self.N_i
            
        
    def BayesFactor_PEuniform(self, event1, event2):
        """
        Bayes factor (BLU) based on uniform prior used in parameter estimation
        
        BLU = A / (B*C)
        A = ∫p(x|d1)p(x|d2)p_pop(x)dx
        B = ∫p(x|d1)p_pop(x)dx
        C = ∫p(x|d2)p_pop(x)dx
        
        p(x) must have a pdf() method and q(x) must have a rvs() method.
        Lists of p and q are also accepted.

        Arguments:
            :list or class instance p: the probability density to evaluate. Must have a pdf() method.
            :list or class instance q: the probability density to sample from. Must have a rvs() method.
            :int n_draws:              number of MC draws
            :bool error:               whether to return the uncertainty on the integral value or not.
        Return:
            :double: integral value
            :double: uncertainty (if error = True)
        """
        population = self.population

        
        A = self.Overlap_integral(event1,event2,population, n_draws=self.Nmc,error=self.error)
        B = self.MC_integral(event1, population, n_draws=self.Nmc,error=self.error)
        C = self.MC_integral(event2, population, n_draws=self.Nmc,error=self.error)
        #print(A,B,C)
        
        
        if self.error:
            dA, dB, dC = A[1], B[1], C[1]
            A, B, C = A[0], B[0], C[0]
            error = dA / (B*C) - dB * A / (C * B**2) - dC * A / (C * B**2)
            blu = A / (B*C)
            return (0,0) if np.isnan(blu) else (blu,error)
        else:
            blu = A / (B*C)
            return 0 if np.isnan(blu) else blu



    def MC_integral(self, p, q, n_draws = 1e4, error = True):
        """
        Monte Carlo integration using FIGARO reconstructions.
            ∫p(x)q(x)dx ~ ∑p(x_i)/N with x_i ~ q(x)

        p(x) must have a pdf() method and q(x) must have a rvs() method.
        Lists of p and q are also accepted.

        Arguments:
            :list or class instance p: the probability density to evaluate. Must have a pdf() method.
            :list or class instance q: the probability density to sample from. Must have a rvs() method.
            :int n_draws:              number of MC draws
            :bool error:               whether to return the uncertainty on the integral value or not.

        Return:
            :double: integral value
            :double: uncertainty (if error = True)
        """
        # Check that both p and q are iterables or callables:
        if not ((hasattr(p, 'pdf') or np.iterable(p)) and (hasattr(q, 'rvs') or np.iterable(q))):
            raise FIGAROException("p and q must be list of callables or having pdf/rvs methods")
        # Number of p draws and methods check
        if np.iterable(p):
            if not np.alltrue([hasattr(pi, 'pdf') for pi in p]):
                raise FIGAROException("p must have pdf method")
            n_p = len(p)
            np.random.shuffle(p)
            iter_p = True
        else:
            if not hasattr(p, 'pdf'):
                raise FIGAROException("p must have pdf method")
            iter_p = False
        # Number of q draws and methods check
        if np.iterable(q):
            if not np.alltrue([hasattr(qi, 'rvs') for qi in q]):
                raise FIGAROException("q must have rvs method")
            n_q = len(q)
            np.random.shuffle(q)
            iter_q = True
        else:
            if not hasattr(q, 'rvs'):
                raise FIGAROException("q must have rvs method")
            iter_q = False

        n_draws = int(n_draws)

        # Integrals
        if iter_q:
            samples = p.rvs(n_draws)
            probabilities = np.array([qi.pdf(samples) for qi in q])
        else:
            probabilities = np.atleast_2d(q.pdf(p.rvs(n_draws)))

        means = probabilities.mean(axis = 1)
        I = means.mean()
        if not error:
            return I
        mc_error = (probabilities.var(axis = 1)/n_draws).mean()
        figaro_error = means.var()/len(means)

        return I, np.sqrt(mc_error + figaro_error)


    def Overlap_integral(self,p, q, r, n_draws = 1e4, error = True):
        """
        Monte Carlo integration using FIGARO reconstructions.
            ∫p1(x)p2(x)p_pop(x)dx ~ ∑p2(x_i)p_pop(x_i)/N with x_i ~ p1(x)
            
        p1(x) must have a rvs() method and p2(x) and p_pop(x) must have a rvs() method.
        Lists of r is also accepted.
    
        Arguments:
            :class instance p1: the probability density to evaluate. Must have a rvs() method.
            :class instance p2: the probability density to sample from. Must have a pdf() method.
            :list or class instance p_pop: the probability density to sample from. Must have a pdf() method.

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

def recursive_grid(bounds, n_pts):
    """
    Recursively generates the n-dimensional grid points (extremes are excluded).
    
    Arguments:
        :list-of-lists bounds: extremes for each dimension (excluded)
        :int n_pts:            number of points for each dimension
        
    Returns:
        :np.ndarray: grid
    """
    bounds = np.atleast_2d(bounds)
    n_pts  = np.atleast_1d(n_pts)
    if len(bounds) == 1:
        d  = np.linspace(bounds[0,0], bounds[0,1], n_pts[0]+2)[1:-1]
        dD = d[1]-d[0]
        return np.atleast_2d(d).T, [dD]
    else:
        grid_nm1, diff = recursive_grid(np.array(bounds)[1:], n_pts[1:])
        
        d = np.linspace(bounds[0,0], bounds[0,1], n_pts[0]+2)[1:-1]
        diff.append(d[1]-d[0])
        grid     = []
        for di in d:
            for gi in grid_nm1:
                grid.append([di,*gi])
        return np.array(grid), diff


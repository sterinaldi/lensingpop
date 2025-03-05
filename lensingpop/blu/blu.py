#!/home/hoitim.cheung/.conda/envs/figaro2/bin/python
import numpy as np
from figaro.exceptions import FIGAROException
import dill
import matplotlib.pyplot as plt

def calculate_blu_and_error(data):
    """
    Calculate blu and error for the given dataset.

    Parameters:
        data (ndarray): Input data array with columns:
                        A, B, C, dA, dB, dC

    Returns:
        ndarray: Array of [blu, error] values.
    """
    A, B, C = data[:, 0], data[:, 1], data[:, 2]
    dA, dB, dC = data[:, 3], data[:, 4], data[:, 5]

    blu = A / B / C
    err = np.sqrt((dA / (B * C))**2 + 
                  (dB * A / (C * B**2))**2 + 
                  (dC * A / (B * C**2))**2)

    return np.array([blu, err]).T

class OddsRatio:
    """
    Compute the odds ratio for gravitational wave lensing events based on Bayesian statistics.

    Attributes:
        gw_pop (object): Population model for gravitational wave events.
        error (bool): Whether to compute and propagate uncertainties.
        N_image (int): Number of observed images per event.
        Nevent (int): Total number of observed events.
        Lensing_rate (float): Fraction of events that are lensed.
        Nmc (int): Number of Monte Carlo samples.
    """

    def __init__(self, gw_pop=None, error=True, N_image=2, Nevent=1000, Lensing_rate=0.001, Nmc=int(1e5)):
        self.Nmc = Nmc  # Number of Monte Carlo draws
        self.error = error  # Compute errors or not
        self.Nevent = Nevent  # Total observed events
        self.R_L = Lensing_rate  # Lensing rate
        self.N_L = Nevent * Lensing_rate  # Number of lensed events
        self.N_U = Nevent - self.N_L  # Number of unlensed events
        self.N_i = N_image  # Number of observed images
        self.population = gw_pop  # Population model for GW events

    def __call__(self, event1, event2, PEuniform=True):
        """
        Compute the odds ratio between two events.

        Parameters:
            event1 (object): The first event data.
            event2 (object): The second event data.
            PEuniform (bool): Whether the prior in parameter estimation is uniform.

        Returns:
            float or tuple: The odds ratio (and uncertainty if `error=True`).
        """
        if PEuniform:
            if self.error:
                plu = self.OddsPrior()
                blu = self.BayesFactor_PEuniform(event1, event2)
                return (plu * blu[0], plu * blu[1])
            else:
                return self.OddsPrior() * self.BayesFactor_PEuniform(event1, event2)

    def OddsPrior(self):
        """
        Compute the prior odds for lensed events based on observed data.

        Returns:
            float: Prior odds value.
        """
        return np.math.factorial(self.N_i) * self.N_L / self.Nevent**self.N_i

    def BayesFactor_PEuniform(self, event1, event2):
        """
        Compute the Bayes factor (BLU) under uniform parameter estimation priors.

        BLU is calculated as A / (B * C), where:
            A =  ∫p(x|d1)p(x|d2)p_pop(x)dx
            Overlap integral of the two events and population.
            
            B, C = ∫p(x|d1)p_pop(x)dx, ∫p(x|d2)p_pop(x)dx
            Monte Carlo integrals of each event with the population.

        Parameters:
            event1, event2 (object): Event data objects.

        Returns:
            float or tuple: BLU value (and uncertainty if `error=True`).
        """
        population = self.population
        A = self.Overlap_integral(event1, event2, population, n_draws=self.Nmc, error=self.error)
        B = self.MC_integral(event1, population, n_draws=self.Nmc, error=self.error)
        C = self.MC_integral(event2, population, n_draws=self.Nmc, error=self.error)

        if self.error:
            dA, dB, dC = A[1], B[1], C[1]
            A, B, C = A[0], B[0], C[0]
            blu = A / (B * C)
            error = np.sqrt((dA / (B * C))**2 + (dB * A / (C * B**2))**2 + (dC * A / (B * C**2))**2)
            return (0, 0) if np.isnan(blu) else (blu, error)
        else:
            blu = A / (B * C)
            return 0 if np.isnan(blu) else blu
               
    def _BayesFactor_PEuniform(self, event1, event2):
        """
        Compute the Bayes factor (BLU) under uniform parameter estimation priors.

        BLU is calculated as A / (B * C), where:
            A =  ∫p(x|d1)p(x|d2)p_pop(x)dx
            Overlap integral of the two events and population.
            
            B, C = ∫p(x|d1)p_pop(x)dx, ∫p(x|d2)p_pop(x)dx
            Monte Carlo integrals of each event with the population.

        Parameters:
            event1, event2 (object): Event data objects.

        Returns:
            float or tuple: BLU value (and uncertainty if `error=True`).
        """
        population = self.population
        A = self.Overlap_integral(event1, event2, population, n_draws=self.Nmc, error=self.error)
        B = self.MC_integral(event1, population, n_draws=self.Nmc, error=self.error)
        C = self.MC_integral(event2, population, n_draws=self.Nmc, error=self.error)

        if self.error:
            dA, dB, dC = A[1], B[1], C[1]
            A, B, C = A[0], B[0], C[0]
            blu = A / (B * C)
            error = np.sqrt((dA / (B * C))**2 + (dB * A / (C * B**2))**2 + (dC * A / (B * C**2))**2)
            return (A, B, C, dA, dB, dC)
        else:
            blu = A / (B * C)
            return 0 if np.isnan(blu) else blu
        
    def MC_integral(self, p, q, n_draws=1e4, error=True):
        """
        Compute a Monte Carlo integral of p(x)q(x)dx.

        Parameters:
            p (object): Probability density to evaluate, must have a `pdf` method.
            q (object): Probability density to sample from, must have a `rvs` method.
            n_draws (int): Number of Monte Carlo samples.
            error (bool): Whether to compute uncertainties.

        Returns:
            float or tuple: Integral value (and uncertainty if `error=True`).
        """
        if np.iterable(p):
            n_p = len(p)
            np.random.shuffle(p)
            iter_p = True
        else:
            iter_p = False
        # Number of q draws and methods check
        if np.iterable(q):
            n_q = len(q)
            np.random.shuffle(q)
            iter_q = True
        else:
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

    def Overlap_integral(self, p, q, r, n_draws=1e4, error=True):
        """
        Compute an overlap integral ∫p1(x)p2(x)p_pop(x)dx.

        Parameters:
            p, q, r (object): Probability densities with `pdf` and/or `rvs` methods.
            n_draws (int): Number of Monte Carlo samples.
            error (bool): Whether to compute uncertainties.

        Returns:
            float or tuple: Integral value (and uncertainty if `error=True`).
        """
        if np.iterable(r):
            
            n_r = len(r)
            np.random.shuffle(r)
            iter_r = True
        else:
            
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


    def BayesFactor_Simple(self, event1, event2, error = True):
        """
        Bayes factor (BLU) based on uniform prior used in parameter estimation
        
        BLU = ∫p(x|d1)p(x|d2)/p_pop(x)dx
        
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
        return self.Overlap_intergral_inversePop(event1, event2, self.population, n_draws=self.Nmc,error=self.error)
        

    def Overlap_intergral_inversePop(self, p, q, r, n_draws = 1e4, error = True):
        """
        Monte Carlo integration using FIGARO reconstructions.
            ∫p1(x)p2(x)/p_pop(x)dx ~ ∑p2(x_i)/p_pop(x_i)/N with x_i ~ p1(x)
            
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
            n_r = len(r)
            np.random.shuffle(r)
            iter_r = True
        else:
            iter_r = False

        n_draws = int(n_draws)
        if iter_r:
            samples = p.rvs(n_draws)
            p1 = q.pdf(samples).flatten()
            p2 = np.array([ri.pdf(samples) for ri in r])
        else:
            samples = p.rvs(n_draws)
            p1 = np.atleast_2d(q.pdf(samples)).flatten()
            p2 = np.atleast_2d(r.pdf(samples))
            
        if p2[p2==0].size != 0:
            probabilities = np.array([(p1[p2i!=0]/p2i[p2i!=0]) for p2i in p2])
        else:
            probabilities = p1/p2
        
        means = probabilities.mean(axis = 1)
    
        I = means.mean()
        if not error:
            return I
        mc_error = (probabilities.var(axis = 1)/n_draws).mean()
        figaro_error = means.var()/len(means)

        return I, np.sqrt(mc_error + figaro_error)
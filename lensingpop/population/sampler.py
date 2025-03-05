import numpy as np
from . import simulated_universe as universe
from figaro.utils import rejection_sampler 
from tqdm import tqdm 

def RedshiftSampler(nSample):
    """
    Sample the redshift distribution using a rejection sampling method.
    
    Parameters:
    - nSample (int): Number of redshift samples to generate.
 
    Returns:
    - sample (ndarray): Array of sampled redshift values.
    """
    sample = rejection_sampler(nSample, universe.redshift_distribution, [universe.z_min, universe.z_max])
    return sample

def MagnificationSampler(nSample):
    """
    Sample magnification values using rejection sampling.
    
    Parameters:
    - nSample (int): Number of magnification samples to generate.
    
    Returns:
    - sample1 (ndarray): Array of sampled primary magnification values.
    - sample2 (ndarray): Array of sampled secondary magnification values.
    """
    sample1 = rejection_sampler(nSample, universe.magnification_distribution, [universe.mag_min, universe.mag_max])
    sample2 = []
    
    # Sample secondary magnification based on the primary sample
    for mu1 in tqdm(sample1, desc='Sampling magnification2'):
        m2sis = 2-mu1
        sample2.append(rejection_sampler(1, lambda mu2: universe.magnification2_distribution(mu2, mu1), [universe.mag2_min, universe.mag2_max]))
    
    sample2 = np.array(sample2).reshape(sample1.shape)
    return sample1, sample2

def MassSampler(z):
    """
    Sample masses (m1 and m2) at given redshifts using a rejection sampling method.
    
    Parameters:
    - z (ndarray): Array of redshift values.
    
    Returns:
    - m1 (ndarray): Array of primary mass samples.
    - m2 (ndarray): Array of secondary mass samples.
    """
    m1 = []
    m2 = []
    
    # Sample masses for each redshift
    for zi in tqdm(z, desc='Sampling masses'):
        masses = rejection_sampler(2, lambda m: universe.mass_distribution(m, zi), [universe.m_min, universe.m_max])
        m1.append(np.max(masses))
        m2.append(np.min(masses))

    return np.array(m1), np.array(m2)


import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# Sigma values
sigma_mass     = 0.088
sigma_z        = 0.3
sigma_q        = 1.0671
sigma_s        = 0.4

def transf(x, l, h):
    """
    Transformation: y = log((x-l)/(h-x))
    """
    return np.log((x-l)/(h-x))

def antitransf(y, l, h):
    """
    Anti-transformation: x = (e^y * h + l)/(e^y + 1)
    """
    return (np.exp(y) * h + l)/(np.exp(y) + 1)

def sampler_log(mu, sigma, n_draws):
    """
    Draws samples from a log-Gaussian distribution
    """
    return np.exp(np.random.normal(np.log(mu), sigma, n_draws))

def sampler_reg(mu, sigma, l, h, n_draws):
    """
    Draws samples from a regolarised normal distribution (for bounded variables): y = log((x-l)/(h-x))
    """
    y  = transf(mu, l, h)
    ss = np.random.normal(y, sigma, n_draws)
    return antitransf(ss, l, h)

def sample_event(m1, q, z, xeff, n_draws):
    """
    Sample events
    """
    m1_cv   = sampler_log(m1, sigma_mass, 1)
    q_cv    = sampler_reg(q, sigma_q, 0, 1, 1)
    z_cv    = sampler_log(z, sigma_z, 1)
    xeff_cv = sampler_reg(xeff, sigma_s, -1, 1, 1)
    
    m1_samples   = sampler_log(m1_cv, sigma_mass, n_draws)
    q_samples    = sampler_reg(q_cv, sigma_q, 0, 1, n_draws)
    z_samples    = sampler_log(z_cv, sigma_z, n_draws)
    xeff_samples = sampler_reg(xeff_cv, sigma_s, -1, 1, n_draws)
    
    return m1_samples, q_samples, z_samples, xeff_samples

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = "Generate population and posterior samples.")
    parser.add_argument("-i", dest = "file", type = str, help = "File with true values")
    parser.add_argument("-N", "--Npos", dest = "Npos", type = int, help = "number of posterior samples per event", default = 1000)
    parser.add_argument("-L", dest = "L", action = 'store_true', help = "Generate lensed population", default = False)
    parser.add_argument("-s", dest = "seed", type = int, help = "Seed", default = 1234)

    args = parser.parse_args()
    np.random.seed(args.seed)
    file = Path(args.file)
    
    # Read file
    evs = np.load(file)
    n_evs = len(evs['m1'])
    
    m1   = evs['m1']
    q    = evs['m2']/evs['m1']
    z    = evs['redshift']
    xeff = evs['xeff']
    
    # Initialise posteriors
    m1_posteriors   = np.zeros((n_evs, args.Npos))
    q_posteriors    = np.zeros((n_evs, args.Npos))
    z_posteriors    = np.zeros((n_evs, args.Npos))
    xeff_posteriors = np.zeros((n_evs, args.Npos))
    
    for i, (m1_i, q_i, z_i, xeff_i) in tqdm(enumerate(zip(m1, q, z, xeff)), total = n_evs, desc = 'Sampling'):
        m1_posteriors[i], q_posteriors[i], z_posteriors[i], xeff_posteriors[i] = sample_event(m1_i*(1+z_i), q_i, z_i, xeff_i, args.Npos)
    
    if args.L:
        output_file = Path('./m1m2zxeff_posterior_PPD_afterSelection_lensed'+str(m1.size)+'.npz')
    else:
        output_file = Path('./m1m2zxeff_posterior_PPD_afterSelection_unlensed'+str(m1.size)+'.npz')
        
    np.savez(output_file,
             m1             = m1,
             m2             = m1*q,
             redshift       = z,
             xeff           = xeff,
             m1_posterior   = m1_posteriors,
             m2_posterior   = m1_posteriors*q_posteriors,
             z_posterior    = z_posteriors,
             xeff_posterior = xeff_posteriors,
             )

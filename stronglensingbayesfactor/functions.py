import numpy as np
import scipy.stats as ss 
import astropy.units as u
import scipy.integrate as integrate
import pickle
from astropy.cosmology import FlatLambdaCDM, z_at_value
from scipy.interpolate import interp1d
from scipy.stats import betaprime, uniform, randint, truncnorm
from scipy.special import erf, erfinv
from scipy.interpolate import interp1d
from gwcosmo import priors as p
cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.31)
### Now we demonstrate with kde first

def DensityEstimator(data):
    
    values = np.vstack(data.T)
    kernel = ss.gaussian_kde(values)
    
    return kernel

def LuminosityDistance(redshift):
    
    dL = cosmo.luminosity_distance(redshift).value
    
    return dL


def TruncNormSampler(clip_a, clip_b, mean, std, Nsamples):

    a, b = (clip_a - mean) / std, (clip_b - mean) / std

    return truncnorm.rvs(a,b,size=Nsamples ) * std + mean

def TruncNormPdf(y, clip_a, clip_b, mean, std):
    x = (y - mean ) / std
    a, b = (clip_a - mean) / std, (clip_b - mean) / std
    return truncnorm.pdf(x,a,b) 
############### Rejection samplig ###########################

def inverse_transform_sampling(bins, pdf, nSamples=1):
    cumValue = np.zeros(bins.shape)
    cumValue[1:] = np.cumsum(pdf[1:] * np.diff(bins))
    """
    if cumValue.max() ==0:
        print(cumValue)
        print(pdf)
    """
    cumValue /= cumValue.max()
    inv_cdf = interp1d(cumValue, bins)
    r = np.random.rand(nSamples)


    sample = inv_cdf(r)
    return inv_cdf(r)



def powerlaw_pdf(mu, alpha, mu_min, mu_max):
    
    if -alpha != -1:
        norm_mass = (mu_max **(1+alpha) - mu_min**(1+alpha))  /  (1+alpha)
    else:
        norm_mass = np.log(mu_max ) - np.log(mu_min)
    
    p_mu = mu**alpha / norm_mass
    
    p_mu[(mu<mu_min) + (mu>mu_max)] = 0

    return p_mu

############# Our density/probability estimator #############
def LogNormIshPPF(u, a = 1.1375, b = 0.8665, zmax = 15):
    
    """For sampling the analytical approximation of the redshift distribution"""
    
    ppf = np.exp(a**2 + b - a*2**0.5*erfinv(1 - u*(1 - erf((a**2 + b - 
                                                np.log(zmax))/2**0.5/a))))
    
    return ppf


def BetaprimePPF(u, a = 2.906, b = 0.0158, c = 0.58, zmax = 15):
    
    """For sampling the analytical approximation of the redshift distribution"""
    
    ppf = betaprime.ppf(u*betaprime.cdf(zmax, a, b, loc = c), a, b, loc = c)
    
    return ppf

def RedshiftSampler(lambda_z = 0.563, a1 = 2.906, b1 = 0.0158, c = 0.58, 
                    a2 = 1.1375, b2 = 0.8665, zmax = 15, Nsample=1):
    
    """
    Function for sampling the redshift distribution using a 
    rejection sampling procedure.
    """


    
    # Random number between 0 and 1 that will define which
    # distribution will be drawn from
    u = uniform.rvs(size=Nsample)
    
    sample = np.zeros(u.shape)
    size1 = u[u >= lambda_z].size
    size2 = u[u < lambda_z].size


    sample[u >= lambda_z] = BetaprimePPF(uniform.rvs(size=size1), a = a1, b = b1, c = c, 
                                        zmax = zmax)


    sample[u < lambda_z] = LogNormIshPPF(uniform.rvs(size=size2), a = a2, b = b2,
                                        zmax = zmax)
    
    
    return sample
def BayesFactor_intrinsic(event1,event2,z,pop_prior,Nsample=10000):
    p1 = DensityEstimator(event1)
    
    p2 = DensityEstimator(event2)

    # Draw samples for Monte Carlo integration
    sample = p2.resample(size=Nsample)
    #print(sample.shape)
    probability_event1 = p1.pdf(sample)

    population_prior = pop_prior.pdf( event1 )
    #print(probability_event1)
    MCsample_mean = np.mean(probability_event1/population_prior)
    print(MCsample_mean)
    prior = hypothesis_prior()

    return prior * MCsample_mean
    #return MCsample_mean




class mass_distribution():
    def __init__(self, **model_pars):
        self.model_pars = model_pars
        self.model_pars_gwcosmo = {'alpha': model_pars['alpha'] ,
                                   'beta': model_pars['beta'],
                                   'delta_m': model_pars['delta_m'],
                                   'mmin': model_pars['mmin'],
                                   'mmax': model_pars['mmax'],
                                   'lambda_peak': model_pars['lam'],
                                   'mu_g': model_pars['mpp'],
                                   'sigma_g': model_pars['sigpp']}
        self.model=p.mass_prior('BBH-powerlaw-gaussian', self.model_pars_gwcosmo)

    def sample(self, Nsamples):
        m01, m02 = self.model.sample(Nsample=Nsamples)
        if np.any(m02>m01):
            raise ValueError("m2>m1 error")
        return np.column_stack((m01, m02))
    
    def prob(self,m1,m2):

        return self.model.joint_prob(m1,m2)

def redshift_pdf(zmin=0, zmax=15.0 ,kappa = 1.0 ,z, norm):
    #p_z = (1.0+z)**(kappa-1.0)*cosmo.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value / norm
    p_z = (1.0+z)**(kappa-1.0)*LuminosityDistance(z) / norm
    return p_z

def pp_pdf(m1,m2,z):
    return mass_dist.prob(m1,m2) * redshift_pdf(z)

def ppd_pdf(m1,m2,z):
    pm1 = TruncNormPdf(m1,mmin, mmax, 50.0, 1.5)
    pm2 = TruncNormPdf(m2 / m1, mmin/m1, 1.0, 25.0 /50.0, 0.05)
    return ((1-lambda_d) * mass_dist.prob(m1,m2) + lambda_d * pm1 * pm2 )* redshift_pdf(z)

def Selection_unlensed(pmass,pz):
    data  =np.load('norm_sample_new1e5.npz')
    pdet = data['pdet']
    pdraw = data['p_draw']
    norm_sample = data['event']
    del data
    prob_m1m2 = pmass.pdf(norm_sample[:,0],norm_sample[:,1]) 
    prob_z = pz.pdf(norm_sample[:,2])
    ## sum p_det * p_pop / p_draw
    return np.mean( pdet* prob_m1m2 * prob_z / pdraw)








import numpy as np
from pycbc import waveform, psd, detector
from scipy.stats import betaprime, uniform, randint
from scipy.special import erf, erfinv
import time
from scipy.interpolate import interp1d
import scipy.interpolate as si
from gwcosmo import priors as p
from scipy.stats import truncnorm
from astropy.cosmology import FlatLambdaCDM, z_at_value
import argparse
import pickle
cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.31)
np.random.seed(7)

parser = argparse.ArgumentParser(description='Generate population and posterior samples.')
parser.add_argument('--N',type=int,help='number of population events',default=10000)
args = parser.parse_args()
N = int(args.N) # nunber of events 

# Here come all the definitions used in this script

def TruncNormSampler(clip_a, clip_b, mean, std, Nsamples):

    a, b = (clip_a - mean) / std, (clip_b - mean) / std
    return truncnorm.rvs(a,b,size=Nsamples ) * std + mean


def AntennaPattern(inclination, rightAscension, declination, polarisation, 
                  GPStime, interferometer = 'L1'):
    
    """
    This is a measure for the detector response, depending on the sky 
    localisation and orbital configuration of the binary and the arrival time 
    of the GW (for the transformation between the source frame and the 
    detector frame).
    """
    scienceMachien = detector.Detector(interferometer)
    Fplus, Fcross = scienceMachien.antenna_pattern(rightAscension, declination, 
                                                   polarisation, GPStime)
    
    Aplus = 0.5*Fplus*(1 + np.cos(inclination)**2)
    Across = Fcross*np.cos(inclination)
    
    A = (Aplus**2 + Across**2)**0.5
    
    return A


def LuminosityDistance(redshift):
    
    dL = cosmo.luminosity_distance(redshift).value
    
    return dL


def InclinationPPF(u):
    
    """For sampling the inclination"""
    ppf = np.arccos(1 - 2*u)
    
    return ppf



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
    

class PowerlawPeak_mass_distribution():
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
    def prob(self, m1, m2):
        return self.model.joint_prob(m1,m2)



# Start the sampling shenanigans
start = time.time()

##################################Draw the binary masses#################################################################

################## parameters for Power-law plus peak model ####################
################## parameters used are the inferred result in https://arxiv.org/pdf/2010.14533.pdf #################
################## alpha is set to be larger from 2.63 to 3.63 (steeper slope) #####################################
mmin = 4.59 
mmax = 86.22
PP_pars = {'alpha': 3.63, 
  'beta': 1.26,
  'delta_m': 4.82,
  'mmin': mmin,
  'mmax': mmax,
  'lam': 0.08,
  'mpp': 33.07,
  'sigpp': 5.69}

lambda_d = 0.10 # fraction for delta peak in the distribution
#######################################################################################################



m1  = np.zeros(N)
m2  = np.zeros(N)

################## delta peark for masses (Gaussian with small sigma) where m1 m2, center at 55.0, 25.0 #############################
N_delta  = int(lambda_d * N )
delta_m1 = TruncNormSampler(mmin, mmax, 50.0, 1.5, N_delta)
q = np.zeros(delta_m1.shape)
for i in range(N_delta):
    q[i] = TruncNormSampler(mmin/delta_m1[i], 1.0, 25.0 /50.0, 0.05, 1)
delta_m2 = q * delta_m1

m1[:N_delta], m2[:N_delta] = delta_m1, delta_m2
##########################################################################################################################################


################## draw mass samples from power-law plus peak ######################################################
pp = PowerlawPeak_mass_distribution(**PP_pars)
m1[N_delta:], m2[N_delta:] = pp.sample(N-N_delta).T

################## rearrange the order #################################################
order = np.random.permutation(N)
m1, m2 = m1[order], m2[order]

################## Draw the redshifts and convert to luminosity distances
redshiftValue = RedshiftSampler(zmax = 10,Nsample=N)
dLValue = LuminosityDistance(redshiftValue)

################## Compute the SNR using gwdet package ###############
################## Default setting https://github.com/dgerosa/gwdet ################################
import gwdet 
pdet  = gwdet.detectability()
snr = pdet.snr(m1,m2,redshiftValue)


################## Compute the associated angles ###########################
rightAscensionValue = uniform.rvs(scale = 2*np.pi,size=N)
declinationValue = np.arcsin(2*uniform.rvs(size=N) - 1)
polarisationValue = uniform.rvs(scale = 2*np.pi,size=N)
inclinationValue = InclinationPPF(uniform.rvs(size=N))

################## Events spread throughout roughly 1 yr (observational run time)
GPStimeValue = randint.rvs(0, 31.6E6,size=N) 
    
################## Calculate the detector SNR using the method from Roulet et al. (2020)
antennaPatternValue = AntennaPattern(inclinationValue, rightAscensionValue,
                                    declinationValue, polarisationValue,
                                    GPStimeValue)




################## Save the data pf intrinsic catalog ########################################################
filename = "PowerlawplusPeakplusDelta{:.0f}Samples.npz".format(N)

np.savez(filename, m1=m1, m2=m2, redshift=redshiftValue, snr=snr, 
        inclinationValue=inclinationValue, polarisationValue=polarisationValue,
        rightAscensionValue=rightAscensionValue, declinationValue=declinationValue,
        GPStimeValue=GPStimeValue)


t1 = time.time()
print('Calculation time: {:.2f} s'.format(t1 - start))


################## applying detectability #########################
################## Default setting https://github.com/dgerosa/gwdet ################################
with open("gwdet_default_interpolator", "rb") as f:
    pdet = pickle.load(f)


pdet_value = pdet(np.array([m1,m2,redshiftValue]).T)
randnum = uniform.rvs(size=N)
index = randnum < pdet_value
print(index)
print("Number of envets in the catalog = ",N, "after selection = ", m1[index].shape)
filename = "PowerlawplusPeakplusDelta{:.0f}Samples_afterSelection.npz".format(N)

################## Save the data after applying selection effect ########################################################
np.savez(filename, m1=m1[index], m2=m2[index], redshift=redshiftValue[index], snr=snr[index], 
        inclinationValue=inclinationValue[index], polarisationValue=polarisationValue[index],
        rightAscensionValue=rightAscensionValue[index], declinationValue=declinationValue[index],
        GPStimeValue=GPStimeValue[index])

print('Calculation time for pdet: {:.2f} s'.format(time.time() - t1))

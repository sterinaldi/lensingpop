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
cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.31)



parser = argparse.ArgumentParser(description='Generate population and posterior samples.')
parser.add_argument('--N',type=int,help='number of population events',default=10000)
parser.add_argument('--Nsample',type=int,help='number of posterior samples per event',default=1000)

args = parser.parse_args()
N = int(args.N) # nunber of events 
Nsample = int(args.Nsample) # nunber of posterior samples per event



# Here come all the definitions used in this script

def SignalToNoiseRatio(hpTilde, fLower = 20, noise = 'LIGO'):
    
    """
    This is the single-detector SNR of an optimally oriented source at a 
    fiducial distance of 1 Mpc.
    
    Different PSDs:
        - aLIGOAPlusDesignSensitivityT1800042 (A+)
        - https://dcc.ligo.org/LIGO-T1500293/public (Voyager)
    """

    if noise == 'LIGO':

        PSD = psd.aLIGOZeroDetHighPower(len(hpTilde), hpTilde.delta_f, fLower)
        # PSD = psd.aLIGOAPlusDesignSensitivityT1800042(len(hpTilde), hpTilde.delta_f, fLower)
        # PSD = psd.from_txt(PATH+"voyager.txt", len(hpTilde), 1/4, fLower)
    elif noise == 'Virgo':
        PSD = psd.AdvVirgo(len(hpTilde), hpTilde.delta_f, fLower)
    elif noise == 'KAGRA':
        PSD = psd.KAGRADesignSensitivityT1600593(len(hpTilde), hpTilde.delta_f, fLower)

    hpTilde.resize(len(PSD))
    
    integrand = []

    # Get frequencies
    f = hpTilde.sample_frequencies
    # Get frequency difference (sampled at a constant frequency interval)
    df = np.diff(f)[0]

    # Compute integrand
    integrand = hpTilde*np.conjugate(hpTilde)/PSD

    # In some versions of the pycbc, PSD gives 0 below f<10 and above some 
    # values, and then the integrand contains an infinity or a nan because 
    # there's a division by zero. Filter ouf these values:
    integrand = integrand.numpy()
    mask = np.isfinite(integrand)

    integral = np.sum(integrand[mask])*df
    
    SNR = (4*np.real(integral))**0.5
   
    return SNR

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
                    a2 = 1.1375, b2 = 0.8665, zmax = 15):
    
    """
    Function for sampling the redshift distribution using a 
    rejection sampling procedure.
    """

    while(True):
        
        # Random number between 0 and 1 that will define which
        # distribution will be drawn from
        u = uniform.rvs()
        
        if u >= lambda_z:
            sample = BetaprimePPF(uniform.rvs(), a = a1, b = b1, c = c, 
                                  zmax = zmax)
            return sample
            break
        
        else:
            sample = LogNormIshPPF(uniform.rvs(), a = a2, b = b2, zmax = zmax)
            return sample
            break


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

# Start the sampling shenanigans

start = time.time()
t1 = time.time()
np.random.seed(0)
# We want one milion samples


#write the results to a file for later reference
file = open('PowerlawplusPeakplusDelta{:.0f}Samples.txt'.format(N), 'w')

file.write('#Results from rejection sampling run with {:.0f} samples\n'.format(N))
file.write('#m1 \t\t m2 \t\t redshift \t SNR \t\t LIGOSNR \t iota \t\t psi \t\t alpha \t\t delta \t GPStime\n')


#Draw the binary masses

## parameters for Power-law plus peak model
PP_pars = {'alpha': 2.63, 
  'beta': 1.26,
  'delta_m': 4.82,
  'mmin': 4.59,
  'mmax': 86.22,
  'lam': 0.10,
  'mpp': 33.07,
  'sigpp': 5.69}

# draw a delta m1, m2 distribution (Gaussian with small sigma) where m1 m2, center at 55.0, 25.0

mmin = 4.59 
mmax = 86.22
# fraction for delta in powerlaw plus peak distribution
lambda_d = 0.15

N_delta  = int(lambda_d * N )
delta_m1 = TruncNormSampler(mmin, mmax, 50.0, 1.5, N_delta)
q = np.zeros(delta_m1.shape)
for i in range(N_delta):
    q[i] = TruncNormSampler(mmin/delta_m1[i], 1.0, 25.0 /50.0, 0.05, 1)


delta_m2 = q * delta_m1

m1  = np.zeros(N)
m2  = np.zeros(N)
m1[:N_delta], m2[:N_delta] = delta_m1, delta_m2

md = mass_distribution(**PP_pars)
m1[N_delta:], m2[N_delta:] = md.sample(N-N_delta).T
m1, m2 = m1[np.random.permutation(N)], m2[np.random.permutation(N)]
for i in range(0, N):
    
    # Draw the redshifts and convert to luminosity distances
    redshiftValue = np.array(RedshiftSampler(zmax = 10))
    dLValue = LuminosityDistance(redshiftValue)
    
    #Compute the SNR

    m1ValueDet = m1[i]*(1 + redshiftValue)
    m2ValueDet = m2[i]*(1 + redshiftValue)
    
    hpTilde, hcTilde = waveform.get_fd_waveform(approximant = 'IMRPhenomD',
                                                mass1 = m1ValueDet, mass2 = m1ValueDet,
                                                delta_f = 1/4, f_lower = 20)
    
    SNRValueLIGO = SignalToNoiseRatio(hpTilde)

    # Draw all the associated angles
    rightAscensionValue = uniform.rvs(scale = 2*np.pi)
    declinationValue = np.arcsin(2*uniform.rvs() - 1)
    polarisationValue = uniform.rvs(scale = 2*np.pi)
    inclinationValue = InclinationPPF(uniform.rvs())

    # Events spread throughout roughly 1 yr (observational run time)
    GPStimeValue = randint.rvs(0, 31.6E6) 


    # Calculate the detector SNR using the method from Roulet et al. (2020)
    antennaPatternValue = AntennaPattern(inclinationValue, rightAscensionValue,
                                        declinationValue, polarisationValue,
                                        GPStimeValue)
    earthSNRValue = SNRValueLIGO*antennaPatternValue/dLValue

    file.write('{:e} \t {:e} \t {:e} \t {:e} \t {:e} \t {:e} \t {:e} \t {:e} \t {:e} \t {:.0f}\n'.format(
                m1[i], m2[i], redshiftValue, earthSNRValue,
                SNRValueLIGO, inclinationValue, polarisationValue, 
                rightAscensionValue, declinationValue, GPStimeValue))

    
    #Check on the progress every 2000 iterations
    if i % 2000 == 0:
        print(i, '/', N,"\t time used =" ,time.time()-t1)
        t1 = time.time()

file.write('#Calculation time: {:.2f} s'.format(time.time() - start))
        
file.close()
###################################### Generate posterior samples #####################################







data = np.loadtxt('PowerlawplusPeakplusDelta{:.0f}Samples.txt'.format(N))
m1, m2, redshift, earthSNRValue = data[:,:4].T
del data 
########################### 

# Can implement some selection function here and select the events.


###########################

SNR_obs = np.zeros(earthSNRValue.shape)
for i in range(earthSNRValue.size):
    SNR_obs[i] = earthSNRValue[i] + TruncNormSampler( -earthSNRValue[i],np.inf, 0.0, 1.0, 1)
# SNR threshold and standard deviations for generating posterior samples  (M Fishbach 2020)

SNR_threshold = 8
sigma_mass = 0.008 * SNR_threshold
sigma_symratio = 0.022 
sigma_theta = 0.21 * SNR_threshold
#dLValue = LuminosityDistance(redshift)




# compute chrip mass and symmetry ratio
Mz = (1+ redshift) * (m1*m2) ** (3./5.) / (m1+m2)** (1./5.)  
sym_mass_ratio = (m1*m2)  / (m1+m2)** 2  

# generate posterior sample for m1 and m2 

m1_posterior = np.zeros((Mz.size,Nsample))
m2_posterior = np.zeros((Mz.size,Nsample))
print('generating posterior')
for i in range(0,Mz.size):
    # chrip mass noise 
    
    Mz_obs = Mz[i] * np.exp( np.random.normal(0, sigma_mass / SNR_obs[i], Nsample) )

    # generate symmetry ratio noise by using truncated normal distribution
    symratio_obs = TruncNormSampler( 0.0, 0.25, sym_mass_ratio[i], sigma_symratio / SNR_obs[i], Nsample)
    
    M = Mz_obs / symratio_obs ** (3./5.)
    m1_obsz = 0.5 * M * (1 + np.sqrt(1 - 4 * sym_mass_ratio[i]) )
    m2_obsz = 0.5 * M * (1 - np.sqrt(1 - 4 * sym_mass_ratio[i]) )
    
    # compute redshifted m1 and m2 
    M = Mz_obs / symratio_obs ** (3./5.)
    m1_obsz = 0.5 * M * (1 + np.sqrt(1 - 4 * sym_mass_ratio[i]) )
    m2_obsz = 0.5 * M * (1 - np.sqrt(1 - 4 * sym_mass_ratio[i]) )

    m1_posterior[i] = m1_obsz / (1 + redshift[i] )
    m2_posterior[i] = m2_obsz / (1 + redshift[i] )
    
# save the posterior 
np.savez('m1m2posterior_PPD_no_selectionbias{:.0f}.npz'.format(N), m1=m1, m2=m2, m1_posterior = m1_posterior, m2_posterior = m2_posterior)








import numpy as np
import argparse
import dill
import time 

from scipy.stats import uniform, randint, norm
from scipy.interpolate import interp1d
import scipy.interpolate as si

from corner import corner

from pathlib import Path
from tqdm import tqdm

from pycbc import waveform, psd, detector
import gwdet
from figaro.utils import rejection_sampler
from figaro.load import _find_redshift
from simulated_universe import *

from calculate_snr_veske import *

def veske_snr(mass1,mass2,z,r):#compute the detection probability with the HLV network at O3 sensitivity with detection threshold POWER SNR=64 (=amplitude SNR=8). Marginalized over the (20^4 x 300) points in the angle and distance grid
    i = int(mass1)
    j = int(mass2) 
    #print(mass1,',',mass2,',',z,r)
    #return  np.sqrt(np.sum((1000**2*np.outer((1+z)**al[int(np.round(100*j/i))-1]/r**2,(o3l[i,j]*anl+o3h[i,j]*anh+o3v[i,j]*anv)))))
    return np.sqrt(numpy.sum((1000**2*numpy.outer((1+z)**al[int(numpy.round(100*j/i))-1]/r**2,(o3l[i,j]*anl+o3h[i,j]*anh+o3v[i,j]*anv)))/(n**4), axis = -1))   
    # Here come all the definitions used in this script
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

def InclinationPPF(u):
    """For sampling the inclination"""
    ppf = np.arccos(1 - 2*u)
    return ppf

def RedshiftSampler(Nsample, lensed):
    """
    Function for sampling the redshift distribution using a 
    rejection sampling procedure.
    """
    if lensed:
        sample = []
        for _ in tqdm(range(Nsample), desc = 'Sampling redshift'):
            sample.append( rejection_sampler(1, lensed_redshift_distribution, [0, z_max]))
        sample = np.array(sample)
    else:
        sample = rejection_sampler(Nsample, redshift_distribution, [0,z_max])
    return sample

def xRedshiftSampler(Nsample, lensed):
    """
    Function for sampling the redshift distribution using a 
    rejection sampling procedure.
    """
    if lensed:
        sample = rejection_sampler(Nsample, lensed_redshift_distribution, [0, z_max])
    else:
        zz=np.linspace(0,z_max,num=Nsample)
        rr=omega.LuminosityDistance(zz) #calculate luminosity distances corresponding to redshifts up to 3.5
        pr= rr ** (2) / (1+zz)**(4)
        r=np.random.choice(rr,size=Nsample,p=pr/np.sum(pr)) #sample the distance points according to r^2/(1+z)^4 distribution
        r=r[np.argsort(r)]
        sample=np.array([zz[np.argsort(abs(ra-rr))[0]] for ra in r]) #find corresponding redshifts

    return sample




def MagnificationSampler(Nsample):
    sample1 = rejection_sampler(Nsample, magnification_distribution, [mag_min,mag_max])
    sample2 = norm(loc = sample1, scale = sample1*rel_sd).rvs()
    return sample1, sample2
"""
def MassSampler(z):
    m1 = []
    m2 = []
    for zi in tqdm(z, desc = 'Sampling masses'):
        masses = rejection_sampler(1, lambda m: mass_distribution(m, zi), [m_min, m_max])
        #m1.append(np.max(masses))
        #m2.append(np.min(masses))
#        masses = rejection_sampler(1, lambda m: mass_distribution(m, zi), [m_min, m_max])
        m1=np.append(m1,masses)
        #m1 = np.array(m1)
        #m2 = np.append(m2,np.random.uniform(m_min / masses, 1 , 1) * masses)
        m2 = np.append(m2, masses*rejection_sampler(1, lambda q: powerlaw_distribution(q),[m_min/masses,1.0]))
#    return m1, m2
    return np.array(m1), np.array(m2)

"""


def MassSampler(z):
    m1 = []
    m2 = []
    for zi in tqdm(z, desc = 'Sampling masses'):
        masses = rejection_sampler(2, lambda m: mass_distribution(m, zi), [m_min, m_max])
        m1.append(np.max(masses))
        m2.append(np.min(masses))

    return np.array(m1), np.array(m2)


np.random.seed(7)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate population and posterior samples.')
    parser.add_argument("-N", type = int, help = "number of population events", default=10000)
    parser.add_argument("-L", dest = "L", action = 'store_true', help = "Generate lensed population", default = False)

    args = parser.parse_args()
    N = args.N # nunber of events
    t0 = time.time()
    interpolator_file = Path("./selection_functions/selfunc_m1m2z_source.pkl").resolve()
    #interpolator_file = Path("./gwdet_default_interpolator.pkl").resolve() 
    catalog_file = Path("./PowerlawplusPeakplusDelta{:.0f}Samples.npz".format(N)).resolve()
    filtered_catalog_file = Path("./Catalog_{:.0f}Samples_afterSelection.npz".format(N)).resolve()

    # Start the sampling shenanigans
    redshiftValue = RedshiftSampler(Nsample=N, lensed = args.L)
    m1, m2 = MassSampler(redshiftValue)
    if args.L:
        magValue1, magValue2 = MagnificationSampler(Nsample=N)
        magValue = np.concatenate([magValue1, magValue2])
    else:
        magValue = np.ones(N)
    dLValue = omega.LuminosityDistance(redshiftValue)
    
    if args.L:
        m1 = np.concatenate([m1,m1])
        m2 = np.concatenate([m2,m2])
        dLValue = np.concatenate([dLValue,dLValue])/np.sqrt(magValue)
        redshiftValue_l = np.array([_find_redshift(omega, l) for l in dLValue])
        redshiftValue = np.concatenate([redshiftValue,redshiftValue])
    else:
        redshiftValue_l = redshiftValue
    ################## Compute the SNR using gwdet package ###############
    ################## Default setting https://github.com/dgerosa/gwdet ################################
    #snr_cal  = gwdet.detectability()

    # don't calculate snr at this stage
    snr = np.random.uniform(0,1,m1.size)
    

    ################## Compute the associated angles ###########################
    rightAscensionValue = uniform.rvs(scale = 2*np.pi,size=len(m1))
    declinationValue = np.arcsin(2*uniform.rvs(size=len(m1)) - 1)
    polarisationValue = uniform.rvs(scale = 2*np.pi,size=len(m1))
    inclinationValue = InclinationPPF(uniform.rvs(size=len(m1)))

    ################## Events spread throughout roughly 1 yr (observational run time)
    GPStimeValue = randint.rvs(0, 31.6E6,size=len(m1))
        
    ################## Calculate the detector SNR using the method from Roulet et al. (2020)
    antennaPatternValue = AntennaPattern(inclinationValue, rightAscensionValue,
                                        declinationValue, polarisationValue,
                                        GPStimeValue)
    ################## Save the data pf intrinsic catalog ########################################################
    np.savez(catalog_file, m1=m1, m2=m2, redshift=redshiftValue, snr=snr,
            inclinationValue=inclinationValue, polarisationValue=polarisationValue,
            rightAscensionValue=rightAscensionValue, declinationValue=declinationValue,
            GPStimeValue=GPStimeValue, magnification = magValue, DL = dLValue)

    ################## applying detectability #########################
    ################## Default setting https://github.com/dgerosa/gwdet ################################
    with open(interpolator_file, "rb") as f:
        pdet = dill.load(f)
    
    pdet_value = pdet(np.array([m1,m2,redshiftValue_l]).T)
    

    randnum = uniform.rvs(0,1,size=len(m1))
    if args.L:
        index = (randnum[:N] < pdet_value[:N]) * (randnum[N:] < pdet_value[N:])
        index = np.concatenate([index, index])
    else:
        index = randnum < pdet_value
    print("Number of events in the catalog: {0} - after selection: {1}".format(N, len(m1[index])))
#    for  m1i,m2i,zi,dli in tqdm(zip(m1[index],m2[index],redshiftValue[index],dLValue[index]), desc = 'SNR'):
#        snr[index] = veske_snr(m1i,m2i,zi,dli)
    # calculate snr for observed events
    snr_cal = gwdet.detectability()
    snr[index]=snr_cal.snr(m1[index],m2[index],redshiftValue[index])


################## Save the data after applying selection effect ########################################################
    snr[index]=snr_cal.snr(m1[index],m2[index],redshiftValue[index])



    np.savez(filtered_catalog_file, m1=m1[index], m2=m2[index], redshift=redshiftValue[index], snr=snr[index],
            inclinationValue=inclinationValue[index], polarisationValue=polarisationValue[index],
            rightAscensionValue=rightAscensionValue[index], declinationValue=declinationValue[index],
            GPStimeValue=GPStimeValue[index], magnification = magValue[index], DL = dLValue[index])
    if args.L:
        samples = np.array([m1, m2, redshiftValue, magValue, snr]).T
        names = ['m1', 'm2', 'z', 'mag', 'snr']
        lens_label = 'lensed'
    else:
        samples = np.array([m1, m2, redshiftValue, snr]).T
        names = ['m1', 'm2', 'z', 'snr']
        lens_label = 'unlensed'
   
    c = corner(samples, labels = names,plot_datapoints=False, levels=(0.5,0.8,0.95))
    c.savefig('source_frame_{0}.pdf'.format(lens_label))
    
    if args.L:
        samplesx = np.array([m1, m2, redshiftValue, magValue, snr]).T
        namesx = ['m1', 'm2', 'z', 'mag', 'snr']
        lens_label = 'lensed'
    else:
        samplesx = np.array([m1[index], m2[index], redshiftValue[index]]).T
        namesx = ['m1', 'm2', 'z']
        lens_label = 'sel'
   
    c = corner(samplesx, labels = namesx,plot_datapoints=False, levels=(0.5,0.8,0.95),plot_density=True)
    c.savefig('source_frame_{0}.pdf'.format(lens_label))
    
    samples[:,0] = samples[:,0]*(1+redshiftValue_l)
    samples[:,1] = samples[:,1]*(1+redshiftValue_l)
    print(time.time()-t0)
    c1 = corner(samples, labels = names,plot_datapoints=False, levels=(0.5,0.8,0.95))
    c1.savefig('detector_frame_{0}.pdf'.format(lens_label))

import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np
import dill
from lensingpop.population.sampler import MagnificationSampler, universe
from lensingpop.posterior.sampler import genPosterior

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate population and posterior samples.')
    parser.add_argument('-i', dest='file', type=str, help='Input file')
    parser.add_argument('--nSample', type=int, help='Number of posterior samples per event', default=1000)
    
    args = parser.parse_args()
    file = args.file
    nSample = int(args.nSample)  # Number of posterior samples per event

    np.random.seed(0)
    t0 = time.time()

    ################## Load SNR interpolator ##################
    snr_file = "./detectability/snr_m1qz_source.pkl"
    with open(snr_file, 'rb') as f:
        f_snr = dill.load(f)

    ################## Read m1, m2, z, snr from mock catalog ##################
    data = np.load(file)
    m1 = data['m1']
    q = data['q']
    redshift = data['redshift']
    m1z = m1 * (1 + redshift)
    dl = data['DL']
    snr = data['snr']
    
    Mz = m1z * q ** (3./5.) / (1+q)**(1./5.)  # Compute chirp mass
    snr_threshold = 8  # Define SNR threshold
    
    ################## Unlensed case ##################
    snr_obs_unlensed = snr + np.random.normal(loc=0.0, scale=1.0, size=snr.size)
    indices_unlensed = np.where(snr_obs_unlensed >= snr_threshold)[0]
    
    m1_unlensed = m1[indices_unlensed]
    m1z_unlensed = m1z[indices_unlensed]
    q_unlensed = q[indices_unlensed]
    Mz_unlensed = Mz[indices_unlensed]
    redshift_unlensed = redshift[indices_unlensed]
    dl_unlensed = dl[indices_unlensed]
    snr_unlensed = snr[indices_unlensed]
    snr_obs_unlensed = snr_obs_unlensed[indices_unlensed]
    
    ################## Lensed case ##################
    magValue1, magValue2 = MagnificationSampler(nSample=Mz.size)
    magValue = np.concatenate([magValue1, magValue2])
    
    m1_lensed = np.concatenate([m1, m1])
    m1z_lensed = np.concatenate([m1z, m1z])
    Mz_lensed = np.concatenate([Mz, Mz])
    q_lensed = np.concatenate([q, q]) 
    redshift_lensed = np.concatenate([redshift, redshift])
    dl_lensed = universe.omega.LuminosityDistance(redshift_lensed) / np.sqrt(np.abs(magValue))
    redshiftValue_lensed = np.array([universe.omega.Redshift(l) for l in dl_lensed])
    snr_lensed = f_snr(np.array([m1_lensed, q_lensed, redshiftValue_lensed]).T)
    snr_obs_lensed = snr_lensed + np.random.normal(loc=0.0, scale=1.0, size=snr_lensed.size)
    
    pairs = [(i, i + snr_lensed.size // 2) for i in range(snr_lensed.size // 2)]
    indices_lensed = [i for i, j in pairs if snr_obs_lensed[i] >= snr_threshold and snr_obs_lensed[j] >= snr_threshold] + \
                     [j for i, j in pairs if snr_obs_lensed[i] >= snr_threshold and snr_obs_lensed[j] >= snr_threshold]
    
    m1_lensed = m1_lensed[indices_lensed]
    m1z_lensed = m1z_lensed[indices_lensed]
    q_lensed = q_lensed[indices_lensed]
    Mz_lensed = Mz_lensed[indices_lensed]
    redshift_lensed = redshift_lensed[indices_lensed]
    dl_lensed = dl_lensed[indices_lensed]
    snr_lensed = snr_lensed[indices_lensed]
    snr_obs_lensed = snr_obs_lensed[indices_lensed]
    mag_lensed = magValue[indices_lensed]
    
    ################## Generate posterior samples ##################
    def generate_posteriors(Mz, q, redshift, snr_obs):
        size = Mz.size
        m1z_posterior = np.zeros((size, nSample))
        q_posterior = np.zeros((size, nSample))
        mc_posterior = np.zeros((size, nSample))
        eta_posterior = np.zeros((size, nSample))
        
        for i in tqdm(range(size), desc='Posteriors'):
            m1z_posterior[i], q_posterior[i], mc_posterior[i], eta_posterior[i] = genPosterior(
                Mz[i], q[i], redshift[i], snr_obs[i], nSample
            )
        return m1z_posterior, q_posterior, mc_posterior, eta_posterior

    m1z_posterior_unlensed, q_posterior_unlensed, mc_posterior_unlensed, eta_posterior_unlensed = \
        generate_posteriors(Mz_unlensed, q_unlensed, redshift_unlensed, snr_obs_unlensed)
    
    m1z_posterior_lensed, q_posterior_lensed, mc_posterior_lensed, eta_posterior_lensed = \
        generate_posteriors(Mz_lensed, q_lensed, redshift_lensed, snr_obs_lensed)
    
    ################## Save results ##################
    output_folder = Path('./catalog/')
    
    np.savez(output_folder / f'm1zq_posterior_afterSelection_unlensed{Mz_unlensed.size}.npz',
             m1z=m1z_unlensed, q=q_unlensed, redshift=redshift_unlensed,
             snr_opt=snr_unlensed, snr_obs=snr_obs_unlensed,
             m1z_posterior=m1z_posterior_unlensed, q_posterior=q_posterior_unlensed,
             mc_posterior=mc_posterior_unlensed, eta_posterior=eta_posterior_unlensed)
    
    np.savez(output_folder / f'm1zq_posterior_afterSelection_lensed{Mz_lensed.size//2}.npz',
             m1z=m1z_lensed, q=q_lensed, redshift=redshift_lensed, redshift_lensed=redshiftValue_lensed, magnification=mag_lensed,
             snr_opt=snr_lensed, snr_obs=snr_obs_lensed,
             m1z_posterior=m1z_posterior_lensed, q_posterior=q_posterior_lensed,
             mc_posterior=mc_posterior_lensed, eta_posterior=eta_posterior_lensed)

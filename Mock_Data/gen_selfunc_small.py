import numpy as np
import matplotlib.pyplot as plt
import optparse as op

import h5py
import subprocess
import dill

from figaro.mixture import DPGMM
from figaro.cosmology import CosmologicalParameters
from figaro.utils import plot_multidim, recursive_grid

from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
from pathlib import Path

"""
This code uses LVK's O3 search sensitivity estimates (https://zenodo.org/record/5546676#.Yw9cZS8RphF) to reconstruct the selection function for BBH coalescences.
"""

inj_file = 'endo3_bbhpop-LIGO-T2100113-v12.hdf5'

ifar_threshold         = 1
n_draws_dpgmm          = 20

# Injected parameters
min_mass1=2.0   # Msun
max_mass1=100.0 # Msun
pow_mass1=-2.35

min_mass2=2.0   # Msun
max_mass2=100.0 # Msun
pow_mass2=1.0

min_z=0.001
max_z=1.9
pow_z=1.0

# Cosmology
h  = 0.679 # km/s/Mpc
om = 0.3065
ol = 0.6935
omega = CosmologicalParameters(h, om, ol, -1, 0)

# Normalisations
logV = np.log(omega.ComovingVolume(max_z) - omega.ComovingVolume(min_z))
logM1 = np.log((max_mass1**(1+pow_mass1)-min_mass1**(1+pow_mass1))/(1+pow_mass1))
logM2 = np.log((max_mass2**(1+pow_mass2)-min_mass2**(1+pow_mass2))/(1+pow_mass2))

# Grid parameters
m_pts = 10
z_pts = 10

def load_injections(file, nsamples=10):
    with h5py.File(file, 'r') as f:
        inj = f['injections']

        ifar_pyCBC     = np.array(inj['ifar_pycbc_bbh'])
        ifar_gstlal    = np.array(inj['ifar_gstlal'])
        threshold = (ifar_pyCBC > ifar_threshold) | (ifar_gstlal > ifar_threshold)
        
        mass1 = np.array(inj['mass1_source'])
        mass2 = np.array(inj['mass2_source'])
        z     = np.array(inj['redshift'])
        
        # Get the total number of injections and then the total number of injections that were detected
        n_gen = float(np.array(f['injections'].attrs['total_generated']))
        n_rec = float(len(mass1))

        # Only choose nsamples to avoid excessive computational cost
        if nsamples == -1:
            nsamples = len(mass1)
        idx = np.random.randint(0,len(mass1),nsamples)
        mass1 = mass1[idx]
        mass2 = mass2[idx]
        z     = z[idx]
        
        # Only load the mass1, mass2, z samples
        samples_source   = np.array([mass1, mass2, z]).T
        samples_detector = np.array([mass1*(1+z), mass2*(1+z), z]).T
     
        det_frac = n_rec/n_gen
        return samples_source, samples_detector, det_frac

def log_mass_distribution_source_frame(m1, m2, log_norm):
    ''' Prior used by the LVK
    '''
    p_m1 = np.log(m1)*pow_mass1 - logM1
    p_m2 = np.log(m2)*pow_mass2 - logM2
    p_m2[m1 < m2] = -np.inf
    return p_m1 + p_m2 - log_norm
    
def injected_distribution_source_frame(m1, m2, z, log_norm_mass):
    p_m = log_mass_distribution_source_frame(m1,m2,z)
    p_z = np.log([omega.ComovingVolumeElement_double(zi) for zi in z]) - logV
    return np.exp(p_m + p_z)

def injected_distribution_detector_frame(m1z, m2z, z, log_norm_mass):
    return injected_distribution_source_frame(m1z/(1+z), m2z/(1+z), z, log_norm_mass)/(1+z)**2

def marginalise_distribution(pdf_reconstructed, det_frac, frame): # 'source', 'detector'
    ''' This function is used to get the pdet(theta); in this case the theta is m1,m2 in source of detector frame. 

        ARGS:
        -----
        - pdf_reconstructed: The reconstructed pdf p(theta|det)
        - det_frac: Fraction of detected events p(det) = N_detected/N_total
        - frame: 'source' or 'detector' frame (returns the pdet function in source or detector frame)
        -----
        - returns: p_m1m2z, p_m1m2, m1, m2, z
    '''
    if frame == 'source':
        upper_mass_1 = max_mass1
        upper_mass_2 = max_mass2
    elif frame == 'detector':
        upper_mass_1 = max_mass1*(1+max_z)
        upper_mass_2 = max_mass2*(1+max_z)
    else:
        raise ValueError("Error: Invalid frame value:",frame)
    
    m1 = np.linspace(min_mass1, upper_mass_1, m_pts)
    m2 = np.linspace(min_mass2, upper_mass_2, m_pts)
    z  = np.linspace(min_z, max_z, z_pts)
    
    grid_norm, dgrid_norm = recursive_grid([[min_mass1, upper_mass_1], [min_mass2, upper_mass_2]], [m_pts, m_pts])
    
    log_norm_mass = np.log(np.sum(np.exp(log_mass_distribution_source_frame(grid_norm[:,0], grid_norm[:,1], 0))*np.prod(dgrid_norm)))

    dm1 = m1[1]-m1[0]
    dm2 = m2[1]-m2[0]
    dz  = z[1]-z[0]
    
    p_m1m2z = np.zeros((m_pts, m_pts, z_pts))
    p_m1m2  = np.zeros((m_pts, m_pts))
    
    for i in tqdm(range(m_pts), desc = frame):
        for j in range(m_pts):
            if m1[i] >= m2[j]:
                vals = np.array([[m1[i], m2[j], zi] for zi in z])
                rec  = pdf_reconstructed(vals)
                if frame == 'source':
                    inj = injected_distribution_source_frame(vals[:,0], vals[:,1], vals[:,2], log_norm_mass)
                    idx = (m1[i] > min_mass1) & (m2[j] > min_mass2) & (m1[i] < max_mass1) & (m2[j] < max_mass2)
                if frame == 'detector':
                    inj = injected_distribution_detector_frame(vals[:,0], vals[:,1], vals[:,2], log_norm_mass)
                    idx = (m1[i]/(1+z) > min_mass1) & (m2[j]/(1+z) > min_mass2) & (m1[i]/(1+z) < max_mass1) & (m2[j]/(1+z) < max_mass2)
                rec[~idx] = 0.

                s = rec/inj
                p_m1m2z[i,j,:] = s
                p_m1m2[i,j] = np.sum(s*dz)
    
    p_m1m2z /= np.sum(p_m1m2z*dm1*dm2*dz)
    p_m1m2z *= det_frac
    
    p_m1m2  /= np.sum(p_m1m2*dm1*dm2)
    p_m1m2  *= det_frac
    
    return p_m1m2z, p_m1m2, m1, m2, z

if __name__ == '__main__':

    parser = op.OptionParser()
    parser.add_option("-d", dest = "draw_dists", action = 'store_false', help = "Skip DPGMM reconstruction", default = True)
    parser.add_option("-e", dest = "evaluate_interpolators", action = 'store_false', help = "Skip both DPGMM reconstruction and interpolator evaluation", default = True)
    parser.add_option("-p", dest = "make_selfuncs", action = 'store_false', help = "Produce plots only", default = True)
    
    (options, args) = parser.parse_args()
    
    if options.make_selfuncs == False:
        options.evaluate_interpolators = False

    if options.evaluate_interpolators == False:
        options.draw_dists = False
    
    if not Path(inj_file).exists():
        print('Please download the dataset endo3_bbhpop-LIGO-T2100113-v12.hdf5 from https://zenodo.org/record/5546676#.Yw9cZS8RphF')
        exit()

    selfunc_folder = Path('selection_functions').resolve()
    if not selfunc_folder.exists():
        selfunc_folder.mkdir()
        
    # Load (m1,m2,z) in source and detector frame, also also give the detected fraction
    samples_source, samples_detector, det_frac = load_injections(inj_file)
    
    if options.draw_dists:
        # Source-frame observed distribution
        bounds_source = np.array([[min_mass1-0.1, max_mass1+0.1], [min_mass2-0.1, max_mass2+0.1], [-0.001, max_z+0.001]]) # Boundaries
        mix_source    = DPGMM(bounds_source) # Initialize DPGMM
        # Infer the density p(\theta) in the source frame quantities (this includes all of the parameters ). 
        # Note: Inferring this N times, where N = n_draws_dpgmm
        draws_source  = np.array([mix_source.density_from_samples(samples_source) for _ in tqdm(range(n_draws_dpgmm), desc = 'Source frame')]) # 
        with open(Path(selfunc_folder, 'draws_source.pkl'), 'wb') as f:
            dill.dump(draws_source, f)
        try:
            plot_multidim(draws_source, samples = samples_source, out_folder = selfunc_folder, name = 'source_frame_obs_dist', labels = ['M_1', 'M_2', 'z'], units = ['M_\\odot', 'M_\\odot', ''])
        except:
            pass
        
        # Detector-frame observed distribution (same as above but for detector frame)
        bounds_detector = np.array([[min_mass1-0.1, max_mass1*(1+max_z)+0.1], [min_mass2-0.1, max_mass2*(1+max_z)+0.1], [-0.001, max_z+0.001]])
        mix_detector    = DPGMM(bounds_detector)
        draws_detector = np.array([mix_detector.density_from_samples(samples_detector) for _ in tqdm(range(n_draws_dpgmm), desc = 'Detector frame')])
        # Save results:
        with open(Path(selfunc_folder, 'draws_detector.pkl'), 'wb') as f:
            dill.dump(draws_detector, f)
        try:
            plot_multidim(draws_detector, samples = samples_detector, out_folder = selfunc_folder, name = 'detector_frame_obs_dist', labels = ['M_1', 'M_2', 'z'], units = ['M_\\odot', 'M_\\odot', ''])
        except:
            pass
    else:
        # Load results:
        with open(Path(selfunc_folder, 'draws_source.pkl'), 'rb') as f:
            draws_source = dill.load(f)
        with open(Path(selfunc_folder, 'draws_detector.pkl'), 'rb') as f:
            draws_detector = dill.load(f)

    if options.evaluate_interpolators:
        # Source-frame grid
        m1 = np.linspace(min_mass1, max_mass1, m_pts)
        m2 = np.linspace(min_mass2, max_mass2, m_pts)
        z  = np.linspace(min_z, max_z, z_pts)
        m1_grid, m2_grid, z_grid = np.meshgrid(m1,m2,z)
        vals_source = np.array([[m1i, m2i, zi] for m1i,m2i,zi in zip(m1_grid.flatten(), m2_grid.flatten(), z_grid.flatten())])
    
        # Detector-frame grid
        m1z = np.linspace(min_mass1, max_mass1*(1+max_z), m_pts)
        m2z = np.linspace(min_mass2, max_mass2*(1+max_z), m_pts)
        z   = np.linspace(min_z, max_z, z_pts)
        m1z_grid, m2z_grid, z_grid = np.meshgrid(m1z,m2z,z)
        vals_detector = np.array([[m1zi, m2zi, zi] for m1zi,m2zi,zi in zip(m1z_grid.flatten(), m2z_grid.flatten(), z_grid.flatten())])

        # Source-frame median distribution
        pdfs_source       = np.array([d.pdf(vals_source) for d in tqdm(draws_source, desc = 'Source-frame draws')]) # 2D array
        median_source     = np.percentile(pdfs_source, 50, axis = 0).reshape((m_pts, m_pts, z_pts)) # Get median for vals_source and put into 3D grid
        pdf_median_source = RegularGridInterpolator(points = (m1, m2, z), values = median_source, method = 'linear') # Make a 3D interpolator
        
        # Detector-frame median distribution
        pdfs_detector       = np.array([d.pdf(vals_detector) for d in tqdm(draws_detector, desc = 'Detector-frame draws')])
        median_detector     = np.percentile(pdfs_detector, 50, axis = 0).reshape((m_pts, m_pts, z_pts))
        pdf_median_detector = RegularGridInterpolator(points = (m1z, m2z, z), values = median_detector, method = 'linear')
        
        # Save interpolators:
        with open('rec_interpolators.pkl', 'wb') as f:
            dill.dump([pdf_median_source, pdf_median_detector], f)
    
    else:
        with open('rec_interpolators.pkl', 'rb') as f:
            pdf_median_source, pdf_median_detector = dill.load(f)
    
    if options.make_selfuncs:
        # Source-frame selection functions (2D & 3D)
        p_m1m2z_source, p_m1m2_source, m1, m2, z = marginalise_distribution(pdf_median_source, det_frac, frame = 'source') # Gets pdet(theta) for theta values m1, m2, z
        # 3D
        selection_function_m1m2z_source = RegularGridInterpolator(points = (m1,m2,z), values = p_m1m2z_source, method = 'linear') # Creates an interpolator for pdet(theta)
        with open(Path(selfunc_folder, 'selfunc_m1m2z_source.pkl'), 'wb') as f:
            dill.dump(selection_function_m1m2z_source, f)
        # 2D
        selection_function_m1m2_source = RegularGridInterpolator(points = (m1,m2), values = p_m1m2_source, method = 'linear')
        with open(Path(selfunc_folder, 'selfunc_m1m2_source.pkl'), 'wb') as f:
            dill.dump(selection_function_m1m2_source, f)

        # Detector-frame selection functions (2D & 3D)
        p_m1m2z_detector, p_m1m2_detector, m1z, m2z, z = marginalise_distribution(pdf_median_detector, det_frac, frame = 'detector')
        # 3D
        selection_function_m1m2z_detector = RegularGridInterpolator(points = (m1z,m2z,z), values = p_m1m2z_detector, method = 'linear')
        with open(Path(selfunc_folder, 'selfunc_m1m2z_detector.pkl'), 'wb') as f:
            dill.dump(selection_function_m1m2z_detector, f)
        # 2D
        selection_function_m1m2_detector = RegularGridInterpolator(points = (m1z,m2z), values = p_m1m2_detector, method = 'linear')
        with open(Path(selfunc_folder, 'selfunc_m1m2_detector.pkl'), 'wb') as f:
            dill.dump(selection_function_m1m2_detector, f)
    else:
        with open(Path(selfunc_folder, 'selfunc_m1m2_source.pkl'), 'rb') as f:
            selection_function_m1m2_source = dill.load(f)
        with open(Path(selfunc_folder, 'selfunc_m1m2_detector.pkl'), 'rb') as f:
            selection_function_m1m2_detector = dill.load(f)
        
    # Plots
    # Source-frame
    fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True)

    m1  = np.linspace(min_mass1, max_mass1, m_pts+2)[1:-1]
    m2  = np.linspace(min_mass2, max_mass2, m_pts+2)[1:-1]
    dm1 = m1[1]-m1[0]
    dm2 = m2[1]-m2[0]
    m1_grid, m2_grid = np.meshgrid(m1, m2)
    vals_source = np.array([[m1i, m2i] for m1i,m2i in zip(m1_grid.flatten(), m2_grid.flatten())])
    sf_m1m2 = selection_function_m1m2_source(vals_source).reshape((m_pts, m_pts))

    # M1
    sf_m1  = np.sum(sf_m1m2, axis = 1)*dm2
    sf_m1 /= np.sum(sf_m1)*dm1
    ax1.plot(m1, sf_m1, lw = 0.8)
    ax1.set_ylabel('$S_1(M_1)$')
    ax1.grid()
    # M2
    sf_m2  = np.sum(sf_m1m2, axis = 0)*dm1
    sf_m2 /= np.sum(sf_m2)*dm2
    ax2.plot(m2, sf_m2, lw = 0.8)
    ax2.set_ylabel('$S_2(M_2)$')
    ax2.set_xlabel('$M_{1,2}$')
    ax2.grid()
    fig.savefig(Path(selfunc_folder, 'selection_function_source_frame.pdf'), bbox_inches = 'tight')

    # Detector-frame
    fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True)

    m1z  = np.linspace(min_mass1, max_mass1*(1+max_z), m_pts+2)[1:-1]
    m2z  = np.linspace(min_mass2, max_mass2*(1+max_z), m_pts+2)[1:-1]
    dm1z = m1z[1]-m1z[0]
    dm2z = m2z[1]-m2z[0]
    m1z_grid, m2z_grid = np.meshgrid(m1z, m2z)
    vals_detector = np.array([[m1zi, m2zi] for m1zi,m2zi in zip(m1z_grid.flatten(), m2z_grid.flatten())])
    sf_m1zm2z = selection_function_m1m2_detector(vals_detector).reshape((m_pts, m_pts))

    # M1
    sf_m1z  = np.sum(sf_m1zm2z, axis = 1)*dm2z
    sf_m1z /= np.sum(sf_m1z)*dm1z
    ax1.plot(m1z, sf_m1z, lw = 0.8)
    ax1.set_ylabel('$S_1(M_1^z)$')
    ax1.grid()
    # M2
    sf_m2z  = np.sum(sf_m1zm2z, axis = 0)*dm1z
    sf_m2z /= np.sum(sf_m2z)*dm2z
    ax2.plot(m2z, sf_m2z, lw = 0.8)
    ax2.set_ylabel('$S_2(M_2^z)$')
    ax2.set_xlabel('$M_{1,2}^z$')
    ax2.grid()
    fig.savefig(Path(selfunc_folder, 'selection_function_detector_frame.pdf'), bbox_inches = 'tight')

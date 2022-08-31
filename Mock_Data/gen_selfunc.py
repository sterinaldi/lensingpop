import numpy as np
import matplotlib.pyplot as plt
import h5py
import subprocess
import dill
from figaro.mixture import DPGMM
from figaro.cosmology import CosmologicalParameters
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
from pathlib import Path

"""
This code uses LVK's O3 search sensitivity estimates (https://zenodo.org/record/5546676#.Yw9cZS8RphF) to reconstruct the selection function for BBH coalescences.
"""

inj_file = 'endo3_bbhpop-LIGO-T2100113-v12.hdf5'

ifar_threshold = 1
n_draws_dpgmm  = 100
draw_dists     = True
make_selfuncs  = True

# Injected parameters
min_mass1=2.0   # Msun
max_mass1=100.0 # Msun
pow_mass1=-2.35

min_mass2=2.0   # Msun
max_mass2=100.0 # Msun
pow_mass2=1.0

max_z=1.9
pow_z=1.0

# Cosmology
h  = 0.679 # km/s/Mpc
om = 0.3065
ol = 0.6935
omega = CosmologicalParameters(h, om, ol, -1, 0)
vol_max = omega.ComovingVolume(max_z)

# Grid parameters
m_pts = 100
z_pts = 100

def load_injections(file):
    with h5py.File(file, 'r') as f:
        inj = f['injections']

        ifar_pyCBC     = np.array(inj['ifar_pycbc_bbh'])
        ifar_gstlal    = np.array(inj['ifar_gstlal'])
        threshold = (ifar_pyCBC > ifar_threshold) | (ifar_gstlal > ifar_threshold)
        
        mass1 = np.array(inj['mass1_source'])
        mass2 = np.array(inj['mass2_source'])
        z     = np.array(inj['redshift'])
        
        n_gen = float(np.array(f['injections'].attrs['total_generated']))
        n_rec = float(len(mass1))
        
        samples_source   = np.array([mass1, mass2, z]).T
        samples_detector = np.array([mass1*(1+z), mass2*(1+z), z]).T
        
        return samples_source, samples_detector, n_rec/n_gen

def injected_distribution_source_frame(m1, m2, z):
    p_m1 = m1**pow_mass1 * (1+pow_mass1)/(max_mass1**(pow_mass1+1) - min_mass1**(pow_mass1+1))
    p_m2 = m2**pow_mass2 * (1+pow_mass2)/(max_mass2**(pow_mass2+1) - min_mass1**(pow_mass2+1))
    p_z  = omega.ComovingVolumeElement(np.ascontiguousarray(z))/vol_max
    # Impose m1 => m2
    p_m2[np.where(m2 > m1)] = np.inf # float/np.inf = 0.0, useful in the following
    return p_m1 * p_m2 * p_z

def injected_distribution_detector_frame(m1z, m2z, z):
    return injected_distribution_source_frame(m1z/(1+z), m2z/(1+z), z)/(1+z)**2

def marginalise_distribution(p_rec, det_frac, frame): # 'source', 'detector'
    
    if frame == 'source':
        upper_mass_1 = max_mass1
        upper_mass_2 = max_mass2
        
    if frame == 'detector':
        upper_mass_1 = max_mass1*(1+max_z)
        upper_mass_2 = max_mass2*(1+max_z)
    
    m1 = np.linspace(min_mass1, upper_mass_1, m_pts+2)[1:-1]
    m2 = np.linspace(min_mass2, upper_mass_2, m_pts+2)[1:-1]
    z  = np.linspace(0, max_z, z_pts+2)[1:-1]

    dm1 = m1[1]-m1[0]
    dm2 = m2[1]-m2[0]
    dz  = z[1]-z[0]
    
    p_m1m2z = np.zeros((m_pts, m_pts, z_pts))
    p_m1m2  = np.zeros((m_pts, m_pts))
    
    for i in tqdm(range(m_pts), desc = frame):
        for j in range(m_pts):
            if m1[i] >= m2[j]:
                vals = np.array([[m1[i], m2[j], zi] for zi in z])
                rec  = p_rec(vals)
                if frame == 'source':
                    inj = injected_distribution_source_frame(vals[:,0], vals[:,1], vals[:,2])
                if frame == 'detector':
                    inj = injected_distribution_detector_frame(vals[:,0], vals[:,1], vals[:,2])
                    idx = (m1[i]/(1+z) > min_mass1) & (m2[j]/(1+z) > min_mass2) & (m1[i]/(1+z) < max_mass1) & (m2[j]/(1+z) < max_mass2)
                    inj[~idx] = np.inf

                s = rec/inj
                p_m1m2z[i,j,:] = s
                p_m1m2[i,j] = np.sum(s*dz)
    
    p_m1m2z /= np.sum(p_m1m2z*dm1*dm2*dz)
    p_m1m2z *= det_frac
    
    p_m1m2  /= np.sum(p_m1m2*dm1*dm2)
    p_m1m2  *= det_frac
    
    return p_m1m2z, p_m1m2, m1, m2, z

if __name__ == '__main__':
    
    if not Path(inj_file).exists():
        print('Please download the dataset endo3_bbhpop-LIGO-T2100113-v12.hdf5 from https://zenodo.org/record/5546676#.Yw9cZS8RphF')
        exit()

    selfunc_folder = Path('selection_functions').resolve()
    if not selfunc_folder.exists():
        selfunc_folder.mkdir()
        
    samples_source, samples_detector, det_frac = load_injections(inj_file)
    
    if draw_dists:
        # Source-frame observed distribution
        bounds_source = np.array([[min_mass1, max_mass1], [min_mass2, max_mass2], [0, max_z]])
        mix_source    = DPGMM(bounds_source)
        draws_source   = np.array([mix_source.density_from_samples(samples_source) for _ in tqdm(range(n_draws_dpgmm), desc = 'Source frame')])
        # Detector-frame observed distribution
        bounds_detector = np.array([[min_mass1, max_mass1*(1+max_z)], [min_mass2, max_mass2*(1+max_z)], [0., max_z]])
        mix_detector    = DPGMM(bounds_detector)
        draws_detector = np.array([mix_detector.density_from_samples(samples_detector) for _ in tqdm(range(n_draws_dpgmm), desc = 'Detector frame')])
        
        # Source-frame grid
        m1 = np.linspace(min_mass1, max_mass1, m_pts+2)[1:-1]
        m2 = np.linspace(min_mass2, max_mass2, m_pts+2)[1:-1]
        z  = np.linspace(0, max_z, z_pts+2)[1:-1]
        m1_grid, m2_grid, z_grid = np.meshgrid(m1,m2,z)
        vals_source = np.array([[m1i, m2i, zi] for m1i,m2i,zi in zip(m1_grid.flatten(), m2_grid.flatten(), z_grid.flatten())])
    
        # Detector-frame grid
        m1z = np.linspace(min_mass1, max_mass1*(1+max_z), m_pts+2)[1:-1]
        m2z = np.linspace(min_mass2, max_mass2*(1+max_z), m_pts+2)[1:-1]
        z   = np.linspace(0, max_z, z_pts+2)[1:-1]
        m1z_grid, m2z_grid, z_grid = np.meshgrid(m1z,m2z,z)
        vals_detector = np.array([[m1zi, m2zi, zi] for m1zi,m2zi,zi in zip(m1z_grid.flatten(), m2z_grid.flatten(), z_grid.flatten())])

        # Source-frame median distribution
        pdfs_source       = np.array([d.pdf(vals_source) for d in tqdm(draws_source, desc = 'Source-frame draws')])
        median_source     = np.percentile(pdfs_source, 50, axis = 0).reshape((m_pts, m_pts, z_pts))
        pdf_median_source = RegularGridInterpolator(points = (m1, m2, z), values = median_source, method = 'linear')
        
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
    
    if make_selfuncs:
        # Source-frame selection functions (2D & 3D)
        p_m1m2z_source, p_m1m2_source, m1, m2, z = marginalise_distribution(pdf_median_source, det_frac, frame = 'source')
        # 3D
        selection_function_m1m2z_source = RegularGridInterpolator(points = (m1,m2,z), values = p_m1m2z_source, method = 'linear')
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
    ax2.set_ylabel('$S_2(M_2)$')
    ax2.set_xlabel('$M_{1,2}^z$')
    ax2.grid()
    fig.savefig(Path(selfunc_folder, 'selection_function_detector_frame.pdf'), bbox_inches = 'tight')

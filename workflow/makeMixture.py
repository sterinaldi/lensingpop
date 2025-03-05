import numpy as np
from figaro.mixture import DPGMM 
from figaro.load import load_density, save_density
from tqdm import tqdm

def process_population(data_file, output_folder, is_lensed=True):
    """
    Processes a given population (lensed or unlensed) by estimating density distributions
    using the Dirichlet Process Gaussian Mixture Model (DPGMM) and saving the results.

    Parameters:
    - data_file (str): Path to the input .npz file containing posterior samples.
    - output_folder (str): Path to the output folder where densities will be saved.
    - is_lensed (bool): Flag to determine whether the data is from lensed events (default: True).
    """
    # Load data from the provided file
    data = np.load(data_file)
    m1z = data['m1z']  # Primary mass in redshifted frame
    q = data['q']       # Mass ratio
    m1zp = data['m1z_posterior']  # Posterior samples for m1z
    qp = data['q_posterior']       # Posterior samples for q

    # Define parameter bounds for DPGMM
    bounds = [[5, 230], [0, 1]]
    dpgmm = DPGMM(bounds)

    # Iterate over all events to estimate density distributions
    for i in tqdm(range(m1z.size // (2 if is_lensed else 1)), desc=f"Processing {'lensed' if is_lensed else 'unlensed'} events"):
        
        # Process first event in the pair (for lensed) or single event (for unlensed)
        name = f"{'lensed' if is_lensed else 'unlensed'}_pair{i+1}_event1"
        s = np.array([m1zp[i], qp[i]]).T
        m = dpgmm.density_from_samples(s)
        save_density([m], output_folder, name)
        dpgmm.initialise()

        # Process second event in the pair (only for lensed events)
        if is_lensed:
            name = f"lensed_pair{i+1}_event2"
            s = np.array([m1zp[i + m1z.size // 2], qp[i + m1z.size // 2]]).T
            m = dpgmm.density_from_samples(s)
            save_density([m], output_folder, name)
            dpgmm.initialise()

def main():
    """
    Main function to process both lensed and unlensed populations and save density distributions.
    """
    # File paths for lensed and unlensed populations
    lensed_file = './catalog/m1zq_posterior_afterSelection_lensed1181.npz'
    unlensed_file = './catalog/m1zq_posterior_afterSelection_unlensed1519.npz'

    # Output folders
    lensed_output_folder = './lensed_events/'
    unlensed_output_folder = './unlensed_events/'

    # Process lensed and unlensed populations
    process_population(lensed_file, lensed_output_folder, is_lensed=True)
    process_population(unlensed_file, unlensed_output_folder, is_lensed=False)

if __name__ == "__main__":
    main()

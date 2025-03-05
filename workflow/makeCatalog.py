import numpy as np
import argparse
import dill
import time 
from tqdm import tqdm
from scipy.stats import uniform
from corner import corner
from pathlib import Path
from lensingpop.population.sampler import RedshiftSampler, MassSampler, universe
np.random.seed(7)

parser = argparse.ArgumentParser(description='Generate population and posterior samples.')
parser.add_argument("-N", type = int, help = "Number of population events", default=10000)
args = parser.parse_args()
N = args.N # nunber of events

t0 = time.time()

selecfun_file = "{}/detectability/selfunc_m1qz_source.pkl".format(homedir)
snr_file = "{}/detectability/snr_m1qz_source.pkl".format(homedir)

with open(snr_file,'rb') as f:
    fun_snr = dill.load(f)
    
with open(selecfun_file, "rb") as f:
    fun_pdet = dill.load(f)
    
#################################### Start the sampling shenanigans ####################################
redshiftValue = RedshiftSampler(nSample=N)
m1, m2 = MassSampler(redshiftValue)
q = m2 / m1
dLValue = universe.omega.LuminosityDistance(redshiftValue)
#################################### Calculate the detector SNR using the method from Roulet et al. (2020) ##################
snr = fun_snr(np.array([m1, q, redshiftValue]).T) 

################## Save the data pf intrinsic catalog ########################################################
output_folder = Path('./catalog/')
output_folder.mkdir(parents=True, exist_ok=True)
intrinsic_catalog_file = "./catalog/IntrinsicCatalog_N{:.0f}.npz".format(N)
    

np.savez(intrinsic_catalog_file, m1=m1, q=q, redshift=redshiftValue, snr=snr, DL = dLValue)
print("Intrinsic catalog saved to {}.".format(intrinsic_catalog_file))

#################################### applying detectability ###########################################
pdet_value = fun_pdet(np.array([m1, q, redshiftValue]).T)

randnum = uniform.rvs(0, 1,size=len(m1))
index = randnum < pdet_value
print("Number of events in the catalog: {0} - after selection: {1}".format(N, len(m1[index])))

#################################### Save the data after applying selection effect 
observed_catalog_file = "./catalog/ObservedCatalog_N{:.0f}_afterSelection_N{:.0f}.npz".format(N, m1[index].size)

np.savez(observed_catalog_file, m1=m1[index], q=q[index], redshift=redshiftValue[index],
         snr=snr[index], DL = dLValue[index])
print("Observed catalog saved to {}.".format(observed_catalog_file))

samples = np.array([m1[index], q[index], redshiftValue[index]]).T
names = [r'$m_1$', r'$q$', r'$z$']

c = corner(samples, labels = names, plot_datapoints=False, fill_contours=False, plot_density=False, levels=(0.5,0.8,0.95))
c.savefig('{0}/catalog/src_frame_pop.pdf'.format(homedir))

samplesx = np.array([m1[index]*(1+redshiftValue[index]), q[index], redshiftValue[index]]).T

c = corner(samplesx, labels = names, plot_datapoints=False, fill_contours=False, plot_density=False, levels=(0.5,0.8,0.95))
c.savefig('{0}/catalog/det_frame_pop.pdf'.format(homedir))

print("Time used = {}.".format(time.time()-t0))

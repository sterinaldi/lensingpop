import numpy as np
import matplotlib.pyplot as plt
import dill                # serialize python objects

from tqdm import tqdm     # advancement bar
from pathlib import Path  # initiate a concrete path for the platform the code is running on

from figaro.mixture import DPGMM, HDPGMM  # for inference on data driven posterior
from figaro.utils import get_priors, plot_median_cr, plot_multidim
from scipy.stats import norm
from ray.util import ActorPool           # to parallelize
import ray                               # to paralallelize


# let's parallelize with ray

@ray.remote
class worker:
    def __init__(self, bounds, out_folder, mu = None, sigma = None, all_samples = None):
        self.bounds               = bounds
        self.mixture              = DPGMM(self.bounds)
        self.hierarchical_mixture = HDPGMM(self.bounds,
                                           prior_pars = get_priors(self.bounds, std = sigma, mean = mu, samples = all_samples),
                                           )
        self.mu    = mu
        self.sigma = sigma
        self.out_folder = out_folder

    def run_event(self, pars):
        # Unpack parameters
        samples, name, n_draws = pars
        ev = np.copy(samples)
        ev.setflags(write = True)
        # Re-initialise the prior parameters of the mixture estimating them from samples (get_priors does the trick)
        self.mixture.initialise(prior_pars = get_priors(self.bounds, samples = ev))
        # Draw samples (the old for loop - now, density_from_samples automatically shuffles the samples and initialises the mixture, returning a draw (the old build_mixture output)
        draws = [self.mixture.density_from_samples(ev) for _ in tqdm(range(n_draws), desc=name)]
        # Fancy method for making plots
#        plot_multidim(draws, samples = ev, out_folder = self.out_folder, labels = ["M_1","M_2"], units = ["M_{\\odot}","M_{\\odot}"], name = name, subfolder = True)
        return draws

    def draw_hierarchy(self, pars):
        draws, n_draws = pars
        # As above
        mix_draws = [self.hierarchical_mixture.density_from_samples(draws) for _ in tqdm(range(n_draws), desc='Sampling')]
        return mix_draws


############################
ray.init()              # initiate ray

n_parallel_workers = 4  # number of ray workers. you can change this number
n_draws_event = 1       # number of draws for single events. you can change this number
n_draws = 4             # number of draws for hierarchical. you can change this number
n_draws_worker = n_draws // n_parallel_workers # number tasks for single worker

file = Path("/pbs/home/m/mtoscani/Martina/lensingpop/Mock_Data/m1m2zxeffxp_posterior_PPD_afterSelection_unlensed3303.npz")  # input file. modify this line according to your path
out_folder = Path("./test_output")                                 # output folder. modify this line according to tour path
if not out_folder.exists():
    out_folder.mkdir()
data = np.load(file)

Mmin = 0      # minimum mass for the mass range. you can change this value accordingly to your data
Mmax = 240    # maximum mass for the mass range. you can change this value accordingly to your data
#M = np.linspace(Mmin, Mmax, 1002)[1:-1]
#dM = M[1] - M[0]
zmin= 1.0e-17       # minimum z
zmax= 1.9     # maximum z
#z = np.linspace(zmin, zmax, 1002)[1:-1] #credo? la len del posterior è 1000
xeff_min=-1
xeff_max=1
xp_min=0
xp_max=1

fivedim_posteriors=[np.array([mass1[(mass1<Mmax) & (z<zmax) & (xeff<xeff_max) & (xeff_min<xeff) & (xp<xp_max)],
                    mass2[(mass1<Mmax) & (z<zmax)& (xeff<xeff_max) & (xeff_min<xeff) & (xp<xp_max)],
                    z[(mass1<Mmax) & (z<zmax) & (xeff<xeff_max) & (xeff_min<xeff) & (xp<xp_max)],
                    xeff[(mass1<Mmax) & (z<zmax) & (xeff<xeff_max) & (xeff_min<xeff) & (xp<xp_max)],
                    xp[(mass1<Mmax) & (z<zmax) & (xeff<xeff_max) & (xeff_min<xeff) & (xp<xp_max)]]).T
                    for mass1,mass2,z,xeff,xp in zip(data["m1_posterior"],data["m2_posterior"],
                    data["z_posterior"],data["xeff_posterior"],data["xp_posterior"])]
true_sample   = np.array([data["m1"]*(1+data["redshift"]),data["m2"]*(1+data["redshift"]),
                         data["redshift"],data["xeff"],data["xp"]]).T # Since we have the true masses, let's use them for the plots
#############################
# single event

run_events  = True
postprocess = False   # If True, the hierarchical analysis is skipped

all_samples = np.array([np.mean(m, axis = 0) for m in fivedim_posteriors])
print(all_samples.shape)
"""
(STE)
Now the probit transformation is completely transparent to the final user
Mean and covariance of all samples are not needed anymore (the samples are sufficient).
We have three different possibilities:

    * specify expected mean and std in *natural* space as one normally would do;
    * pass the samples to the get_priors() method;
    * rely on default prior parameters.

By default, mean and std overrides samples.
"""
mu    = None
sigma = None # Msun. Use None if you want to let the code estimate it from the samples themselves.
bounds = np.array([[Mmin, Mmax], [Mmin, Mmax], [zmin, zmax], [xeff_min, xeff_max], [xp_min, xp_max]])
pool = ActorPool(
  [
    worker.remote(bounds, out_folder = out_folder, mu = mu, sigma = sigma, all_samples = all_samples)
    for _ in range(n_parallel_workers)
  ]
)
names = [
    str(i) for i in range(len(fivedim_posteriors))
]

if not postprocess:
    if run_events:
        posteriors = []
        for s in pool.map_unordered(
            lambda a, v: a.run_event.remote(v),
            [[ev, name, n_draws_event] for ev, name in zip(fivedim_posteriors, names)],
        ):
            posteriors.append(s)
        with open(Path(out_folder, "posteriors.pkl"), "wb") as dill_file:
            dill.dump(
                posteriors, dill_file
            )  # scrivo e salvo il file posterios.pkl; lo statement with chiude il file in automatico
    else:
        with open(Path(out_folder, "posteriors.pkl"), "rb") as dill_file:
            posteriors = dill.load(dill_file)

    posteriors = np.array(posteriors)

    #############################
    # hierchical

    draws = []
    for s in pool.map_unordered(
        lambda a, v: a.draw_hierarchy.remote(v),
        [[posteriors, n_draws_worker] for _ in range(n_parallel_workers)],
    ):
        draws.append(s)
    draws = np.array(draws)
    draws = np.concatenate(draws)
    with open(Path(out_folder, "posteriors_hier.pkl"), "wb") as dill_file:
        dill.dump(draws, dill_file)
else:
    with open(Path(out_folder, "posteriors_hier.pkl"), "rb") as dill_file:
        draws = dill.load(dill_file)

plot_multidim(draws, samples = true_sample, out_folder = out_folder, name = '5D_model', labels = ["M_1","M_2","z","x_eff","x_p"], units = ["M_{\\odot}","M_{\\odot}","","",""], hierarchical = True)

import numpy as np
import corner.corner as cc
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as tck
import dill
import time
import emcee
from pathlib import Path
from lensingpop.population import simulated_universe as universe
from lensingpop.utils.utils import funInterpolator, KDEEstimator

# Load data
file = './catalog/m1zq_posterior_afterSelection_unlensed1519.npz'
data = np.load(file)
m1z = data['m1z']
q = data['q'] 
samples = np.array([m1z, q]).T

# ======================= KDE Estimation =======================
# Fit and save KDE estimator
kde = KDEEstimator(samples)

out_folder = Path('./prior/population_models/')
output_folder.mkdir(parents=True, exist_ok=True)

kde.save(out_folder + "benchmark_obs_m1zq_kde.pkl")

# Load the saved KDE estimator
loaded_kde = KDEEstimator.load(out_folder + "benchmark_obs_m1zq_kde.pkl")
# Define grid for evaluation
x = np.linspace(5, 140, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
positions = np.vstack([X.ravel(), Y.ravel()])
Z = loaded_kde.pdf(positions.T).reshape(X.shape)

# Plot KDE estimate
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.8)
plt.colorbar(label="Estimated Probability Density")
plt.xlabel(r"$m_1^z$")
plt.ylabel(r"$q$")
plt.title("Estimated 2D PDF from Samples")
plt.savefig(out_folder + "benchmark_kde.png")  # Save plot

# ======================= Hierarchical PDFs =======================
from figaro.load import load_density
hier = load_density(out_folder + "/draws_observed_figaro.json")

# Compute PDF values across all distributions in hier
pdf_values = np.array([dist.pdf(np.array([X.flatten(), Y.flatten()]).T) for dist in hier])
Z_median = np.median(pdf_values, axis=0).reshape(100, 100)

# Plot median hierarchical PDF
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z_median, levels=30, cmap='viridis')
plt.colorbar(label="Median Probability Density")
plt.xlabel("m1z")
plt.ylabel("q")
plt.title("Median of 2D Probability Density Functions")
plt.xlim(5, 140)
plt.ylim(0, 1)
plt.savefig(out_folder + "(H)DPGMM.png")  # Save plot
plt.show()

# ======================= Power-Law Inference =======================
# Define power-law models
def pl_mass(x, a, xmin, xmax):
    return (x ** -a) * (1 - a) / (xmax ** (1 - a) - xmin ** (1 - a))

def pl_q(x, a):
    return (x ** -a) * (1 - a)

def par_model(s, a, b):
    return pl_mass(s[:, 0], a, mmin, mmax) * pl_q(s[:, 1], b)

class par_dist:
    def __init__(self, pars):
        self.pars = pars
    def pdf(self, x):
        return par_model(np.atleast_2d(x), *self.pars)

# Define log-likelihood function
def LogLikelihood(cube, ss, n_ss):
    norm = np.mean(par_model(n_ss, cube[0], cube[1])) * volume
    single_mean = par_model(ss, cube[0], cube[1]) / norm 
    return np.sum(np.log(single_mean[np.isfinite(single_mean)]))

# Define priors
bound = np.array([[0, 5], [-5, 5]])  # Parameter bounds
def LogPrior(cube):
    if any(cube[i] <= bound[i, 0] or cube[i] >= bound[i, 1] for i in range(cube.size)):
        return -np.inf
    return np.log(1.0 / np.prod(bound[:, 1] - bound[:, 0]))

def LogProb(cube, ss, n_ss):
    prior = LogPrior(cube)
    return prior + LogLikelihood(cube, ss, n_ss) if np.isfinite(prior) else -np.inf

# MCMC Sampling
npoints = 5000
nwalkers, ndim = 50, 2
p0 = np.random.uniform(bound[:, 0], bound[:, 1], size=(nwalkers, ndim))
sampler = emcee.EnsembleSampler(nwalkers, ndim, LogProb, args=[samples, norm_samples])
sampler.run_mcmc(p0, npoints, progress=True)
flat_samples = sampler.get_chain(flat=True, discard=200, thin=25)

# Plot corner plot
fig = cc(flat_samples[-1000:], labels=[r'$\alpha$', r'$\beta$'])
fig.savefig(out_folder + "power_law_corner.png")  # Save plot
plt.show()

# Save MCMC samples
np.savez(out_folder + "pl_mcmc_m1zq.npz", flat_samples=flat_samples[-100:])


# Define bounds and compute volume
z_bds = [universe.z_min, universe.z_max]
m1_bds = [universe.m_min * (1 + z_bds[0]), universe.m_max * (1 + z_bds[1])]
q_bds = [0.1, 1]
bounds = np.array([m1_bds, q_bds])

n_pts  = np.array([200,200])
m1 = np.linspace(m1_bds[0], m1_bds[1], n_pts[0])
q = np.linspace(q_bds[0], q_bds[1], n_pts[1])
grid = np.array([[m1i, qi] for m1i in m1 for qi in q])
dgrid = [m1[1]-m1[0], q[1]-q[0]]

# Compute 2D probability distributions and save
prob = [par_dist(p).pdf(grid).reshape(n_pts) for p in flat_samples[-100:]]
dists = [a / np.sum(a * np.prod(dgrid)) for a in prob]

np.savez(out_folder + '/pl_mcmc_m1zq.npz', flat_samples=flat_samples[-100:])

# Save interpolators
out_name = 'powerlaw_m1zq'
interpolators = [funInterpolator((m1, q), d) for d in dists]
with open(out_folder + out_name + '.pkl', 'wb') as f:
    dill.dump(interpolators, f)


prob_1d = [np.sum(p, axis=1)*dgrid[1] for p in prob]
dists = [a / np.sum(a*dgrid[0]) for a in prob_1d]
    
out_name = 'powerlaw_m1z'
interpolators = [funInterpolator((m1,), d) for d in dists]
with open(out_folder+out_name+'.pkl', 'wb') as f:
    dill.dump(interpolators, f)
    
# Compute and print power-law indices
x = np.median(flat_samples[:, 0])
dx = flat_samples[:, 0].std()
print(f"Power-law index for m1^z = {x:.2f} ± {dx:.2f}")

x = np.median(flat_samples[:, 1])
dx = flat_samples[:, 1].std()
print(f"Power-law index for q = {x:.2f} ± {dx:.2f}")


# ======================= Uniform Model =======================
class UniDist:
    def __init__(self, vol):
        self.Vol = vol
    def pdf(self, x):
        return np.ones(x.shape[0]) / self.Vol

# Save uniform distributions
m1_bds = [universe.m_min * (1 + universe.z_min), universe.m_max * (1 + universe.z_max)]
q_bds = [0.1, 1]
vol = (m1_bds[1] - m1_bds[0]) * (q_bds[1] - q_bds[0])
uni_pop = UniDist(vol=vol)
with open(out_folder+'uni_m1zq.pkl', 'wb') as f:
    dill.dump(uni_pop, f)

vol = m1_bds[1] - m1_bds[0]
uni_pop = UniDist(vol=vol)
with open(out_folder+'uni_m1z.pkl', 'wb') as f:
    dill.dump(uni_pop, f)

vol = 120 - 5
uni_pop = UniDist(vol=vol)
with open(out_folder+'uni_m1z_120.pkl', 'wb') as f:
    dill.dump(uni_pop, f)

vol = 300 - 5
uni_pop = UniDist(vol=vol)
with open(out_folder+'uni_m1z_230.pkl', 'wb') as f:
    dill.dump(uni_pop, f)

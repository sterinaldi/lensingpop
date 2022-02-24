import numpy as np
import matplotlib.pyplot as plt
import dill

from tqdm import tqdm
from pathlib import Path

from figaro.mixture import DPGMM, HDPGMM

file = '/Users/stefanorinaldi/Documents/Repo/lensingpop/Mock_Data/m1m2posterior_PPD_no_selectionbias1000.npz'
out_folder = Path('/Users/stefanorinaldi/Documents/Repo/lensingpop/output')

data = np.load(Path(file))
m1_posteriors = data['m1_posterior']
n_draws_se = 10
n_draws_hier = 10
Mmin = 4
Mmax = 90

mixture = DPGMM([[Mmin, Mmax]])

M = np.linspace(Mmin*1.01, Mmax*0.99, 1000)
dM = M[1]-M[0]

posteriors = []

for i in tqdm(range(len(m1_posteriors[:5])), desc = 'Events'):
    draws = []
    post  = m1_posteriors[i]
    
    for _ in tqdm(range(n_draws_se), desc = str(i)):
        np.random.shuffle(post)
        mixture.density_from_samples(post)
        draws.append(mixture.build_mixture())
        mixture.initialise()
    
    draws = np.array(draws)
    posteriors.append(draws)

    ss = np.array([d.evaluate_mixture(np.atleast_2d(M).T) for d in draws])
    percentiles = [50, 5, 16, 84, 95]
    p = {}

    for perc in percentiles:
        p[perc] = np.percentile(ss.T, perc, axis = 1)

    norm = p[50].sum()*dM

    for perc in percentiles:
        p[perc] = p[perc]/norm
        
    fig, ax = plt.subplots()
    ax.hist(post, bins = int(np.sqrt(len(post))), histtype = 'step', density = True, stacked = True)
    ax.fill_between(M, p[95], p[5], color = 'mediumturquoise', alpha = 0.5)
    ax.fill_between(M, p[84], p[16], color = 'darkturquoise', alpha = 0.5)
    ax.plot(M, p[50], lw = 0.5, color = 'steelblue', label = '$\mathrm{DPGMM}$')
    ax.set_xlabel('$M$')
    ax.set_ylabel('$p(M)$')
    ax.grid(True,dashes=(1,3))
    ax.legend(loc = 0, frameon = False)
    fig.savefig(Path(out_folder, '{0}.pdf'.format(i+1)), bbox_inches = 'tight')
    plt.close()

posteriors = np.array(posteriors)

mixture = HDPGMM([[Mmin, Mmax]])
draws = []

for _ in tqdm(range(n_draws_hier), desc = 'Hierarchical'):
    np.random.shuffle(posteriors)
    mixture.density_from_samples(posteriors)
    draws.append(mixture.build_mixture())
    mixture.initialise()

draws = np.array(draws)

ss = np.array([d.evaluate_mixture(np.atleast_2d(M).T) for d in draws])
percentiles = [50, 5, 16, 84, 95]
p = {}

for perc in percentiles:
    p[perc] = np.percentile(ss.T, perc, axis = 1)

norm = p[50].sum()*dM

for perc in percentiles:
    p[perc] = p[perc]/norm
    
fig, ax = plt.subplots()
#ax.hist(post, bins = int(np.sqrt(len(post))), histtype = 'step', density = True, stacked = True)
ax.fill_between(M, p[95], p[5], color = 'mediumturquoise', alpha = 0.5)
ax.fill_between(M, p[84], p[16], color = 'darkturquoise', alpha = 0.5)
ax.plot(M, p[50], lw = 0.5, color = 'steelblue', label = '$\mathrm{DPGMM}$')
ax.set_xlabel('$M$')
ax.set_ylabel('$p(M)$')
ax.grid(True,dashes=(1,3))
ax.legend(loc = 0, frameon = False)
fig.savefig(Path(out_folder, 'mass_function.pdf'), bbox_inches = 'tight')
plt.close()

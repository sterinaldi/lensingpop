import numpy as np
from matplotlib import rcParams
from matplotlib import axes
import matplotlib.pyplot as plt
from figaro.utils import get_priors, plot_median_cr, plot_multidim, recursive_grid
from figaro.credible_regions import ConfidenceArea
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid.inset_locator import inset_axes  
import dill
import copy
from tqdm import tqdm
from figaro.marginal import marginalise as marg_figaro
from figaro.plot import plot_median_cr
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use('./plotrc.mplstyle')
import numpy as np
import matplotlib as mpl
from cycler import cycler

mpl.rcParams['lines.linewidth'] = 2
#mpl.rcParams['lines.linestyle'] = '--'

mpl.rcParams['xtick.major.width'] = 0.6  # default 0.8
mpl.rcParams['ytick.major.width'] = 0.6  # default 0.8
mpl.rcParams['axes.linewidth'] = 0.6  # default 0.8 
mpl.rcParams['lines.linewidth'] = 0.6  # default 1.5 
mpl.rcParams['lines.markeredgewidth'] = 0.6  # default 1
# The magic sauce
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = ''.join([r'\usepackage[T1]{fontenc}'
                                           r'\usepackage{cmbright}'])

math_blu = '$\log \mathcal{B}_{\mathcal{H}_{\mathrm{U}}}^{\mathcal{H}_{\mathrm{L}}}$'
"""
def marginalise(draws, dims, dgrid):
    marg = draws
    marg = np.array([m.sum(axis = tuple(dims))*np.prod([dgrid[k] for k in dims]) for m in marg])
    return marg
"""
def plot_multidim(axs, draws, dim, sample_dist = None, labels = None, units = None, 
                  true_value = None, figsize = 7, levels = [0.5, 0.68, 0.9],
                  real_dist = False, PL_dist = False, uni_dist = False, HDPGMM_model = False, plt_lim = None):

    rec_label = '\mathrm{(H)DPGMM}'

    if labels is None:
        labels = ['$x_{0}$'.format(i+1) for i in range(dim)]
    else:
        labels = ['${0}$'.format(l) for l in labels]
    
    if units is not None:
        labels = [l[:-1]+'\ [{0}]$'.format(u) if not u == '' else l for l, u in zip(labels, units)]
    column = 0
    ax = axs 
    levels = np.atleast_1d(levels)
    z_bds  = [0.01,1.3]
    lim = [15*(1+z_bds[0]), 98*(1+z_bds[1])]
    if plt_lim is None:
        plt_lim = lim
    #lim = [1.52,150]
    if HDPGMM_model: 

        n_pts  = np.array([75,75,75])
        q_bds  = [0.2, 1.]
        q  = np.linspace(q_bds[0], q_bds[1], n_pts[1]+2)[1:-1]
        m = np.linspace(lim[0], lim[1], n_pts[0]+2)[1:-1]
        dm = m[1]-m[0]
        chieff = np.linspace(-1.,1., n_pts[2]+2)[1:-1]

        dgrid = [m[1]-m[0], m[1]-m[0], chieff[1]-chieff[0]]

        dims = list(np.arange(dim, dtype = int))
        dims.remove(column)
        grid  = np.zeros(shape = (np.prod(n_pts), 3))
        kk=0
        for i, m1i in enumerate(m):
            for j, qi in enumerate(q):
                    for l, xi in enumerate(chieff):
                        grid[i*(n_pts[1]*n_pts[2]) + j*n_pts[2] + l] = [m1i, m1i*qi, xi]
 
        astro_dists = np.array([(d.pdf(grid).reshape(n_pts)) for d in tqdm(draws, total = len(draws), desc = 'Astro Dists')])
        
        probs = np.array([m.sum(axis = tuple(dims))*np.prod([dgrid[k] for k in dims]) for m in astro_dists])
        # Credible regions

        x = np.linspace(lim[0], lim[1], n_pts[column]+2)[1:-1]
        dx = x[1]-x[0]  
        percentiles = [50, 5, 16, 84, 95]
        p = {}
        for perc in percentiles:
            p[perc] = np.percentile(probs, perc, axis = 0)
        
        norm = p[50].sum()*dx
        for perc in percentiles:
            p[perc] = p[perc]/norm
        # CR
        ax.fill_between(x, p[95], p[5], color = 'mediumturquoise', alpha = 0.5)
        ax.fill_between(x, p[84], p[16], color = 'darkturquoise', alpha = 0.5)
        ax.plot(x, p[50], lw = 0.7, color = 'steelblue', label = r'$\mathrm{(H)DPGMM}$')
    data= np.load('/Users/damon/Desktop/lensingpop/Mock_Data/PowerlawplusPeakplusDelta30000Samples.npz')

#data= np.load('/Users/damon/Desktop/lensingpop/Mock_Data/m1m2zxeffxp_posterior_PPD_afterSelection_104_lensed.npz')
    m1 = data['m1']
    
    z = data['redshift']
    ax.hist(m1*(1+z),bins=50,density=True,histtype='step',label=r'$\mathrm{observation}$')
    del m1, z
    m = np.linspace(lim[0], lim[1], 80)
    dm = m[1]-m[0]
    m1z_grid, m2z_grid = np.meshgrid(m,m,indexing='ij')
    
    f_m = sample_dist[0](np.array([m1z_grid.flatten(), m2z_grid.flatten()]).T).reshape(m1z_grid.shape)
    f_m[np.isnan(f_m)] =0
    f_m = np.sum(f_m, axis=1)*dm
    norm = np.sum(f_m)*dm
    ax.plot(m, f_m/norm, lw = 1.5, color='black', label=r'$\mathrm{Real dist}$')
    
    if PL_dist:
        f_m = sample_dist[1](np.array([m1z_grid.flatten(), m2z_grid.flatten()]).T).reshape(m1z_grid.shape)
        f_m[np.isnan(f_m)] =0
        f_m = np.sum(f_m, axis=1)*dm
        norm = np.sum(f_m)*dm
        ax.plot(m, f_m/norm, lw = 1.5, color='orange', label=r'$\mathrm{PL prior}$')
    elif uni_dist:
        f_m = np.repeat(lim[0]*lim[1],80**2).reshape(m1z_grid.shape)
        f_m[np.isnan(f_m)] =0
        f_m = np.sum(f_m, axis=1)*dm
        norm = np.sum(f_m)*dm
        ax.plot(m, f_m/norm, lw = 1.5, color='green', label=r'$\mathrm{uniform prior}$')
    ax.set_ylim(0,0.04)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.yaxis.set_major_locator(MaxNLocator(3)) 

    ax.set_ylabel(r'$p(\theta)$',fontsize=18)
    #if labels is not None:
        #ax.set_xlabel(r'$m_1^z [M_{\odot}]$')
    ticks = np.linspace(lim[0], lim[1], 5)
    ax.set_xticks(ticks)
    [l.set_rotation(45) for l in ax.get_xticklabels()]
    ax.set_xlim(plt_lim[0],plt_lim[1])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend()
        

def plotting_two(data1,data2, pop_model, true_samples, wrong_samples,
                 m1, xlabel,real_dist=False,PL_dist=False,
                 uni_dist=False,HDPGMM_model=False,xlim=None, ylim=None):
    
    fig, axes = plt.subplots(nrows=2, sharex=True,
                             figsize=(6,8),gridspec_kw={'height_ratios': [1, 3]})
    fig.subplots_adjust(0,0,1,1,0,0)
    
    plt.errorbar(m1,data1[:,0], data1[:,1],solid_capstyle='projecting',color='black',
                 capsize=4,fmt='o',label=r'$\mathrm{True dist}$')
    if PL_dist:
        plt.errorbar(m1,data2[:,0], data2[:,1],solid_capstyle='projecting',color='orange',
                     capsize=4,fmt='o',label=r'$\mathrm{PL prior}$')
    elif uni_dist:
        plt.errorbar(m1,data2[:,0], data2[:,1],solid_capstyle='projecting',color='green',
                     capsize=4,fmt='o',label=r'$\mathrm{uniform prior}$')
    elif HDPGMM_model:
        plt.errorbar(m1,data2[:,0], data2[:,1],solid_capstyle='projecting',color='darkturquoise',
                     capsize=4,fmt='o',label=r'$\mathrm{(H)DPGMM}$')

    plot_multidim(axes[0], pop_model, 3, sample_dist = (true_samples, wrong_samples),
                  real_dist=real_dist,PL_dist=PL_dist,uni_dist=uni_dist,HDPGMM_model=HDPGMM_model,
                 plt_lim=xlim)
    #plt.yscale('log')
    axes[1].set_ylabel(r''+xlabel,fontsize=20)
    axes[1].set_xlabel(r'$m_1^z [M_{\odot}]$',fontsize=20)
    axes[1].set_ylim(ylim[0],ylim[1])
    axes[1].set_xlim(xlim[0],xlim[1])
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(loc=4)

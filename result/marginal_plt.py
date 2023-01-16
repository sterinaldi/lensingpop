import numpy as np
import matplotlib.pyplot as plt
from figaro.utils import get_priors, plot_median_cr, plot_multidim, recursive_grid
from matplotlib.ticker import (MaxNLocator, MultipleLocator, AutoMinorLocator)
from tqdm import tqdm
plt.style.use('./plotrc.mplstyle')
import matplotlib as mpl
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

def plot_multidim(ax, draws, dim, sample_dist = None, label = None, color = 'blue', 
                  figsize = 7, levels = [0.5, 0.68, 0.9], n_pts  = np.array([100,100,70]), 
                  xlim = None,ticks = [20,50,80,110,140]):

    if label is None:
        label = ['$x_{0}$'.format(i+1) for i in range(dim)]

        
    column = 0
    levels = np.atleast_1d(levels)
    z_bds  = [0.2,1.3]
    mlim = [15*(1+z_bds[0]), 98*(1+z_bds[1])]
    if xlim is None:
        xlim = mlim
    
    q_bds  = [0.2, 1.]
    q  = np.linspace(q_bds[0], q_bds[1], n_pts[1]+2)[1:-1]
    m = np.linspace(mlim[0], mlim[1], n_pts[0]+2)[1:-1]
    dm = m[1]-m[0]
    chieff = np.linspace(-1.,1., n_pts[2]+2)[1:-1]
    dgrid = [m[1]-m[0], q[1]-q[0], chieff[1]-chieff[0]]
    dims = list(np.arange(dim, dtype = int))
    dims.remove(column)
    grid  = np.zeros(shape = (np.prod(n_pts), 3))

    for i, m1i in enumerate(m):
        for j, qi in enumerate(q):
                for l, xi in enumerate(chieff):
                    grid[i*(n_pts[1]*n_pts[2]) + j*n_pts[2] + l] = [m1i, qi, xi]
                    
    #Benchmark plot                
    probs = sample_dist(grid).reshape(n_pts)
    #probs[np.isnan(probs)] = 0
    probs = np.array([probs.sum(axis = tuple(dims))*np.prod([dgrid[k] for k in dims])]).reshape(m.shape)
    norm = probs.sum()*dgrid[0]
    ax.plot(m, probs/norm, lw = 1.5, color='black', label=r'$\mathrm{Benchmark}$')
    if np.iterable(draws): 
        # grid
        np.array([75,75,70])
        q  = np.linspace(q_bds[0], q_bds[1], n_pts[1]+2)[1:-1]
        m = np.linspace(mlim[0], mlim[1], n_pts[0]+2)[1:-1]
        dm = m[1]-m[0]
        chieff = np.linspace(-1.,1., n_pts[2]+2)[1:-1]
        dgrid = [m[1]-m[0], q[1]-q[0], chieff[1]-chieff[0]]
        grid  = np.zeros(shape = (np.prod(n_pts), 3))

        for i, m1i in enumerate(m):
            for j, qi in enumerate(q):
                    for l, xi in enumerate(chieff):
                        grid[i*(n_pts[1]*n_pts[2]) + j*n_pts[2] + l] = [m1i, qi, xi]

        # HDPGMM models
        astro_dists = np.array([(d.pdf(grid).reshape(n_pts)) for d in tqdm(draws, total = len(draws), desc = 'dist')])
        probs = np.array([m.sum(axis = tuple(dims))*np.prod([dgrid[k] for k in dims]) for m in astro_dists])
        # Credible regions
        percentiles = [50, 5, 16, 84, 95]
        p = {}
        for perc in percentiles:
            p[perc] = np.percentile(probs, perc, axis = 0)

        norm = p[50].sum()*dgrid[column]
        for perc in percentiles:
            p[perc] = p[perc]/norm
        # CR
        ax.fill_between(m, p[95], p[5], color = 'mediumturquoise', alpha = 0.5)
        ax.fill_between(m, p[84], p[16], color = 'darkturquoise', alpha = 0.5)
        ax.plot(m, p[50], lw = 0.7, color = color, label=label)
    else:
        # Uniform distribution
        probs = np.ones(n_pts[0]) 
        norm = (m[-1]-m[0])
        ax.plot(m, probs/norm, lw = 1.5, color=color, label=label)
        
    ax.set_ylim(0,0.04)
    ax.set_yticks([])
    ax.yaxis.set_major_locator(MaxNLocator(3)) 
    ax.set_ylabel(r'$p(\theta)$',fontsize=18)
    [l.set_rotation(45) for l in ax.get_xticklabels()]
    ax.set_xlim(xlim[0],xlim[1])
    ax.xaxis.set_minor_locator(MultipleLocator(5)) 
    ax.set_xticks(ticks)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend()
    
def plotting_two(m1,d1,d2, model, true_model, color, label, 
                  xlabel, xlim=None, ylim=None, yscale='log',ticks = [20,40,60,80,100,120,140,160,180,200,220,240]):
    
    fig, axes = plt.subplots(nrows=2, sharex=True,
                             figsize=(8,8),gridspec_kw={'height_ratios': [1, 3]})
    fig.subplots_adjust(0,0,1,1,0,0)
    
    # Benchmark data
    axes[1].errorbar(m1,d1[:,0], d1[:,1],solid_capstyle='projecting',color='black',
                 capsize=4,fmt='o',label=r'$\mathrm{Benchmark}$')
    
    # Model data
    axes[1].errorbar(m1,d2[:,0], d2[:,1],solid_capstyle='projecting',color=color,
                 capsize=0,fmt='o',elinewidth=5,label=label)
    print('Nmodel=',len(model))
    plot_multidim(axes[0], model, 3, true_model, label=label, color=color, xlim=xlim,ticks=ticks)
    
    ax = axes[1]
    ax.set_ylabel(r''+xlabel,fontsize=20)
    ax.set_xlabel(r'$m_1^z [M_{\odot}]$',fontsize=20)
    ax.set_ylim(ylim[0],ylim[1])
    ax.xaxis.set_minor_locator(MultipleLocator(5)) 
    ax.set_xticks(ticks)
    ax.set_xlim(xlim[0],xlim[1])
    plt.yscale(yscale)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(loc=4)
    return fig, ax

    
    
def hdpgmm_plt(ax, draws, dim, figsize = 7, benchmark = None, levels = [0.5, 0.68, 0.9], plt_lim = None):
    column = 0 
    levels = np.atleast_1d(levels)
    z_bds  = [0.01,1.3]
    lim = [15*(1+z_bds[0]), 98*(1+z_bds[1])]
    if plt_lim is None:
        plt_lim = lim
    lim = plt_lim 
    n_pts  = np.array([75,75,75])
    q_bds  = [0.2, 1.]
    q  = np.linspace(q_bds[0], q_bds[1], n_pts[1]+2)[1:-1]
    m = np.linspace(lim[0], lim[1], n_pts[0]+2)[1:-1]
    dm = m[1]-m[0]
    chieff = np.linspace(-1.,1., n_pts[2]+2)[1:-1]
    dgrid = [m[1]-m[0], q[1]-q[0], chieff[1]-chieff[0]]

    dims = list(np.arange(dim, dtype = int))
    dims.remove(column)
    grid  = np.zeros(shape = (np.prod(n_pts), 3))

    for i, m1i in enumerate(m):
        for j, qi in enumerate(q):
                for l, xi in enumerate(chieff):
                    grid[i*(n_pts[1]*n_pts[2]) + j*n_pts[2] + l] = [m1i, m1i*qi, xi]
 
    astro_dists = np.array([((d.pdf(grid)*grid[:,0]).reshape(n_pts)) for d in tqdm(draws, total = len(draws), desc = 'Astro Dists')])
    probs = np.array([m.sum(axis = tuple(dims))*np.prod([dgrid[k] for k in dims]) for m in astro_dists])
    # Credible regions
    percentiles = [50, 5, 16, 84, 95]
    p = {}
    for perc in percentiles:
        p[perc] = np.percentile(probs, perc, axis = 0)

    norm = p[50].sum()*dgrid[0]
    for perc in percentiles:
        p[perc] = p[perc]/norm
    # CR
    ax.fill_between(m, p[95], p[5], color = 'mediumturquoise', alpha = 0.5)
    ax.fill_between(m, p[84], p[16], color = 'darkturquoise', alpha = 0.5)
    ax.plot(m, p[50], lw = 0.7, color = 'steelblue', label = r'$\mathrm{(H)DPGMM}$')

    probs = (benchmark(grid)*grid[:,0]).reshape(n_pts)
    probs[np.isnan(probs)] =0
    probs = np.array([probs.sum(axis = tuple(dims))*np.prod([dgrid[k] for k in dims])]).reshape(m.shape)

    norm = probs.sum()*dgrid[0]
    ax.plot(m, probs/norm, lw = 1.5, color='black', label='Benchmark')
    
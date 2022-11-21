import numpy as np
import matplotlib.pyplot as plt
import dill

from figaro.utils import recursive_grid
from figaro.credible_regions import ConfidenceArea

from pathlib import Path
from tqdm import tqdm
from corner import corner
from matplotlib import rcParams
from scipy.interpolate import RegularGridInterpolator

class AstroDist:
    def __init__(self, points, pdet):
        self.points = points
        self.pdet   = pdet
        self.I      = RegularGridInterpolator() # FIXME: sistemare
    
    def __call__(self, x):
        return pdf(x)
        
    def pdf(self, x):
        m1  = x[:,0]
        m2  = x[:,1]
        chi = x[:,2]
        return self.I(m1, m2/m1, chi)/m1
    
    

def marginalise(draws, dims, dgrid):
    marg = draws
    marg = np.array([m.sum(axis = tuple(dims))*np.prod([dgrid[k] for k in dims]) for m in marg])
    return marg

def plot_multidim(draws, dim, bounds, n_pts, dgrid, out_folder = '.', name = 'density', labels = None, units = None, show = False, save = True, subfolder = False, true_value = None, figsize = 7, levels = [0.5, 0.68, 0.9]):

    rec_label = '\mathrm{(H)DPGMM}'
    
    if labels is None:
        labels = ['$x_{0}$'.format(i+1) for i in range(dim)]
    else:
        labels = ['${0}$'.format(l) for l in labels]
    
    if units is not None:
        labels = [l[:-1]+'\ [{0}]$'.format(u) if not u == '' else l for l, u in zip(labels, units)]
    
    levels = np.atleast_1d(levels)
    
    K = dim
    factor = 2.0          # size of one side of one panel
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.1         # w/hspace size
    plotdim = factor * dim + factor * (K - 1.0) * whspace
    dim_plt = lbdim + plotdim + trdim
    
    fig, axs = plt.subplots(K, K, figsize=(figsize, figsize))
    # Format the figure.
    lb = lbdim / dim_plt
    tr = (lbdim + plotdim) / dim_plt
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace)
    
    # 1D plots (diagonal)
    for column in range(K):
        ax = axs[column, column]
        # Marginalise over all uninterested columns
        dims = list(np.arange(dim, dtype = int))
        dims.remove(column)
        marg_draws = marginalise(draws, dims, dgrid)
        # Credible regions
        lim = bounds[column]
        x = np.linspace(lim[0], lim[1], n_pts[column]+2)[1:-1]
        dx   = x[1]-x[0]
        
        probs = marg_draws
        
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
        ax.plot(x, p[50], lw = 0.7, color = 'steelblue')
        if column < K - 1:
            ax.set_xticks([])
            ax.set_yticks([])
        elif column == K - 1:
            ax.set_yticks([])
            if labels is not None:
                ax.set_xlabel(labels[-1])
            ticks = np.linspace(lim[0], lim[1], 5)
            ax.set_xticks(ticks)
            [l.set_rotation(45) for l in ax.get_xticklabels()]
        if column < 2:
            ax.set_xlim(lim[0], lim[1])
        else:
            ax.set_xlim(lim[0], 1.)
    
    # 2D plots (off-diagonal)
    for row in range(K):
        for column in range(K):
            ax = axs[row,column]
            ax.grid(visible=False)
            if column > row:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif column == row:
                continue
            
            # Marginalise
            dims = list(np.arange(dim, dtype = int))
            dims.remove(column)
            dims.remove(row)
            marg_draws = marginalise(draws, dims, dgrid)
            
            # Credible regions
            lim = np.array(bounds[[row, column]])
            grid, i_dgrid = recursive_grid(lim[::-1], np.ones(2, dtype = int)*np.array([int(n_pts[column]), int(n_pts[row])], dtype = int))
            
            x = np.linspace(lim[0,0], lim[0,1], n_pts[row]+2)[1:-1]
            y = np.linspace(lim[1,0], lim[1,1], n_pts[column]+2)[1:-1]
            
            dd = marg_draws
            median = np.percentile(dd, 50, axis = 0)
            median = median/(median.sum()*np.prod(i_dgrid))
            median = median.reshape(n_pts[column], n_pts[row])
            
            X,Y = np.meshgrid(x,y)
            with np.errstate(divide = 'ignore'):
                logmedian = np.nan_to_num(np.log(median), nan = -np.inf, neginf = -np.inf)
            _,_,levs = ConfidenceArea(logmedian, x, y, adLevels=levels)
            ax.contourf(Y, X, np.exp(logmedian), cmap = 'Blues', levels = 100)
            try:
                c1 = ax.contour(Y, X, logmedian, np.sort(levs), colors='k', linewidths=0.3)
                if rcParams["text.usetex"] == True:
                    ax.clabel(c1, fmt = {l:'{0:.0f}\\%'.format(100*s) for l,s in zip(c1.levels, np.sort(levels)[::-1])}, fontsize = 3)
                else:
                    ax.clabel(c1, fmt = {l:'{0:.0f}\%'.format(100*s) for l,s in zip(c1.levels, np.sort(levels)[::-1])}, fontsize = 3)
            except ValueError:
                pass
            ax.set_xticks([])
            ax.set_yticks([])
            
            if column == 0:
                ax.set_ylabel(labels[row])
                ticks = np.linspace(lim[0,0], lim[0,1], 5)
                ax.set_yticks(ticks)
                [l.set_rotation(45) for l in ax.get_yticklabels()]
            if row == K - 1:
                ticks = np.linspace(lim[1,0], lim[1,1], 5)
                ax.set_xticks(ticks)
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                ax.set_xlabel(labels[column])
                
            elif row < K - 1:
                ax.set_xticks([])
            elif column == 0:
                ax.set_ylabel(labels[row])
            if row == 2:
                ax.set_ylim(0,1)
    
    if not subfolder:
        fig.savefig(Path(out_folder, '{0}.pdf'.format(name)), bbox_inches = 'tight')
    else:
        if not Path(out_folder, 'density').exists():
            try:
                Path(out_folder, 'density').mkdir()
            except FileExistsError:
                pass
        fig.savefig(Path(out_folder, 'density', '{0}.pdf'.format(name)), bbox_inches = 'tight')
    plt.close()


if __name__ == '__main__':
    
    # Edit here
    
    postprocess = False
    out_name = 'astro_m1m2chieff'
    
    draws_file   = Path('posteriors_hier.pkl') # Change for specific paths
    selfunc_file = Path('selfunc_m1m2z_source.pkl')
    n_pts  = np.array([50,50,100,30])
    m1_bds = [5.,120]#240.]
    q_bds  = [5./240.,1.]#[5.,240.]
    z_bds  = [0.01,1.9]
    X_bds  = [-1.,1.]
    bounds    = np.array([m1_bds, q_bds, z_bds, X_bds]) # Please check that these bounds are
    bounds_3d = np.array([m1_bds, q_bds, X_bds])
    
    m1 = np.linspace(m1_bds[0], m1_bds[1], n_pts[0]+2)[1:-1]
    q  = np.linspace(q_bds[0], q_bds[1], n_pts[1]+2)[1:-1]
    z  = np.linspace(z_bds[0], z_bds[1], n_pts[2]+2)[1:-1]
    chieff = np.linspace(X_bds[0], X_bds[1], n_pts[3]+2)[1:-1]
    
    dgrid = [m1[1]-m1[0], [], z[1]-z[0], chieff[1]-chieff[0]]

    with open(draws_file, 'rb') as f:
        draws = dill.load(f)
    
    with open(selfunc_file, 'rb') as f:
        selfunc = dill.load(f)

    labels = ['M_1', 'q', '\\chi_{\\mathrm{eff}}']
    units  = ['M_{\\odot}', '', '']
    
    dn_m1 = np.prod(n_pts[1:])
    dn_q  = np.prod(n_pts[2:])
    dn_z  = np.prod(n_pts[3:])
    dn_X  = n_pts[-1]
    grid    = np.zeros(shape = (np.prod(n_pts), 4))
    grid_3d = np.zeros(shape = (np.prod(np.delete(n_pts, 2)), 3))
    for i, m1i in tqdm(enumerate(m1), desc = 'Grid', total = n_pts[0]):
        for j, qi in enumerate(q):
            for k, zi in enumerate(z):
                for l, xi in enumerate(chieff):
                    grid[i*dn_m1 + j*dn_q + k*dn_z + l] = [m1i, qi*m1i, zi, xi]
                    grid_3d[i*(n_pts[1]*n_pts[3]) + j*n_pts[3] + l] = [m1i, qi, xi]
    
    dgrid[1] = 1.
    print('Evaluating pdet...')
    m1_g = grid[:,0]
    m2_g = grid[:,1]
    z_g  = grid[:,2]
    det_jacobian = (1+z_g)/m1_g
    pdet = (selfunc((m1_g/(1+z_g), m2_g/(1+z_g), z_g))*det_jacobian).reshape(n_pts)
    pdet[np.where(pdet == 0.)] = np.inf
    '''
    To do: move to griddata (https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html) and take points in m2 between 0 and m1
    '''
    astro_dists   = np.array([np.sum((d.pdf(grid).reshape(n_pts) / pdet) * dgrid[2], axis = 2).reshape(n_pts[0]*n_pts[1]*n_pts[3]) for d in tqdm(draws, total = len(draws), desc = 'Astro Dists')])
    interpolators = [AstroDist(grisd_3d, d) for d in astro_dists]
    with open(out_name+'.pkl', 'wb') as f:
        dill.dump(interpolators, f)
    astro_dists = [d.reshape(n_pts[0], n_pts[1], n_pts[3]) for d in astro_dists]
    _ = dgrid.pop(2)
    print('Making plot...')
    
    n_spls = int(1e5)
    pmax = np.max(astro_dists[0])
#
#    samples = np.random.uniform(low = bounds_3d[:,0], high = bounds_3d[:,1], size = (n_spls, 3))
#    refs = interpolators[0].pdf(samples)
#    vals = np.random.uniform(low = 0, high = pmax, size = n_spls)
#    samples = samples[np.where(vals < refs)]
#    c = corner(samples)
#    c.savefig('crnr.pdf', bbox_inches = 'tight')
    plot_multidim(astro_dists, 3, bounds_3d, np.delete(n_pts, 2), dgrid, out_folder = '.', name = out_name, labels = labels, units = units)
##    plot_multidim([pdet], 3, bounds_3d, n_pts[:-1], dgrid, out_folder = '.', name = out_name, labels = labels, units = units)

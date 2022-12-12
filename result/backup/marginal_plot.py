import numpy as np
from matplotlib import rcParams
from matplotlib import axes
import matplotlib.pyplot as plt
from figaro.utils import get_priors, plot_median_cr, plot_multidim
from figaro.marginal import marginalise
from figaro.credible_regions import ConfidenceArea
from mpl_toolkits.axes_grid.inset_locator import inset_axes  

math_blu = '$\log \mathcal{B}_{\mathcal{H}_{\mathrm{U}}}^{\mathcal{H}_{\mathrm{L}}}$'

def plotting_two(data1,data2, pop_model, true_samples, wrong_samples, m1, xlim, ylim,xlabel):
    print('hi')
    fig = plt.figure(figsize=(20, 14),facecolor='white')
    ax = fig.add_subplot(121)
    # the main axes is subplot(111) by default
    plt.errorbar(m1,data1[:,0], data1[:,1],solid_capstyle='projecting',color='black',capsize=4,fmt='o',label='True prior (non-lensing)')
    plt.errorbar(m1,data2[:,0], data2[:,1],solid_capstyle='projecting',color='orange',capsize=4,fmt='o',label='power-law')
    #plt.yscale('log')
    plt.ylabel(r''+xlabel,fontsize=20)
    plt.xlabel(r'$m_1^z [M_{\odot}]$',fontsize=20)
    plt.ylim(ylim[0],ylim[1])
    plt.xlim(xlim[0],xlim[1])
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(loc=4)
    # this is an inset axes over the main axes
    iax = inset_axes(ax, 
                        width="98%", # width = 30% of parent_bbox
                        height=2.0, # height : 1 inch
                        loc=1)
    plot_marginal(iax,pop_model, sample_dist = (true_samples, wrong_samples), labels = ["M_1","M_2"], units = ["M_{\\odot}","M_{\\odot}"], hierarchical = True)



def plot_marginal(axs, draws, sample_dist = None, labels = None, units = None, hierarchical = False, n_pts = 200, true_value = None, figsize = 7, levels = [0.5, 0.68, 0.9]):
    """
    Plot the recovered multidimensional distribution along with samples from the true distribution (if available) as corner plot.
    
    Arguments:
        :iterable draws:         container for mixture instances
        :int dim:                number of dimensions
        :np.ndarray samples:     samples from the true distribution (if available)
        :str or Path out_folder: output folder
        :str name:               name to be given to outputs
        :list-of-str labels:     LaTeX-style quantity label, for plotting purposes
        :list-of-str units:      LaTeX-style quantity unit, for plotting purposes
        :bool hierarchical:      hierarchical inference, for plotting purposes
        :bool save:              whether to save the plot or not
        :bool show:              whether to show the plot during the run or not
        :bool subfolder:         whether to save in a dedicated subfolder
        :int n_pts:              number of grid points (same for each dimension)
        :iterable true_value:    true value to plot
        :double figsize:         figure size (matplotlib)
        :iterable levels:        credible levels to plot
    """
    
    dim = draws[0].dim
    
    if hierarchical:
        rec_label = '\mathrm{(H)DPGMM}'
    else:
        rec_label = '\mathrm{DPGMM}'
    
    if labels is None:
        labels = ['$x_{0}$'.format(i+1) for i in range(dim)]
    else:
        labels = ['${0}$'.format(l) for l in labels]
    
    if units is not None:
        labels = [l[:-1]+'\ [{0}]$'.format(u) if not u == '' else l for l, u in zip(labels, units)]
    
    levels = np.atleast_1d(levels)

    all_bounds = np.atleast_2d([d.bounds for d in draws])
    x_min = np.min(all_bounds, axis = -1).max(axis = 0)
    #x_max = np.max(all_bounds, axis = -1).min(axis = 0)
    x_max = (200,200)
    bounds = np.array([x_min, x_max]).T
    K = dim
    factor = 2.0          # size of one side of one panel
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.1         # w/hspace size
    plotdim = factor * dim + factor * (K - 1.0) * whspace
    dim_plt = lbdim + plotdim + trdim
    
    # Format the figure.
    lb = lbdim / dim_plt
    tr = (lbdim + plotdim) / dim_plt

    # 1D plots (diagonal)
    for column in range(1):
        ax = axs
        # Marginalise over all uninterested columns
        dims = list(np.arange(dim))
        dims.remove(column)
        marg_draws = marginalise(draws, dims)
        # Credible regions
        lim = bounds[column]
        x = np.linspace(lim[0], lim[1], n_pts+2)[1:-1]
        dx   = x[1]-x[0]
        
        probs = np.array([d.pdf(x) for d in marg_draws])
        
        percentiles = [50, 5, 16, 84, 95]
        p = {}
        for perc in percentiles:
            p[perc] = np.percentile(probs, perc, axis = 0)
        norm = p[50].sum()*dx
        for perc in percentiles:
            p[perc] = p[perc]/norm
        
        # Samples (if available)
        if sample_dist is not None:
            if True:
                m  = np.linspace(0, 100*(1+2.3), 500)
                dm = m[1]-m[0]
                m1z_grid, m2z_grid = np.meshgrid(m,m,indexing='ij')
                f_m = sample_dist[0](np.array([m1z_grid.flatten(), m2z_grid.flatten()]).T)
                f_m = np.reshape(f_m, (500,500))
                f_m = np.sum(f_m, axis=1)*dm

                norm = np.sum(f_m)*dm
                plt.plot(m, f_m/norm, lw = 1.5, label = '$Real\ distribution$')

                f_m = sample_dist[1](np.array([m1z_grid.flatten(), m2z_grid.flatten()]).T)
                f_m = np.reshape(f_m, (500,500))
                f_m = np.sum(f_m, axis=1)*dm

                norm = np.sum(f_m)*dm
                plt.plot(m, f_m/norm, lw = 1.5, label = '$PL\ distribution$')
                #ax.set_xlim(0,100)
            else:
                samples = sample_dist.rvs(3000)
                ax.hist(samples[:,column], bins = int(np.sqrt(len(samples[:,column]))), histtype = 'step', density = True)
        # CR
        ax.fill_between(x, p[95], p[5], color = 'mediumturquoise', alpha = 0.5)
        ax.fill_between(x, p[84], p[16], color = 'darkturquoise', alpha = 0.5)
        if true_value is not None:
            if true_value[column] is not None:
                ax.axvline(true_value[column], c = 'orangered', lw = 0.5)
        ax.plot(x, p[50], lw = 0.7, color = 'steelblue',label = '$HDPGMM$')
       
        #ax.set_xticks([],fontsize=10)
        ax.set_yticks([])

        ax.set_yticks([])
        #if labels is not None:
            #ax.set_xlabel(r'$m_1^z [M_{\odot}]$')
        ticks = np.linspace(lim[0], lim[1], 5)
        #ax.set_xticks(ticks)
        [l.set_rotation(45) for l in ax.get_xticklabels()]
        ax.set_xlim(lim[0], lim[1])

        ax.legend()


from mpl_toolkits.axes_grid.inset_locator import inset_axes
def plotting(ans, true_ans, pop_model, samples, m1, xlim, ylim):

    fig = plt.figure(figsize=(20, 14),facecolor='white')
    ax = fig.add_subplot(121)
    # the main axes is subplot(111) by default
    diff = (ans[:,0]-true_ans[:,0])
    err = np.sqrt(  ans[:,1]**2+true_ans[:,1]**2 ) 
    plt.errorbar(m1,diff,err,solid_capstyle='projecting',capsize=4,fmt='o')

    plt.ylabel(r''+math_blu,fontsize=20)
    plt.xlabel(r'$m_1^z [M_{\odot}]$',fontsize=20)
    #plt.ylim(ylim[0],ylim[1])
    plt.xlim(xlim[0],xlim[1])
    #plt.yscale('log')
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=18)
    # this is an inset axes over the main axes
    iax = inset_axes(ax, 
                        width="98%", # width = 30% of parent_bbox
                        height=2.0, # height : 1 inch
                        loc=1)
    plot_marginal(iax,pop_model, sample_dist = samples, labels = ["M_1","M_2"], units = ["M_{\\odot}","M_{\\odot}"], hierarchical = True)
from mpl_toolkits.axes_grid.inset_locator import inset_axes

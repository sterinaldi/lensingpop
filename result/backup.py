""" 
# HDPGMM 
#with open('./posteriors_hier.pkl', 'rb') as f:
#    pop_model=dill.load(f)



#pop_model = pop_model[:500]

#from figaro.marginal import marginalise
#pop_model = marginalise(pop_model)
#pop_model = marginalise(pop_model,axis=-2)
#with open('/Users/damon/Desktop/lensingpop/result/obs_pop_m1m2xeff.pkl', 'rb') as f:
#    pop_obs=dill.load(f)

class mix_pop():
    def __init__(self,file,peak=True,marginal_xeff=False):
        self.peak = peak
        self.marginal_xeff = marginal_xeff
        with open(file, 'rb') as f: self.pop_pdf = dill.load(f)
        self.spin_pop = Gaussian_spin_distribution(**spin_pars)
    def __call__(self, x):
        return self.pdf(x)
    
    def rvs(self, x):
        return 1
    
    def pdf(self, x):
        if self.marginal_xeff:
            return self.pop_pdf(x[:,:2]) *self.spin_pop.marginal_pxeff(x[:,-1])
        else:
            return self.pop_pdf(x[:,:2])  *self.spin_pop.pdf(x[:,-2],x[:,-1],x=self.peak)
"""   


"""


   
    def pdf(self, x):
        if self.marginal_xeff:
            return self.pop_pdf(x[:,:2]) *redshift_distribution(x[:,2]) *self.spin_pop.marginal_pxeff(x[:,3])
        else:
            return self.pop_pdf(x[:,:2]) *redshift_distribution(x[:,2]) *self.spin_pop.pdf(x[:,3],x[:,4],x=self.peak)
    
        

Here is the example of using this class.

# Importing packages:

import numpy as np
from blu import *
import dill 
from simulated_universe import *



# Inferred HDPGMM model from observations
with open('/Users/damon/Desktop/blu_upload/git_download/hier/test_output/posteriors_hier.pkl', 'rb') as f:
    pop_model=dill.load(f)
    
# Importing the stored theoretical pdf and make it usable (has a pdf function) for computing BLU.
class true_dist():
    def __init__(self):
        with open('./real_dist.pkl', 'rb') as f: self.pop_pdf = dill.load(f)
        
    def __call__(self, x):
        return self.pdf(x)
        
    def pdf(self, x):
        return self.pop_pdf(x)
        
# Wrong population model (power law)
with open('./pop_power.pkl', 'rb') as f:
    pop_power=dill.load(f)
    
# Uniform population model
with open('./pop_uniform.pkl', 'rb') as f:
    pop_uni=dill.load(f)

# Import the posterior of lensing pair (DPGMM model) 
with open('./img1_det.pkl', 'rb') as f:
    img1=dill.load(f)
with open('./img2_det.pkl', 'rb') as f:
    img2=dill.load(f)


pop_true=true_dist() #Initialize the theoretical pdf 


N = 1e4 # Number of Monte Carlo samples
OLU_true = OddsRatio(gw_pop=pop_true,Nmc=N) # BLU class by using the theoretical pdf 


blu_true = []

for i in range(len(img1)):
    print(i,'-th pair')
    blu_true.append(OLU_true.BayesFactor_PEuniform(img1[i],img2[i]))

def log_blu(data,error=True):
    
    if error:
        # for y = log(x)
        #    dy = dx/x 
        blu, error = data
        error /= blu 
        blu = np.log(blu)
        return (blu,error)
    else:
        return np.log(data)
    

true_ans = np.array(true_ans)



def plot_marginal(axs, draws, sample_dist = None,
                  labels = None, units = None, hierarchical = False, 
                  n_pts = 200, true_value = None, figsize = 7, levels = [0.5, 0.68, 0.9],
                 xlim=None, ylim=None,p1=False,p2=False,p3=False,obs=False):

    
    dim = draws[0].dim
    #dim =3
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
    x_max = np.max(all_bounds, axis = -1).min(axis = 0)
    x_min = (xlim[0],xlim[0])
    x_max = (xlim[1],xlim[1])
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
        ymax = 0
        if sample_dist is not None:
            if True:
                m  = np.linspace(5, 100*(1+2.3), 500)
                dm = m[1]-m[0]
                m1z_grid, m2z_grid = np.meshgrid(m,m,indexing='ij')
                
                
                if not obs:
                    f_m = sample_dist[0](np.array([m1z_grid.flatten(), m2z_grid.flatten()]).T)
                    f_m = np.reshape(f_m, m1z_grid.shape)
                    f_m[np.isnan(f_m)] = 0
                    f_m = np.sum(f_m, axis=1)*dm

                    norm = np.sum(f_m)*dm
                    ax.plot(m, f_m/norm, lw = 1.5, color='black', label = ' Real distribution')
                
                if obs:                    
                    f_m =sample_dist[1](np.array([m1z_grid.flatten(), m2z_grid.flatten()]).T)
                    f_m = np.reshape(f_m, m1z_grid.shape)
                    f_m[np.isnan(f_m)] =0
                    f_m = np.sum(f_m, axis=1)*dm

                    norm = np.sum(f_m)*dm
                    ax.plot(m, f_m/norm, lw = 1.5, color='black', label = ' Obs distribution')


                if p2: 
                    f_m = sample_dist[1](np.array([m1z_grid.flatten(), m2z_grid.flatten()]).T)
                    f_m = np.reshape(f_m, m1z_grid.shape)
                    f_m[np.isnan(f_m)] =0
                    f_m = np.sum(f_m, axis=1)*dm

                    norm = np.sum(f_m)*dm
                    ax.plot(m, f_m/norm, lw = 1.5, color='orange', label = ' PL distribution')
                if p3: 
                    f_m = np.repeat(100*(1+2.3)*100*(1+2.3),500**2)
                    f_m = np.reshape(f_m, m1z_grid.shape)
                    f_m[np.isnan(f_m)] =0
                    f_m = np.sum(f_m, axis=1)*dm

                    norm = np.sum(f_m)*dm
                    ax.plot(m, f_m/norm, lw = 1.5, color='green', label = ' uniform prior')
               # ax.set_xlim(0,200)
            else:
                samples = sample_dist.rvs(3000)
                ax.hist(samples[:,column], bins = int(np.sqrt(len(samples[:,column]))), histtype = 'step', density = True)
        # CR
        if p1:
            ax.fill_between(x, p[95], p[5], color = 'mediumturquoise', alpha = 0.5)
            ax.fill_between(x, p[84], p[16], color = 'darkturquoise', alpha = 0.5)
            if true_value is not None:
                if true_value[column] is not None:
                    ax.axvline(true_value[column], c = 'orangered', lw = 0.5)
            ax.plot(x, p[50], lw = 0.7, color = 'steelblue',label = 'HDPGMM')
       
        ax.set_xticks([])
        ax.set_yticks([])
        ax.yaxis.set_major_locator(MaxNLocator(3)) 
     
        ax.set_ylabel(r'$p(\theta)$',fontsize=18)
        #if labels is not None:
            #ax.set_xlabel(r'$m_1^z [M_{\odot}]$')
        ticks = np.linspace(lim[0], lim[1], 5)
        ax.set_xticks(ticks)
        [l.set_rotation(45) for l in ax.get_xticklabels()]
        #ax.set_xlim(lim[0], lim[1])
        ax.set_xlim(xlim[0],xlim[1])
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.legend()

"""

"""
def plotting_two(data1,data2, pop_model, true_samples, wrong_samples, m1, xlim, ylim,xlabel,p1=False,p2=False,p3=False,obs=False):
    fig, axes = plt.subplots(nrows=2, sharex=True,figsize=(6,8),gridspec_kw={'height_ratios': [1, 3]})
    fig.subplots_adjust(0,0,1,1,0,0)
    
    
    # the main axes is subplot(111) by default
    if not obs:
        plt.errorbar(m1,data1[:,0], data1[:,1],solid_capstyle='projecting',color='black',capsize=4,fmt='o',label='True prior')
    else:
        plt.errorbar(m1,data1[:,0], data1[:,1],solid_capstyle='projecting',color='black',capsize=4,fmt='o',label='obs prior')
    if p3: plt.errorbar(m1,data2[:,0], data2[:,1],solid_capstyle='projecting',color='green',capsize=4,fmt='o',label='uniform prior')
    if p2: plt.errorbar(m1,data2[:,0], data2[:,1],solid_capstyle='projecting',color='orange',capsize=4,fmt='o',label='power law')
    if p1: plt.errorbar(m1,data2[:,0], data2[:,1],solid_capstyle='projecting',color='darkturquoise',capsize=4,fmt='o',label='(H)DPGMM')
    #plt.yscale('log')
    axes[1].set_ylabel(r''+xlabel,fontsize=20)
    axes[1].set_xlabel(r'$m_1^z [M_{\odot}]$',fontsize=20)
    axes[1].set_ylim(ylim[0],ylim[1])
    axes[1].set_xlim(xlim[0],xlim[1])
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(loc=4)
    # this is an inset axes over the main axes
    #iax = inset_axes(axes[0], 
    #                    width="98%", # width = 30% of parent_bbox
    #                    height=2.0, # height : 1 inch
    #                    loc=1)
    plot_marginal(axes[0],pop_model, sample_dist = (true_samples, wrong_samples),
                  labels = ["M_1","M_2"], units = ["M_{\\odot}","M_{\\odot}"],
                  hierarchical = True, xlim = xlim, ylim = ylim,p1=p1,p2=p2,p3=p3,obs=obs)
    

def plotting_diff(data1,data2, pop_model, true_samples, wrong_samples, m1, xlim, ylim,xlabel,p1=False,p2=False,p3=False,obs=False):
    fig, axes = plt.subplots(nrows=2, sharex=True,figsize=(6,8),gridspec_kw={'height_ratios': [1, 3]})
    fig.subplots_adjust(0,0,1,1,0,0)
    
    d2 = data1.copy()
    d2[:,0] = data2[:,0] - data1[:,0]
    d2[:,1] = np.sqrt( data2[:,1]**2 + data1[:,1]**2  )
    
    if p3: plt.errorbar(m1,d2[:,0], d2[:,1],solid_capstyle='projecting',color='green',capsize=4,fmt='o',label='uniform - true')
    if p2: plt.errorbar(m1,d2[:,0], d2[:,1],solid_capstyle='projecting',color='orange',capsize=4,fmt='o',label='power law - true')
    if p1: plt.errorbar(m1,d2[:,0], d2[:,1],solid_capstyle='projecting',color='darkturquoise',capsize=4,fmt='o',label='(H)DPGMM - true')
    #plt.yscale('log')
    axes[1].set_ylabel(r'$\Delta$'+xlabel,fontsize=20)
    axes[1].set_xlabel(r'$m_1^z [M_{\odot}]$',fontsize=20)
    axes[1].set_ylim(ylim[0],ylim[1])
    axes[1].set_xlim(xlim[0],xlim[1])
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(loc=4)
    # this is an inset axes over the main axes
    #iax = inset_axes(axes[0], 
    #                    width="98%", # width = 30% of parent_bbox
    #                    height=2.0, # height : 1 inch
    #                    loc=1)
    plot_marginal(axes[0],pop_model, sample_dist = (true_samples, wrong_samples),
                  labels = ["M_1","M_2"], units = ["M_{\\odot}","M_{\\odot}"],
                  hierarchical = True, xlim = xlim, ylim = ylim,p1=p1,p2=p2,p3=p3,obs=obs)

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
    
"""
    
    
    
# Analysis 

"""

unblu_uni=[]
unblu_true=[]
unblu_hdpgmm=[]
unblu_pl=[]
unblu_obs=[]
N = 1e5
OLU = OddsRatio(gw_pop=pop_model,Nmc=N)
OLU_true = OddsRatio(gw_pop=pop_true,Nmc=N)
OLU_pl = OddsRatio(gw_pop=pop_power,Nmc=N)
OLU_uni = OddsRatio(gw_pop=pop_uni,Nmc=N)
OLU_obs = OddsRatio(gw_pop=pop_obs,Nmc=N)
for i in range(8):
    t1 = time.time()
    for j in range(i+1,8):
        
        unblu_true.append(OLU_true.BayesFactor_PEuniform(img1[i],img1[j]))
        unblu_hdpgmm.append(OLU.BayesFactor_PEuniform(img1[i],img1[j]))
        unblu_pl.append(OLU_pl.BayesFactor_PEuniform(img1[i],img1[j]))
        unblu_uni.append(OLU_uni.BayesFactor_PEuniform(img1[i],img1[j]))
        unblu_obs.append(OLU_obs.BayesFactor_PEuniform(img1[i],img1[j]))
    print(time.time()-t1)

unblu_true = np.array(unblu_true)
unblu_hdpgmm = np.array(unblu_hdpgmm)
unblu_pl = np.array(unblu_pl)
unblu_uni = np.array(unblu_uni)
unblu_obs = np.array(unblu_obs)

unblu_true = log_blu(unblu_true)
unblu_hdpgmm = log_blu(unblu_hdpgmm)
unblu_pl = log_blu(unblu_pl)
unblu_uni = log_blu(unblu_uni)
unblu_obs = log_blu(unblu_obs)





###
m1_unlensed = [] 
uli = []
for i in range(8):
    t1 = time.time()
    for j in range(i+1,8):
        m1_unlensed.append((m1[i]+m2[j])/2)
        uli.append([i,j])
m1_unlensed = np.array(m1_unlensed)
uli = np.array(uli)



#from marginal_plot import *
#plotting_two(n_true, n_pow, n_hdpgmm, pop_model, pop_true,pop_power, nm1_ref[:37], xlim, ylim,math_blu)
#plotting_two(blu_true[:n], blu_uni[:n],pop_model, pop_true,pop_power, m1z[:n], xlim, ylim,math_blu,p3=True)
#plotting_two(blu_true[:20], blu_uni[:20],pop_model, pop_true,pop_power, m1[:20], xlim, ylim,math_blu)
#plt.savefig('blu_lensed_uni.pdf',bbox_inches = 'tight')
#ylim=(0,3.1)
#plotting_two(blu_true[:n], blu_pl[:n],pop_model, pop_true,pop_power, m1z[:n], xlim, ylim,math_blu,p2=True)
#plotting_two(blu_true[:n], blu_obs[:n],pop_model, pop_true,pop_obs, m1[:n], xlim, ylim,math_blu,p2=True)

#plotting_two(blu_true[:20], blu_uni[:20],pop_model, pop_true,pop_power, m1[:20], xlim, ylim,math_blu)
#plt.savefig('blu_lensed_pl.pdf',bbox_inches = 'tight')
#ylim=(0.5,2.26)
#ylim=(0,5.1)
#plotting_two(blu_true[:n], blu_hdpgmm[:n],pop_model, pop_true,pop_true, m1z[:n], xlim, ylim,math_blu,p1=True,obs=True)
#plotting_two(blu_true[:20], blu_uni[:20],pop_model, pop_true,pop_power, m1[:20], xlim, ylim,math_blu)
#plt.savefig('blu_lensed_dpgmm.pdf',bbox_inches = 'tight')

with open('./real_dist.pkl', 'rb') as f:
    pop_true=dill.load(f)
# HDPGMM 
with open('./PL_pdf.pkl', 'rb') as f:
    pop_power=dill.load(f)
    
with open('/Users/damon/Desktop/lensingpop/result/obs_pop_m1m2.pkl', 'rb') as f:
    pop_obs=dill.load(f)[0]
xlim=(10,170)
ylim=(-2.1,4.3)
n = 1
m1z=m1*(1+redshift)
#plotting_two(n_true, n_pow, n_hdpgmm, pop_model, pop_true,pop_power, nm1_ref[:37], xlim, ylim,math_blu)
plotting_two(blu_true[:n], blu_uni[:n],pop_model, pop_true,pop_power, m1z[:n], xlim, ylim,math_blu,p3=True)
#plotting_two(blu_true[:20], blu_uni[:20],pop_model, pop_true,pop_power, m1[:20], xlim, ylim,math_blu)
#plt.savefig('blu_lensed_uni.pdf',bbox_inches = 'tight')
#ylim=(0,3.1)
plotting_two(blu_true[:n], blu_pl[:n],pop_model, pop_true,pop_power, m1z[:n], xlim, ylim,math_blu,p2=True)
#plotting_two(blu_true[:n], blu_obs[:n],pop_model, pop_true,pop_obs, m1[:n], xlim, ylim,math_blu,p2=True)

#plotting_two(blu_true[:20], blu_uni[:20],pop_model, pop_true,pop_power, m1[:20], xlim, ylim,math_blu)
#plt.savefig('blu_lensed_pl.pdf',bbox_inches = 'tight')
#ylim=(0.5,2.26)
#ylim=(0,5.1)
plotting_two(blu_obs[:n], blu_hdpgmm[:n],pop_model, pop_true,pop_obs, m1z[:n], xlim, ylim,math_blu,p1=True,obs=True)
#plotting_two(blu_true[:20], blu_uni[:20],pop_model, pop_true,pop_power, m1[:20], xlim, ylim,math_blu)
#plt.savefig('blu_lensed_dpgmm.pdf',bbox_inches = 'tight')


fig1 = corner(img1[0].rvs(10000),bins=30,plot_datapoints=False,levels=(0.5,0.8,0.95),plot_density=False,color=colors[0])
fig2 = corner(img1[4].rvs(10000),bins=30,fig=fig1,plot_datapoints=False,levels=(0.5,0.8,0.95),plot_density=False,color=colors[1])

xlim=(5,90)
ylim=(-10,5.6)
n = 30

plotting_two(unblu_true[:n], unblu_uni[:n],pop_model, pop_true,pop_power, m1_unlensed[:n], xlim, ylim,math_blu,p3=True)
plt.savefig('blu_unlensed_uni.pdf',bbox_inches = 'tight')
#ylim=(0,3.1)
plotting_two(unblu_true[:n], unblu_pl[:n],pop_model, pop_true,pop_power, m1_unlensed[:n], xlim, ylim,math_blu,p2=True)
plt.savefig('blu_unlensed_pl.pdf',bbox_inches = 'tight')
#ylim=(0,5.1)
plotting_two(unblu_obs[:n], unblu_hdpgmm[:n],pop_model, pop_true,pop_obs, m1_unlensed[:n], xlim, ylim,math_blu,p1=True,obs=True)
plt.savefig('blu_unlensed_dpgmm.pdf',bbox_inches = 'tight')


import corner.corner as cn 
for i in range(36):
    if unblu_true[i,0] > 0:
        ii, jj = uli[i]
    
        corn 
    
    
#pop_obs = pop_obs[0]
xlim=(5,170)
ylim=(-3,3.1)
#ylim=(1e-3,1e3)
n = 25

plotting_diff(blu_true[:n], blu_uni[:n],pop_model, pop_true,pop_power, m1z[:n], xlim, ylim,math_blu,p3=True)
#plt.savefig('blu_lensed_uni.pdf',bbox_inches = 'tight')
#ylim=(0,3.1)
plotting_diff(blu_true[:n], blu_pl[:n],pop_model, pop_true,pop_power, m1z[:n], xlim, ylim,math_blu,p2=True)
#plt.savefig('blu_lensed_pl.pdf',bbox_inches = 'tight')

#ylim=(0,5.1)
plotting_diff(blu_obs[:n], blu_hdpgmm[:n],pop_model, pop_true,pop_obs, m1z[:n], xlim, ylim,math_blu,p1=True,obs=True)
#plt.savefig('blu_lensed_dpgmm.pdf',bbox_inches = 'tight')

"""
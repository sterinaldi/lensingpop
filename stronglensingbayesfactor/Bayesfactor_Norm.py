
from functions import *
import scipy.integrate as integrate
np.random.seed(7)

with open("gwdet_default_interpolator", "rb") as f:
    fpdet = pickle.load(f)

kappa = 1.0
zmin = 0 
zmax = 2.3
norm = integrate.quad(lambda x: (1+x)**(kappa-1)*cosmo.differential_comoving_volume(x).to(u.Gpc**3/u.sr).value, zmin, zmax)[0]
print(norm)

# make a redshift-luminosity distance interpolator
zmin = 1e-7
zmax = 15
z_grid = np.geomspace(zmin,zmax, 400)
z_eval = interp1d(LuminosityDistance(z_grid), z_grid)


## parameters for Power-law plus peak model
lambda_d = 0.10
mmin = 4.59 
mmax = 86.22
PP_pars = {'alpha': 3.63, 
  'beta': 1.26,
  'delta_m': 4.82,
  'mmin': mmin,
  'mmax': mmax,
  'lam': 0.08,
  'mpp': 33.07,
  'sigpp': 5.69}

mass_dist = mass_distribution(**PP_pars)



def Selection_lensed(pmass,pz,Nzsample = int(1e6)):
    ######## draw sample from plenz    
    z_sample = pz.resample(size=Nzsample)
    z_sample = inverse_transform_sampling(z_grid, redshift_pdf(z_grid,norm=norm), nSamples=Nzsample)
    z_sample = z_sample[z_sample>zmin]
    z_sample = z_sample[z_sample<zmax]
    
    mu1 = inverse_transform_sampling(bins=mu_bins, pdf= mu_pdf, nSamples=z_sample.size)
    mu2 = mu1.copy()
    for i in range(mu1.size):
        mu2[i] = TruncNormSampler(1.5,13,mu1[i],0.01*mu1[i],Nsamples=1)
    
    lm = LuminosityDistance(z_sample)
    lmm1 = lm/mu1
    lmm2 = lm/mu2
    
    zeff1 = lmm1.copy()
    index_in = np.where((lmm1 <=lm_max) *(lmm1 >= lm_min))[0]
    zeff1[lmm1 >lm_max]= zmax
    zeff1[lmm1 <lm_min]= zmin
    zeff1[index_in]= z_eval(lmm1[index_in])
    
    zeff2 = lmm2.copy()
    index_in = np.where((lmm2 <=lm_max) *(lmm2 >= lm_min))[0]
    zeff2[lmm2 >lm_max]= zmax
    zeff2[lmm2 <lm_min]= zmin
    zeff2[index_in]= z_eval(lmm2[index_in])
    

    zeff1 = z_eval(lmm1)
    zeff2 = z_eval(lmm2)
    mass_sample= pmass.resample(size=z_sample.size)

    arr = np.array([mass_sample[0],mass_sample[1],zeff1]).T
    arr2 = np.array([mass_sample[0],mass_sample[1],zeff2]).T
    
    ans  = fpdet(arr)*fpdet(arr2)

    return np.mean(ans)





############## import population prior ################
############## should replaced by DPGMM ###############
data = np.load('PowerlawplusPeak5000Samples.npz')
m1= data['m1']
m2 = data['m2']
redshift = data['redshift']
pop_data = np.array([m1,m2,redshift]).T
#print(pop_data.shape)
#pop_src = DensityEstimator(pop_data)

pz = DensityEstimator(redshift.reshape(redshift.size,1))

pmass = DensityEstimator(np.array([m1,m2]).T)
############## import posterior data ################

# m1,m2 posterior for observed events
data = np.load("m1m2posterior_PPD_afterSelection1000000.npz")
m1_posterior = data['m1_posterior'][:50]
m2_posterior = data['m2_posterior'][:50]

print('we have {:d} events with {:d} posterior sample each.'.format(m1_posterior.shape[0],m1_posterior.shape[1]))

#pmass = DensityEstimator(np.array([m1src,m2src]).T)

########## computation of Bayes factor based on the overlapping of parameters

mu_bins = np.linspace(0,15,200)
mu_pdf = powerlaw_pdf(mu_bins, -3, 1, 13)

zmin = 1e-4
zmax = 15
z_grid = np.geomspace(zmin,zmax, 400)
z_eval = interp1d(LuminosityDistance(z_grid), z_grid)

lm_min = LuminosityDistance(zmin)
lm_max = LuminosityDistance(zmax)



alpha = Selection_unlensed(pmass,pz)
print(alpha)
beta = Selection_lensed(pmass,Nzsample=100000)
print(beta)

def BayesFactor(event1,event2,z,pmass,pz,pop_prior,Nsample=int(1e6)):
    p1 = DensityEstimator(event1)
    p2 = DensityEstimator(event2)

    # Draw samples for Monte Carlo integration
    sample = p2.resample(size=Nsample)

    probability_event1 = p1.pdf(sample)
    pdet_p2 = fpdet(sample)
    #### this population prior is reconstruncted with selection effect  #### 

    population_prior = pop_prior.pdf(np.array([sample[0],sample[1],np.repeat(z,Nsample)]))

    MCsample_mean = np.mean(probability_event1*pdet_p2 /population_prior)
    
    return alpha / beta * MCsample_mean
    
################# Compute Bayes factor for each pairs #################################


Nevent = m1_posterior.shape[0]
bayesf = np.zeros(Nevent*(Nevent-1))

index = 0 
for i in range(Nevent):

    for j in range(i+1,Nevent):
        e1 = np.array([m1_posterior[i], m2_posterior[i]]).T
        e2 = np.array([m1_posterior[j], m2_posterior[j]]).T
        bayesf[index] = BayesFactor(e1,e2,redshift[1],pmass,pz,Nsample=int(1e5))
        index +=1
        print(i,j)

filename = "BayesFactor_Result.npz"

np.savez(filename, Bayes = bayesf)

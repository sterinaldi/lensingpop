#!/users/hoi-tim.cheung/.conda/envs/dmm/bin/python

import dill
import numpy as np

np.random.seed(0)
rdir = '/Users/damon/desktop/lensingpop/'


method = 'pp'

selection = True 
# Load detectability function 
if selection:
    with open(rdir+'/Mock_Data/gwdet_default_interpolator.pkl', 'rb') as f:
        detectability = dill.load(f)

# Load image 2 
with open(rdir+'/Mock_Data/lensed_data/l1_psm3.pkl', 'rb') as f:
    LensedEvent1_List = dill.load(f)

# Load image 2 
with open(rdir+'/Mock_Data/lensed_data/l2_psm3.pkl', 'rb') as f:
    LensedEvent2_List = dill.load(f)


# Load population prior
filename = 'true_prior' if method == 'ppd' else 'wrong_prior' 

with open(rdir+'/Mock_Data/priors/'+filename+'.pkl', 'rb') as f:
    population_prob = dill.load(f)


bayes = np.zeros(1986)


Nmc = int(1e6)
# Compute Blu for each lensed pair

for i in range(bayes.size):
    event1_posterior = LensedEvent1_List[i]
    event2_posterior = LensedEvent2_List[i]
    # perform importance sampling
    print(i) 
    # take samples from event 1 
    sample = event1_posterior.sample_from_dpgmm(Nmc)
    # evaluate event2 posterior on event1 samples
    event2_pdf = event2_posterior.evaluate_mixture(sample)
    # evaluate population probablity on event1 samples
    pop = population_prob.evaluate_mixture(sample)
    if selection:
        pdet = detectability(sample)
        bayes[i]=np.mean(event2_pdf[pop>0] * pdet[pop>0] / pop[pop>0])
    else:
        bayes[i]=np.mean(event2_pdf[pop>0] / pop[pop>0])


        
if selection:
    np.savez(rdir+'/stronglensingbayesfactor/result_data/'+filename+'_selection'+'.npz',bayesfactor=bayes)
else:
    np.savez(rdir+'/stronglensingbayesfactor/result_data/'+filename+'.npz',bayesfactor=bayes)

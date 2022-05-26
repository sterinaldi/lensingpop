#!/users/hoi-tim.cheung/.conda/envs/dmm/bin/python

import dill
import numpy as np
import os 
np.random.seed(0)
import os 
import sys
cdir = os.path.dirname(sys.path[0])

import argparse
parser = argparse.ArgumentParser(description='Generate population and posterior samples.')
parser.add_argument('--N',type=int,help='number of events in the catalog',default=1000000)
parser.add_argument('--p_prior',type=str,help='choose of population prior',default='ppd')

args = parser.parse_args()
N = int(args.N) # Sunber of events 
method = args.p_prior

selection = True 
# Load detectability function 
if selection:
    with open(cdir+'/Mock_Data/gwdet_default_interpolator.pkl', 'rb') as f:
        detectability = dill.load(f)

# Load image 2 
with open(cdir+'/Mock_Data/lensed_data/l1_ps{:.0f}.pkl'.format(N), 'rb') as f:
    LensedEvent1_List = dill.load(f)

# Load image 2 
with open(cdir+'/Mock_Data/lensed_data/l2_ps{:.0f}.pkl'.format(N), 'rb') as f:
    LensedEvent2_List = dill.load(f)


# Load population prior
filename = 'true_prior' if method == 'ppd' else 'wrong_prior' 

with open(cdir+'/Mock_Data/priors/'+filename+'.pkl', 'rb') as f:
    population_prob = dill.load(f)


def blu(event1_post,event2_post, Nmc):
    Nmc = int(Nmc)
    sample = event1_post.rvs(Nmc)
    # evaluate event2 posterior on event1 samples
    event2_pdf = event2_post.pdf(sample)
    # evaluate population probablity on event1 samples
    pop = population_prob.pdf(sample)
    if selection:
        pdet = detectability(sample)
        blu = np.mean(event2_pdf[pop>0] * pdet[pop>0] / pop[pop>0])
    else:
        blu = np.mean(event2_pdf[pop>0] / pop[pop>0])
    return blu

bayes = np.zeros(len(LensedEvent1_List))


Nmc = int(1e6)
# Compute Blu for each lensed pair

for i in range(len(LensedEvent1_List)):
    print(i+1,'-th lensed pair calculating...')
    bayes[i] = blu(LensedEvent1_List[i], LensedEvent2_List[i],1e6)
    print('BLU = ', bayes[i])



	
if selection:
    save_path = cdir+'stronglensingbayesfactor/result_data/'
    filename +='_selection.npz'
else:
    save_path = cdir+'stronglensingbayesfactor/result_data/'
    filename += '.npz'
if not os.path.exists(save_path):
    os.makedirs(save_path)


np.savez(save_path + filename,bayesfactor=bayes)

print('BLU result is saved at ' + save_path + filename)

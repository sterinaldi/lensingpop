#!/users/hoi-tim.cheung/.conda/envs/dmm/bin/python
from figaro.mixture import DPGMM 
import numpy as np
import dill
import os 
import sys
cdir = os.path.dirname(os.path.dirname(os.path.dirname(sys.path[0])))
np.random.seed(0)


import argparse
parser = argparse.ArgumentParser(description='Generate population and posterior samples.')
parser.add_argument('--N',type=int,help='number of events in the catalog',default=1000000)
args = parser.parse_args()
N = int(args.N) # Sunber of events 

filename = cdir + '/Mock_Data/lensed_posterior{:.0f}.npz'.format(N)
data = np.load(filename)
l1m1 = data['m1p1']
l1m2 = data['m2p1']
l2m1 = data['m1p2']
l2m2 = data['m2p2']
l1z = data['zp1']
l2z = data['zp2']

mmin = min(np.min(l1m1),np.min(l1m2),np.min(l2m1),np.min(l2m2)) - 1.0 
mmax = max(np.max(l1m1),np.max(l1m2),np.max(l2m1),np.max(l2m2)) + 1.0
zmin = min(np.min(l1z),np.min(l2z)) - 0.01
zmax = max(np.max(l1z),np.max(l2z)) + 0.01


bounds = [[mmin,mmax],[mmin,mmax],[zmin,zmax]]
print(bounds)


mix = DPGMM(bounds)

model = []
i = 0
for m1,m2,z in zip(l1m1,l1m2,l1z):
    print(i)
    i+=1
    model.append(mix.density_from_samples(np.array([m1,m2,z]).T))
    #model.append( mix.build_mixture() )
    #mix.initialise()
# model[] list contains m1.size models   
# save
with open(cdir+'/Mock_Data/lensed_data/l1_ps{:.0f}.pkl'.format(N), 'wb') as f:
    dill.dump(model, f)

print('dpgmm model for image1 is saved at ' + cdir+'/Mock_Data/lensed_data/l1_ps{:.0f}.pkl'.format(N))
model = []
i = 0 
for m1,m2,z in zip(l2m1,l2m2,l2z):
    print(i)
    i+=1
    model.append(mix.density_from_samples(np.array([m1,m2,z]).T))
    #model.append( mix.build_mixture() )
    #mix.initialise()

# model[] list contains m1.size models   

# save
with open(cdir+'/Mock_Data/lensed_data/l2_ps{:.0f}.pkl'.format(N), 'wb') as f:
    dill.dump(model, f)
#
print('dpgmm model for image2 is saved at ' + cdir+'/Mock_Data/lensed_data/l2_ps{:.0f}.pkl'.format(N)) 

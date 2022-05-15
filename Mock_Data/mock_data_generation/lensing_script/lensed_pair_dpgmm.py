#!/users/hoi-tim.cheung/.conda/envs/dmm/bin/python
from figaro.mixture import DPGMM 
import numpy as np
import dill

np.random.seed(0)

rdir = '/Users/damon/desktop/lensingpop/'

filename = rdir+'/Mock_Data/lensed_data/lensed_ps1986.npz'
data = np.load(filename)
l1m1 = data['l1m1']
l1m2 = data['l1m2']
l2m1 = data['l2m1']
l2m2 = data['l2m2']
l1z = data['l1z']
l2z = data['l2z']

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
    mix.density_from_samples(np.array([m1,m2,z]).T)
    model.append( mix.build_mixture() )
    mix.initialise()
# model[] list contains m1.size models   
# save
with open(rdir+'/Mock_Data/lensed_data/l1_psm3.pkl', 'wb') as f:
    dill.dump(model, f)
model = []
i = 0 
for m1,m2,z in zip(l2m1,l2m2,l2z):
    print(i)
    i+=1
    mix.density_from_samples(np.array([m1,m2,z]).T)
    model.append( mix.build_mixture() )
    mix.initialise()

# model[] list contains m1.size models   

# save
with open(rdir+'/Mock_Data/lensed_data/l2_psm3.pkl', 'wb') as f:
    dill.dump(model, f)
# Models for each events

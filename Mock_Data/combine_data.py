import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Generate population and posterior samples.')
parser.add_argument("-N", type = int, help = "number of population events", default=10000)
parser.add_argument("-L", dest = "L", action = 'store_true', help = "Generate lensed population", default = False)

args = parser.parse_args()
N = args.N # nunber of events

m1m2zfile = './catalog/m1m2z_posterior_PPD_afterSelection_'+str(N)
spinfile = './catalog/spin_data_'+str(N)
outputfile = './catalog/m1m2zxeff_posterior_'+str(N)

if args.L:
    m1m2zfile += '_lensed.npz'
    spinfile += '_lensed.npz'
    outputfile += '_lensed.npz'

else:
    m1m2zfile += '_unlensed.npz'
    spinfile += '_unlensed.npz'
    outputfile += '_unlensed.npz'

data= np.load(m1m2zfile)
m1 = data['m1']
m2 = data['m2']
z = data['redshift']

m1p = data['m1_posterior']
m2p = data['m2_posterior']
zp = data['z_posterior']

data= np.load(spinfile)
xeff=data['chi_eff']
xeffp=data['eff_posterior']


np.savez(outputfile,m1=m1,m2=m2,redshift=z,xeff=xeff,
        m1_posterior=m1p,m2_posterior=m2p,z_posterior=zp,xeff_posterior = xeffp)

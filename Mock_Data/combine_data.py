import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Generate population and posterior samples.')
parser.add_argument("-N", type = int, help = "number of population events", default=10000)
args = parser.parse_args()


N = args.N # nunber of events

data= np.load('./m1m2z_posterior_PPD_afterSelection_'+str(N)+'.npz')
m1 = data['m1']
m2 = data['m2']
z = data['redshift']

m1p = data['m1_posterior']
m2p = data['m2_posterior']
zp = data['z_posterior']


data= np.load('./spin_data_'+str(N)+'_unlensed.npz')
xeff=data['chi_eff']
xp=data['chi_p']
xeffp=data['eff_posterior']
xpp=data['p_posterior']

np.savez('m1m2zxeffxp_posterior_PPD_afterSelection_'+str(N)+'_unlensed.npz',m1=m1,m2=m2,redshift=z,
         xeff=xeff,xp=xp,m1_posterior=m1p,m2_posterior=m2p,z_posterior=zp,
        xeff_posterior = xeffp, xp_posterior=xpp)

import numpy as np
import argparse
from scipy.stats import truncnorm   
from scipy.interpolate import interp1d
from simulated_universe import *
from figaro.load import _find_redshift
from pathlib import Path
#import matplotlib.pyplot as plt
from calculate_snr_veske import *

np.random.seed(0)
#pdetfunction = gwdet.detectability()
# Here come all the definitions used in this script
def veske_snr(mass1,mass2,z):#compute the detection probability with the HLV network at O3 sensitivity with detection threshold POWER SNR=64 (=amplitude SNR=8). Marginalized over the (20^4 x 300) points in the angle and distance grid
#    i = min(99,int(mass1))
#    j = min(99,int(mass2)) 
    i = int(mass1)
    j = int(mass2) 
#    print(r,n)
    #x1 = np.sqrt(np.sum((1000**2*np.outer((1+z)**al[int(np.round(100*j/i))-1]/r**2,(o3l[i,j]*anl+o3h[i,j]*anh+o3v[i,j]*anv)))/(n**4), axis = -1))   
    x2 = np.sqrt(np.sum((1000**2*np.outer((1+z)**al[int(np.round(100*j/i))-1],(o3l[i,j]*anl+o3h[i,j]*anh+o3v[i,j]*anv)))/(n**4), axis = -1))   
    return x2

m = np.linspace(0,100,100)
redshift = np.linspace(0,z_max,100)


############################################################################################################
snr = np.zeros((100,100,100))
for j in tqdm(range(5,100)): #Consider small mass between [5,99] solar masses, in accordance with the SNR code
    for i in range (j,min(100,10*j+1)): #Consider heavy mass between [m2,min(99,10*m2)] solar masses, in accordance with the SNR code
        for k in range(0,100):
            snr[i,j,k] = veske_snr(i,j,redshift[k])
m = np.arange(0, 100)
interp = RegularGridInterpolator(points = (m, m, redshift), values = snr, method = 'linear', bounds_error = False, fill_value = 0.)
with open('snr_m1m2z.pkl', 'wb') as f:
    dill.dump(interp, f)



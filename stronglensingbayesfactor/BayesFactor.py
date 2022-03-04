import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss 
import corner
import scipy.integrate as integrate

############# Our density/probability estimator #############

### Now we demonstrate with kde first
def DensityEstimator(data):
    
    values = np.vstack(data.T)
    kernel = ss.gaussian_kde(values)
    
    return kernel




############## import population prior ################

# now we use kde to construct the population prior for demostration, will replace it later 
data = np.load("m1m2posterior_PPD_afterSelection1000000.npz")
m1= data['m1']
m2 = data['m2']


pop_data = np.array([m1,m2]).T
#print(pop_data.shape)
pop_prior = DensityEstimator(pop_data)


############## import posterior data ################

# m1,m2 posterior for observed events

m1_posterior = data['m1_posterior']
m2_posterior = data['m2_posterior']

print('we have {:d} events with {:d} posterior sample each.'.format(m1_posterior.shape[0],m1_posterior.shape[1]))


########## computation of Bayes factor based on the overlapping of parameters


############# notice that the Bayes Factor is not properly normalized #################################
############# working on the normalization (selection effect)  #################################

"""
def hypothesis_prior():

    return 

def Selection_lensed():

    return 

def Selection_unlensed():

    return 
"""

def BayesFactor(event1,event2,pop_prior,Nsample=10000):
    p1 = DensityEstimator(event1)
    
    p2 = DensityEstimator(event2)
    # Draw samples for Monte Carlo integration
    sample = p2.resample(size=Nsample)
   
    probability_event1 = p1.pdf(sample)

    population_prior = pop_prior.pdf(sample)

    MCsample_mean = np.mean(probability_event1/population_prior)

    #prior = hypothesis_prior()

    #alpha = Selection_unlensed()
    #beta = Selection_lensed()


    #return prior * alpha / beta * MCsample_mean
    return MCsample_mean




################# some examples #################################


e1 = np.array([m1_posterior[1], m2_posterior[1]]).T

e2 = np.array([m1_posterior[10], m2_posterior[10]]).T

np.random.seed(0)

####################### unlensed pair #############################
# suppose we now we want to compute the Bayes Factor of a unlensed pair 
print("unlensed example")
observed_event1 = e1
observed_event2 = e2

#print(np.linalg.eigh(np.cov(observed_event1.T))[0] <= 0)

approx_m1 = np.mean(observed_event1,axis=0)[0]
approx_m2 = np.mean(observed_event1,axis=0)[1]

print("event1 approx: m1 = {:.2f}, m2 = {:.2f}.\n".format(approx_m1,approx_m2))
print("Approx. population likelihood for event1 = ",pop_prior.pdf(np.array([approx_m1,approx_m2]).T),"\n")
approx_m1 = np.mean(observed_event2,axis=0)[0]
approx_m2 = np.mean(observed_event2,axis=0)[1]
print("event2 approx: m1 = {:.2f}, m2 = {:.2f}.\n".format(approx_m1,approx_m2))
print("Approx. population likelihood for event2 = ",pop_prior.pdf(np.array([approx_m1,approx_m2]).T),"\n")

Bayes = BayesFactor(observed_event1,observed_event2,pop_prior,Nsample=int(5e4))
print("Bayes Factor = {:.2f} \n".format(Bayes) )


###################  lensed pair  ####################
# suppose we now we want to compute the Bayes Factor of a lensed pair 
print("lensed example 1 ----------------------------\n")
# make them to be identical, one with some noise
observed_event1 = e1
observed_event2 = e1 
approx_m1 = np.mean(observed_event1,axis=0)[0]
approx_m2 = np.mean(observed_event1,axis=0)[1]
print("event1 approx: m1 = {:.2f}, m2 = {:.2f}.\n".format(approx_m1,approx_m2))
print("Approx. population likelihood for event1 = ",pop_prior.pdf(np.array([approx_m1,approx_m2]).T),"\n")
approx_m1 = np.mean(observed_event2,axis=0)[0]
approx_m2 = np.mean(observed_event2,axis=0)[1]
print("event2 approx: m1 = {:.2f}, m2 = {:.2f}.\n".format(approx_m1,approx_m2))
print("Approx. population likelihood for event2 = ",pop_prior.pdf(np.array([approx_m1,approx_m2]).T),"\n")
#print(observed_event1-observed_event2)
Bayes = BayesFactor(observed_event1,observed_event2,pop_prior,Nsample=int(5e4))
print("Bayes Factor = {:.2f} \n".format(Bayes) )


print("lensed example 2---------------------------\n")
observed_event1 = e2
observed_event2 = e2 
approx_m1 = np.mean(observed_event1,axis=0)[0]
approx_m2 = np.mean(observed_event1,axis=0)[1]
print("event1 approx: m1 = {:.2f}, m2 = {:.2f}.\n".format(approx_m1,approx_m2))
print("Approx. population likelihood for event1 = ",pop_prior.pdf(np.array([approx_m1,approx_m2]).T),"\n")
approx_m1 = np.mean(observed_event2,axis=0)[0]
approx_m2 = np.mean(observed_event2,axis=0)[1]
print("event2 approx: m1 = {:.2f}, m2 = {:.2f}.\n".format(approx_m1,approx_m2))
print("Approx. population likelihood for event2 = ",pop_prior.pdf(np.array([approx_m1,approx_m2]).T),"\n")
Bayes = BayesFactor(observed_event1,observed_event2,pop_prior,Nsample=int(5e4))
print("Bayes Factor = {:.2f} \n".format(Bayes) )


print("The normalization of Bayes factor is still missing. Working on it.")




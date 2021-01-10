import numpy as np
from BanditTools import *
from BanditBaselines import *
from Arms import *
from StochasticBandit import *

def SinBandit(X,sigma=0.5):
    """X : matrix of features of dimension (K,d), 
    theta : regression vector of dimension (d,1), 
    sigma : stdev of Gaussian noise"""
    Arms = []
    (K,d)=np.shape(X)
    normalizing_factor=(1/d)

    for k in range(K):
        x_k=np.array(X[k])
        mu = normalizing_factor*np.sum(np.sin(x_k))
        arm = arms.Gaussian(mu,sigma**2)
        Arms.append(arm)

    return MAB(Arms)

def ExpBandit(X,sigma=0.5):
    """X : matrix of features of dimension (K,d), 
    theta : regression vector of dimension (d,1), 
    sigma : stdev of Gaussian noise"""
    Arms = []
    (K,d)=np.shape(X)
    normalizing_factor=(1/d)
    
    for k in range(K):
        x_k=np.array(X[k])
        mu = normalizing_factor*np.sum(np.exp(x_k)-1)
        arm = arms.Gaussian(mu,sigma**2)
        Arms.append(arm)

    return MAB(Arms)
    
def TrickyBandit(X,sigma=0.5):
    """X : matrix of features of dimension (K,d), 
    theta : regression vector of dimension (d,1), 
    sigma : stdev of Gaussian noise"""
    Arms = []
    (K,d)=np.shape(X)
    normalizing_factor=(1/d)
    
    for k in range(K):
        x_k=np.array(X[k])
        mu = normalizing_factor*np.sum(np.sin(np.power(x_k,-1)))
        arm = arms.Gaussian(mu,sigma**2)
        Arms.append(arm)

    return MAB(Arms)
    
def SuperTrickyBandit(X,sigma=0.5):
    """X : matrix of features of dimension (K,d), 
    theta : regression vector of dimension (d,1), 
    sigma : stdev of Gaussian noise"""
    Arms = []
    (K,d)=np.shape(X)
    
    for k in range(K):
        x_k=np.array(X[k].cpu())
        mu=(1/d)*np.sum(np.sin(np.power(x_k,-1))*np.cos(np.power(x_k-1,-1))*np.sin(np.power(x_k+1,-1)))
        arm = arms.Gaussian(mu,sigma**2)
        Arms.append(arm)

    return MAB(Arms)
    
def CosInverseBandit(X,sigma=0.5):
    """X : matrix of features of dimension (K,d), 
    theta : regression vector of dimension (d,1), 
    sigma : stdev of Gaussian noise"""
    Arms = []
    (K,d)=np.shape(X)
    
    for k in range(K):
        x_k=np.array(X[k])
        mu=(1/d)*np.sum(np.cos(0.1*np.power(x_k,-1)))
        arm = arms.Gaussian(mu,sigma**2)
        Arms.append(arm)

    return MAB(Arms)
    

import numpy as np
import numpy.linalg as la
import pandas as pd
from matplotlib import pyplot as plt
import time

import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split

# Bandit specific functions and classes (same as last time) 
import Arms as arm
from StochasticBandit import * 
from BanditBaselines import * # you will need UCB alpha to your baselines 

from Experiments import * # all the previous functions to run experiments

# library enables to save some interesting vectors/arrays :

import pickle

#Neural network for estimating the mean reward of each arm 
class MeanEstimator(nn.Module):
    def __init__(self,d,m,L):
        super().__init__()
        self.d=d
        self.m=m
        self.L=L
        
        self.modules = [nn.Linear(d,m,bias=False),nn.ReLU()]
        
        first_init=np.sqrt(4/m)*torch.randn((self.m,(self.d//2)-1))
        self.modules[0].weight.data=torch.cat([first_init,torch.zeros(self.m,1),torch.zeros(self.m,1),first_init],axis=1)
        
        for i in range (1,L-1):
            init=np.sqrt(4/m)*torch.randn((self.m,(self.m//2)-1))
            self.modules.append(nn.Linear(m,m,bias=False))
            self.modules[-1].weight.data=torch.cat([init,torch.zeros(self.m,1),torch.zeros(self.m,1),init],axis=1)
            self.modules.append(nn.Tanh())
            
        last_init=np.sqrt(2/m)*torch.randn(1,self.m//2)
        self.modules.append(nn.Linear(m,1,bias=False))
        self.modules[-1].weight.data=torch.cat([last_init,last_init],axis=1)
        self.modules.append(nn.Tanh())

        self.sequential = nn.Sequential(*self.modules)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x=self.sequential(x)
    
        return np.sqrt(m)*x
    
## flatten a large tuple containing tensors of different sizes
def flatten(tensor):
    T=torch.tensor([])
    for element in tensor:
        T=torch.cat([T,element.flatten()])
    return T
    
#concatenation of all the parameters of a NN
def get_theta(model):
    return flatten(model.parameters())

#loss function of the neural TS
def criterion(estimated_reward,reward,m,reg,theta,theta_0):
    return 0.5*torch.sum(torch.square(estimated_reward-reward))+0.5*m*reg*torch.square(torch.norm(theta-theta_0))

#creation of a Bandit with h bounded
def Bandit(X,theta,sigma):
    """X : matrix of features of dimension (K,d), 
    theta : regression vector of dimension (d,1), 
    sigma : stdev of Gaussian noise"""
    Arms = []
    (K,d)=np.shape(X)
    for k in range(K):
        mu = float((theta.T)@np.sin(np.array(X[k])))
        arm = arms.Gaussian(mu,sigma**2)
        Arms.append(arm)
    return MAB(Arms)

#make the transformation of the context vectors so that we met the assumption of the authors
def transform(x):
    return np.vstack([x/(np.sqrt(2)*la.norm(x)),x/(np.sqrt(2)*la.norm(x))]).reshape(-1)


#generation of a MAB problem
def generate_MAB_problem(K,d,sigma):
    #Generation of normalized features - ||x|| = 1 and x symmetric
    X = torch.randn((K,d))
    cut=(K//2)
    X=torch.Tensor([transform(x) for x in X])

    #Generation of a random vector theta with norm 1
    theta = np.random.random((X.size(1),1))
    theta=theta/la.norm(theta,ord=2)

    B=Bandit(X,theta,sigma)
    # print the means of the best two arms
    print(np.sort(B.means)[-2:])

    with open(f'X_theta_{K}_{d}.pickle', 'wb') as setting:
        pickle.dump((X,theta), setting)
        
#generation of a MAB problem
def load_MAB_problem(K,d,sigma,display=True):
    for K,d in zip(K,d):
        with open(f'X_theta_{K}_{d}.pickle', 'rb') as setting:
            X,theta=pickle.load(setting)
            B=Bandit(X,theta,sigma)
            if display:
                #We display the means of the arms by ascending order
                print(f'Contextual MAB K={K} d={d}')
                print(np.sort(B.means))
                print('\n')
                

#NeuralTS algorithm
class NeuralTS:
    """Linear Thompson Sampling Strategy"""
    def __init__(self,X,nu,m,L,estimator,criterion,reg=1,name='NeuralTS'):
        self.features=X
        self.reg=reg
        (self.K,self.d) = np.shape(X)  
        self.nu=nu
        self.L=L
        self.m=m
        self.strat_name=name
        self.estimator=estimator(self.d,m,L)
        self.theta_zero=get_theta(self.estimator)
        self.optimizer = torch.optim.SGD(self.estimator.parameters(), lr = 10**(-5))
        self.current_loss=0
        self.criterion=criterion
        self.p=(self.theta_zero).size(0)
        self.clear()

    def clear(self):
        # initialize the design matrix, its inverse, 
        # the vector containing the sum of r_s*x_s and the least squares estimate
        self.t=1
        self.Design = self.reg*np.eye(self.p)
        self.DesignInv = (1/self.reg)*np.eye(self.p)
        self.estimated_rewards=torch.Tensor([])
        self.rewards=torch.Tensor([])
    
    def chooseArmToPlay(self):
        estimated_rewards=[]
        for k in range(self.K):
            f=self.estimator(self.features[k])
            g=torch.autograd.grad(outputs=f,inputs=self.estimator.parameters())
            g=flatten(g).detach().numpy()
            
            sigma=(self.reg*(g)@self.DesignInv@(g)/(self.m))
            r_tilda=(self.nu*self.nu)*sigma*torch.randn(1)+f.detach()
            estimated_rewards.append(r_tilda.detach().numpy())
            
        arm_to_pull=np.argmax(estimated_rewards)
        
        return arm_to_pull

    def receiveReward(self,arm,reward):
        self.estimated_rewards=torch.cat([self.estimated_rewards,self.estimator(self.features[arm])])
        self.rewards=torch.cat([self.rewards,torch.Tensor([reward])])
        self.optimizer.zero_grad() 
    
        self.current_loss=self.criterion(self.estimated_rewards,self.rewards,self.m,self.reg,get_theta(self.estimator),self.theta_zero)
        
        if self.t==1:
            self.current_loss.backward(retain_graph=True)    
        else:
            self.current_loss.backward
            
        self.optimizer.step() 
       
        #f(x,theta_t)
      
        f_t=self.estimator(self.features[arm])
        estimated_reward=f_t.detach().numpy()
        if self.t%10==0:
            print("estimated reward",estimated_reward)
            print("real reward",reward)
            print("loss ",self.current_loss)

        g=torch.autograd.grad(outputs=f_t,inputs=self.estimator.parameters())
        g=flatten(g).detach().numpy()
        g=g/(np.sqrt(m))
        # online update of the inverse of the design matrix
#         self.DesignInv=np.matrix.clip(self.DesignInv,-100,100)  
#         omega=self.DesignInv@g
#         self.DesignInv-=(omega@(omega.T))/(1+(g.T)@omega)
        self.Design+=g@(g.T)
        self.DesignInv=la.inv(self.Design)
        print(la.norm(self.DesignInv))
        self.t+=1

    def name(self):
        return self.strat_name

import numpy as np
from math import log,sqrt
import torch
import numpy.linalg as la
from BanditTools import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#make the transformation of the context vectors so that we met the assumption of the authors
def transform(x):
    return np.vstack([x/(np.sqrt(2)*la.norm(x)),x/(np.sqrt(2)*la.norm(x))]).reshape(-1)

def feature_to_contexts(x,K):
    '''
    Transform a feature vector x into K context vectors x_k
    '''
    
    d=len(x)
    C=np.zeros(shape=(K,K*d))
    for i in range(K):
        C[i][i*d:(i+1)*d]=x
    return C.tolist()

class FTL:
    """follow the leader (a.k.a. greedy strategy)"""
    def __init__(self,nbArms):
        self.nbArms = nbArms
        self.clear()

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
    
    def chooseArmToPlay(self):
        if (min(self.nbDraws)==0):
            return randmax(-self.nbDraws)
        else:
            return randmax(self.cumRewards/self.nbDraws)

    def receiveReward(self,arm,reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] +1
    

    def name(self):
        return "FTL"

    
    
class UCB:
    """UCB with parameter alpha"""
    def __init__(self,nbArms,alpha,c=3):
        self.nbArms = nbArms
        self.alpha=alpha
        self.t=1
        self.c=c
     
        self.clear()

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
    
    def chooseArmToPlay(self):
        if (min(self.nbDraws)==0):
            return randmax(-self.nbDraws)
        else:
            return np.argmax(self.cumRewards/self.nbDraws + np.sqrt(self.alpha*(np.log(self.t)+self.c*np.log(np.log(self.t)))/self.nbDraws))

    def receiveReward(self,arm,reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] +1
        self.t+=1

    def name(self):
        return "UCB"

class LinUCB:
    """LinUCB with threshold function beta"""
    def __init__(self,X,beta,bandit_generator=None,reg=1,sigma=0.5,name='LinUCB'):
        self.features=X
        self.reg=reg
        self.strat_name=name
        self.sigma=sigma
        (self.nbArms,self.dimension) = self.features.size()
        self.beta=beta
        self.bandit_generator=bandit_generator
            
        self.clear()

    def clear(self):
        with torch.no_grad():
            # initialize the design matrix, its inverse, 
            # the vector containing the sum of r_s*x_s and the least squares estimate
            self.t=1
            self.Design = self.reg*torch.eye(self.dimension)
            self.DesignInv = (1/self.reg)*torch.eye(self.dimension)
            self.Vector = torch.zeros((self.dimension,1))
            self.thetaLS = torch.zeros((self.dimension,1)) # regularized least-squares estimate
    
    
    def chooseArmToPlay(self):
        with torch.no_grad():
            norms=torch.Tensor([torch.sqrt(torch.matmul(torch.matmul(feature,self.DesignInv),feature.T)) for feature in self.features])
            # norms=np.array([np.sqrt(feature@self.DesignInv@feature.T) for feature in self.features])
            UCBs=torch.matmul(self.features,self.thetaLS.view(-1))+self.beta(self.t)*norms  
            return torch.argmax(UCBs).item()
        

    def receiveReward(self,arm,reward):
        with torch.no_grad():
            x = self.features[arm,:].view((self.dimension,1)) #column vector
            self.Design = self.Design + torch.matmul(x,x.T) 
            self.Vector = self.Vector + reward*x
           
            # online update of the inverse of the design matrix
            omega=torch.matmul(self.DesignInv,x)
            self.DesignInv= self.DesignInv-torch.matmul(omega,omega.T)/(1+torch.matmul(x.T,omega).item())
            # update of the least squares estimate 
            self.thetaLS = torch.matmul(self.DesignInv,self.Vector)
            
            self.t+=1
            
    def update_features(self):
        '''This method simulates the situation where the features are changed at each time t'''
        with torch.no_grad():
            K=self.nbArms
            d=self.dimension//2
            #Generation of normalized features - ||x|| = 1 and x symmetric
            X=torch.randn((K,d))
            X=torch.Tensor([transform(x) for x in X])
            self.features=X
           
    def new_MAB(self):
        '''This method actualize the MAB problem given the new features'''
        self.update_features()
        
        return self.bandit_generator(self.features,sigma=self.sigma)
        
       
    def name(self):
        return self.strat_name
        
        
class LinUCBClassifier:
    """LinUCB with threshold function beta for a classification task"""
    def __init__(self,X,Y,beta,bandit_generator=None,reg=1,name='LinUCBClassifier'):
        self.X=X
        self.Y=Y
        self.K=len(np.unique(Y))
        self.d=self.K*len(X[0])
        self.reg=reg
        self.strat_name=name
        self.beta=beta
        self.bandit_generator=bandit_generator
            
        self.clear()

    def clear(self):
        with torch.no_grad():
            # initialize the design matrix, its inverse, 
            # the vector containing the sum of r_s*x_s and the least squares estimate
            self.t=1
            self.Design = self.reg*torch.eye(self.d)
            self.DesignInv = (1/self.reg)*torch.eye(self.d)
            self.Vector = torch.zeros((self.d,1))
            self.thetaLS = torch.zeros((self.d,1)) # regularized least-squares estimate
    
    
    def chooseArmToPlay(self):
        with torch.no_grad():
            norms=torch.Tensor([torch.sqrt(torch.matmul(torch.matmul(feature,self.DesignInv),feature.T)) for feature in self.features])
            # norms=np.array([np.sqrt(feature@self.DesignInv@feature.T) for feature in self.features])
            UCBs=torch.matmul(self.features,self.thetaLS.view(-1))+self.beta(self.t)*norms  
            # print(UCBs)
            return torch.argmax(UCBs).item()
        

    def receiveReward(self,arm,reward):
        with torch.no_grad():
            x = self.features[arm,:].view((self.d,1)) #column vector
            self.Design = self.Design + torch.matmul(x,x.T) 
            self.Vector = self.Vector + reward*x
           
            # online update of the inverse of the design matrix
            omega=torch.matmul(self.DesignInv,x)
            self.DesignInv= self.DesignInv-torch.matmul(omega,omega.T)/(1+torch.matmul(x.T,omega).item())
            # update of the least squares estimate 
            self.thetaLS = torch.matmul(self.DesignInv,self.Vector)
            
            self.t+=1
            
           
    def update_features(self):
        '''This method simulates the situation where the features are changed at each time t'''
        i=np.random.randint(len(self.X))
        self.i=i
        x=self.X[i]
        y=self.Y[i]
        c=feature_to_contexts(x,self.K)
        self.features=torch.tensor(c)
        
    def new_MAB(self):
        '''This method actualize the MAB problem given the new features'''
        self.update_features()
        bandit=self.bandit_generator(self.Y[self.i],self.K)
     
        return bandit
        
    def name(self):
        return self.strat_name
    
class LinTS:
    """Linear Thompson Sampling Strategy"""
    def __init__(self,X,nu,bandit_generator=None,reg=1,sigma=0.5,name='LinTS'):
        self.features=X
        self.reg=reg
        (self.nbArms,self.dimension) = X.size()
        self.nu=nu
        self.sigma=sigma
        self.strat_name=name
        self.bandit_generator=bandit_generator
        self.clear()

    def clear(self):
        with torch.no_grad():
            # initialize the design matrix, its inverse, 
            # the vector containing the sum of r_s*x_s and the least squares estimate
            self.t=1
            self.Design = self.reg*torch.eye(self.dimension)
            self.DesignInv = (1/self.reg)*torch.eye(self.dimension)
            self.Vector = torch.zeros((self.dimension,1))
            self.thetaLS = torch.zeros((self.dimension,1)) # regularized least-squares estimate
        
    
    def chooseArmToPlay(self):
        with torch.no_grad():
            N=torch.distributions.multivariate_normal.MultivariateNormal(self.thetaLS.view(-1),(self.nu*self.nu)*self.DesignInv)
            theta_tilda=N.sample()
            return torch.argmax(torch.matmul(self.features,theta_tilda)).item()

    def receiveReward(self,arm,reward):
        with torch.no_grad():
            x = self.features[arm,:].view((self.dimension,1)) #column vector
            self.Design = self.Design + torch.matmul(x,x.T) 
            self.Vector = self.Vector + reward*x
            # online update of the inverse of the design matrix
            omega=torch.matmul(self.DesignInv,x)
            self.DesignInv= self.DesignInv-torch.matmul(omega,omega.T)/(1+torch.matmul(x.T,omega).item())
            # update of the least squares estimate 
            self.thetaLS = torch.matmul(self.DesignInv,self.Vector)
            self.t+=1
        
    def update_features(self):
        with torch.no_grad():
            '''This method simulates the situation where the features are changed at each time t'''
            K=self.nbArms
            d=self.dimension//2
            
            #Generation of normalized features - ||x|| = 1 and x symmetric
            X = torch.randn((K,d))
            X=torch.Tensor([transform(x) for x in X])
            self.features=X
            
    def new_MAB(self):
        '''This method actualize the MAB problem given the new features'''
        self.update_features()
        
        return self.bandit_generator(self.features,sigma=self.sigma)

    def name(self):
        return self.strat_name
        
class LinTSClassifier:
    """Linear Thompson Sampling Strategy for a classification task"""
    def __init__(self,X,Y,nu,bandit_generator=None,reg=1,name='LinTSClassifier'):
        self.X=X
        self.Y=Y
        self.K=len(np.unique(Y))
        self.d=self.K*len(X[0])
        self.reg=reg
        self.nu=nu
        self.strat_name=name
        self.bandit_generator=bandit_generator
        self.clear()

    def clear(self):
        with torch.no_grad():
            # initialize the design matrix, its inverse, 
            # the vector containing the sum of r_s*x_s and the least squares estimate
            self.t=1
            self.Design = self.reg*torch.eye(self.d)
            self.DesignInv = (1/self.reg)*torch.eye(self.d)
            self.Vector = torch.zeros((self.d,1))
            self.thetaLS = torch.zeros((self.d,1)) # regularized least-squares estimate
        
    
    def chooseArmToPlay(self):
        with torch.no_grad():
            N=torch.distributions.multivariate_normal.MultivariateNormal(self.thetaLS.view(-1),(self.nu*self.nu)*self.DesignInv)
            theta_tilda=N.sample()
            return torch.argmax(torch.matmul(self.features,theta_tilda)).item()

    def receiveReward(self,arm,reward):
        with torch.no_grad():
            x = self.features[arm,:].view((self.d,1)) #column vector
            self.Design = self.Design + torch.matmul(x,x.T) 
            self.Vector = self.Vector + reward*x
            # online update of the inverse of the design matrix
            omega=torch.matmul(self.DesignInv,x)
            self.DesignInv= self.DesignInv-torch.matmul(omega,omega.T)/(1+torch.matmul(x.T,omega).item())
            # update of the least squares estimate 
            self.thetaLS = torch.matmul(self.DesignInv,self.Vector)
            self.t+=1
        
    def update_features(self):
        '''This method simulates the situation where the features are changed at each time t'''
        i=np.random.randint(len(self.X))
        self.i=i
        x=self.X[i]
        y=self.Y[i]
        c=feature_to_contexts(x,self.K)
        self.features=torch.tensor(c)
        
        
        
    def new_MAB(self):
        '''This method actualize the MAB problem given the new features'''
        self.update_features()
        bandit=self.bandit_generator(self.Y[self.i],self.K)
     
        return bandit

    def name(self):
        return self.strat_name

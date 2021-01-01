import numpy as np
from math import log,sqrt
from BanditTools import *


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


class UniformExploration:
    """a strategy that uniformly explores arms"""
    def __init__(self,nbArms):
        self.nbArms = nbArms
        self.clear()

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
    
    def chooseArmToPlay(self):
        return np.random.randint(0,self.nbArms)

    def receiveReward(self,arm,reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] +1

    def name(self):
        return "Uniform"
    
    
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


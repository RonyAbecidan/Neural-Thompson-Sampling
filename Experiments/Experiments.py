import numpy as np
import numpy.linalg as la
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm
from IPython.display import clear_output


def OneBanditOneLearnerOneRun(bandit, strategy, timeHorizon,update=False):
    """
    Run a bandit strategy (strategy) on a MAB instance (bandit) for (timeHorizon) time steps
    output : sequence of arms chosen, sequence of rewards obtained
    """
    selections = []
    rewards = []
    best_strategy=[]
    already_visited_means=[]
    strategy.clear() # reset previous history
    for t in tqdm(range(timeHorizon)):
        # choose the next arm to play with the bandit algorithm
        arm = strategy.chooseArmToPlay()
        # get the reward of the chosen arm
        reward = bandit.generateReward(arm)
        # update the algorithm with the observed reward
        strategy.receiveReward(arm, reward)
        if (not update):
            # store what happened
            selections.append(arm)
            rewards.append(reward)
        else:
            
            already_visited_means.append(bandit.means[arm])
            best_strategy.append(max(bandit.means))
            bandit=strategy.new_MAB()
           
            
    clear_output(wait=True)
    if update:
        return already_visited_means,best_strategy
            
    else:
        return selections, rewards

def CumulativeRegret(bandit=None,selections=None,update=False,best_strategy=None,already_visited_means=None):
    """Compute the pseudo-regret associated to a sequence of arm selections"""
    
    if update:
        return np.cumsum(np.array(best_strategy) - np.array(already_visited_means))     
    else:
        return np.cumsum(max(bandit.means)*np.ones(len(selections)) - np.array(bandit.means)[selections])

def OneBanditOneLearnerMultipleRuns(bandit, strategy, timeHorizon, N_exp, tsave=[],update=False):
    """
    Perform N_exp runs of a bandit strategy (strategy) on a MAB instance (bandit) for (timeHorizon) time steps 
    and compute the pseudo-regret of each run 
    optional : tsave gives a grid of time steps in which the results will be stored (set to 1:timeHorizon by default)
    output : a table of size N_exp x |tsave| in which each row is the pseudo-regret at the sub-sampled times 
    """
    if (len(tsave) == 0):
        tsave = np.arange(timeHorizon)
    savedTimes = len(tsave)
    
    Regret = np.zeros((N_exp, savedTimes)) # Store the regret values on different runs
    for n in range(N_exp):
        np.random.seed()
        if not(update):
            # run the bandit strategy
            selections,_= OneBanditOneLearnerOneRun(bandit, strategy, timeHorizon)
            # compute its pseudo-regret
            regret_one_run = CumulativeRegret(bandit=bandit,selections=selections)
        else:
            # run the bandit strategy
           
            already_visited_means,best_strategy = OneBanditOneLearnerOneRun(bandit, strategy, timeHorizon,update=True)
            # compute its pseudo-regret
            regret_one_run = CumulativeRegret(update=True,already_visited_means=already_visited_means,best_strategy=best_strategy)
            
        # store (a sub-sampling of) the cumulative regret
        Regret[n, :] = np.array(regret_one_run)[tsave] 
        
    return Regret

def RunExpes(algorithms,bandit,N_exp,timeHorizon,step=10,quantiles = "on",names=[],update=False):
    """run experiments with multiple algorithms"""
    tsave = np.arange(1,timeHorizon,step)
    colors = ["black","tomato","cadetblue","green","m"]
    if (names==[]):
        names = [algo.name() for algo in algorithms]
    for i in range(len(algorithms)):
        algo=algorithms[i]
        Regret = OneBanditOneLearnerMultipleRuns(bandit, algo, timeHorizon, N_exp, tsave,update=update)
        plt.plot(tsave, np.mean(Regret, 0), linewidth=2.0, color=colors[i], label="mean regret of "+ names[i])
        if (quantiles == "on"):
            plt.plot(tsave, np.quantile(Regret, 0.95, 0), tsave, np.quantile(Regret,0.05,0), linestyle="dashed", color=colors[i])
            
    plt.legend()
    plt.grid()
    
#To use only if the arms are fixed
def multiple_runs(nb_runs,bandit,strategy,horizon=10):
    T=horizon
    count={}

    for i in range(nb_runs):
        selections,rewards = OneBanditOneLearnerOneRun(bandit,strategy,T)

        count.setdefault(bandit.arms[selections[-1]],0)
        count[bandit.arms[selections[-1]]]+=1
       
    most_frequent_arm=max(count, key=count.get)

    #global results
    results=pd.DataFrame()
    results['most_frequent_empirical_best_arm']=[most_frequent_arm]
    results['real_best_arm']=[bandit.arms[bandit.bestarm]]
    
    return results
   

   
#DEPENDENCIES
import Graph
#TODO import model

import numpy as np
import math
import random
import copy

import torch
import torch.nn
import torch.optim
import torch.autograd
import torch.nn.functional as F

from collections import namedtuple, deque

#loop
#2. ENV: outputs list of possible actions v = V \ S
#3. AGENT: RNG, v* = random step or max of calculated Q values for every v
#4. AGENT: output v* 
#5. ENV: v*, update graph (g.selectNode(nodeID))
#6. ENV: return reward, new state
#7. Update params (backwards pass)
#8. Re-encode

#***********************************************************************
#CLASSES
#***********************************************************************

#Placeholder class so pylint doesn't get mad
#TODO Bing to create model
class Model:
    pass 

#Wrapper class holding model and memory, will handle the prediction and learning steps
class Agent(Model):

    def __init__(self) -> None:
        super().__init__()
        self.memory = ReplayMemory(128)#TODO decide on replay memory size

    def action(self, state):
        """Given a state, choose an action"""
        pass

    def cache(self, experience):
        """Add the experience to memory"""
        pass

    def recall(self):
        """Sample experiences from memory"""
        pass

    def update(self):
        """Update online action value (Q) function with a batch of experiences"""
        pass

#Class to hold replay mem training cache
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


#*********************************************************************
# INITIALIZE 
#*********************************************************************
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Training Params
GAMMA = 0.999
EPS = 0.333
#TODO Decaying epsilon, relay memory
#BATCH_SIZE = 128
#EPS_START = 0.9
#EPS_END = 0.05
#EPS_DECAY = 200
#TARGET_UPDATE = 10

#Load instances
instances = [creategraph('Instances/'+file) for file in os.listdir('Instances/')]
for i in instances:
    print(len(i.teams),len(i.nodedict), len(i.nodedict)/(2*len(i.teams)*(len(i.teams)-1)**2)) #max num of nodes is 2*n*(n-1)^2

#Create agent
agent = Agent()

#***********************************************************************
#TRAINING
#***********************************************************************
for i in instances:
    
    episodes = 10 #TODO decide on reasonable # of episodes
    for e in range(episodes):

        graph = copy.deepcopy(i) #TODO make more efficient
        
        #Train
        while True:

            state = graph.getState()
            actions = graph.getActions()

            if random.random() <= EPS:
                pass 
                #TODO take random action
            else:
               pass
                #Greedy Q action
            
            #update graph -> return new state, reward
            #cache update in agent.model.memory()
            #update agent.model
            #check if end condition
            break #TODO take out

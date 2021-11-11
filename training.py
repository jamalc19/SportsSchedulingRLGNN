#DEPENDENCIES
import Graph
#TODO import model

import numpy as np
import math
import random
import copy

from collections import namedtuple, deque

#*********************************************************************
# INITIALIZE 
#*********************************************************************
#FIXME device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Training Params
GAMMA = 0.999
EPS = 0.333
TARGET_UPDATE = 10
#TODO Decaying epsilon, relay memory
#BATCH_SIZE = 128
#EPS_START = 0.9
#EPS_END = 0.05
#EPS_DECAY = 200


#Load instances
instances = [creategraph('Instances/'+file) for file in os.listdir('Instances/')]
for i in instances:
    print(len(i.teams),len(i.nodedict), len(i.nodedict)/(2*len(i.teams)*(len(i.teams)-1)**2)) #max num of nodes is 2*n*(n-1)^2


#***********************************************************************
#CLASSES
#***********************************************************************

#Placeholder class so pylint doesn't get mad
#TODO Bing to create model
class Model:
    pass 

class Agent(Model):
    """
    Wrapper class holding the model to train, and a cache of previous steps

    Attributes
        model
        memory
    """

    def __init__(self) -> None:
        super().__init__()
        self.memory = ReplayMemory(128)#TODO decide on replay memory size
        self.state = np.array()#FIXME
    
    def action(self, graph: Graph) -> int:
        """
        Given a graph, update the current state (struc2vec) and pick the next action (Q-value)
        
        Parameters
            graph (Graph): a graph representation of the current state of the instance

        Returns
            an int corresponding to the idnum of the selected node
            an embedded node
        """
        for i in graph.getActions():
            #calculate Q values

        if random.random() <= EPS:
                action = random.choice(graph.getActions()) #TODO make sure getActions() returns a list of possible actions
                #TODO select encoded node
        else:


    def cache(self, state, newState, action, reward):
        """Add the experience to memory"""
        pass

    def recall(self):
        """Sample experiences from memory"""
        pass

    def learn(self):
        """Backwards pass for model"""
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

#***********************************************************************
#TRAINING
#***********************************************************************
#Create agent
agent = Agent()

for i in instances:
    
    episodes = 10 #TODO decide on reasonable # of episodes
    for e in range(episodes):

        graph = copy.deepcopy(i) #TODO make more efficient
        state = np.zeros(embedLength) #TODO determine length of vector
        t = 1

        #Training
        while True:

            #Determine which action to take
            action, nodeToAdd = agent.action(graph)
            
            #Take action, recieve reward
            reward = graph.select(nodeToAdd)

            #Update state
            newState = state + np.array(list(action))#FIXME

            #Cache result
            agent.cache(state, newState, action, reward)

            #Train
            if t >= TARGET_UPDATE:
                agent.learn()
            else:
                t += 1

            break #TODO figure out break condition

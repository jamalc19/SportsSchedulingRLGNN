#DEPENDENCIES
import numpy as np
import random
import copy
import os

import torch
import torch.optim as optim

from collections import namedtuple, deque
from Graph import Graph, creategraph
from s2v_scheduling import Model, structure2vec

#*********************************************************************
# INITIALIZE 
#*********************************************************************

#Training Params
GAMMA = 0.999
EPS = 0.333
TARGET_UPDATE = 10
EMBED_SIZE = 128
BATCH_SIZE = 128
EPISODES = 100
#TODO Decaying epsilon, relay memory, remove constants from global
#EPS_START = 0.9
#EPS_END = 0.05
#EPS_DECAY = 200


#Load instances
instances = [creategraph('Instances/'+file) for file in os.listdir('Instances/')]
for i in instances:
    print(len(i.teams),len(i.nodedict), len(i.nodedict)/(2*len(i.teams)*(len(i.teams)-1)**2)) #max num of nodes is 2*n*(n-1)^2

#Use cuda
#FIXME device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")


#***********************************************************************
#CLASSES
#***********************************************************************

class Agent:
    """
    Wrapper class holding the model to train, and a cache of previous steps

    Attributes
        memory
        model
        use_cuda
    """

    def __init__(self) -> None:
        super().__init__()
        self.memory = deque(maxlen=1000)#TODO decide on replay memory size
        self.model = Model(EMBED_SIZE)

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.net = self.model.to(device="cuda")
    

    def action(self, graph: Graph) -> int:
        """
        Given a graph, update the current state (struc2vec) and pick the next action (Q-value)
        
        Parameters
            graph (Graph): a graph representation of the current state of the instance

        Returns
            actionID: the nodeID of the selected node
            action: the embedding for the corresponding nodeID
        """
        graphEmbeddings = Model.structure2vec(self.model, graph) #FIXME won't use the correct model
        
        qValueDict = {}
        for nodeID in graph.getActions():
            qValueDict[nodeID] = self.model.estimateQ(graphEmbeddings, nodeID)

        #eps-greedy action
        if random.random() <= EPS: #select random node to add
            actionID = random.choice(list(qValueDict.keys()))
        else: #select node with highest q
            actionID = max(qValueDict, key=qValueDict.get)
        
        action = graphEmbeddings[actionID]

        return actionID, action.toList() #int, list


    def cache(self, state, nextState, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (list),
        next_state (list),
        action (list),
        reward () #TODO decide on datatype 
        done(bool) #FIXME decide when done
        """

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            nextState = torch.tensor(next_state).cuda()
            action = torch.tensor(action).cuda()
            reward = torch.tensor([reward]).cuda() #TODO get rid of list comprehension?
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            nextState = torch.tensor(next_state)
            action = torch.tensor(action)
            reward = torch.tensor([reward])#TODO get rid of list comprehension?
            done = torch.tensor([done])

        self.memory.append((state, nextState, action, reward, done,))


    def recall(self):
        """
        Retrieve a batch of experiences from memory

        Returns
            state:
            next_state:
            action: 
            reward:
            done:
            
        """
        batch = random.sample(self.memory, BATCH_SIZE)
        state, nextState, action, reward, done = map(torch.stack, zip(*batch))

        return state, nextState, action.squeeze(), reward.squeeze(), done.squeeze() #TODO remove squeeze?

    def learn(self):
        """Backwards pass for model"""

        state, nextState, action, reward, done = self.recall()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9) #TODO tune params

        pass


#***********************************************************************
#TRAINING
#***********************************************************************

#Create agent
agent = Agent()

#Training loop
for i in instances:
    
    for e in range(EPISODES):

        graph = copy.deepcopy(i) #TODO make more efficient
        state = np.zeros(EMBED_SIZE)
        done = False
        t = 1 #TODO set training delay

        #Training
        while not done:

            #Determine which action to take
            nodeToAdd, action = agent.action(graph)
            
            #Take action, recieve reward
            reward, done = graph.select(nodeToAdd)

            #Determine state t+1
            nextState = np.add(state, action)

            #Cache result
            agent.cache(state, nextState, action, reward, done)

            #Train
            if t >= TARGET_UPDATE:
                agent.learn()
            else:
                t += 1

            #TODO In case of emergencies
            if t >= 200:
                done = True
            
            #Update state
            state = nextState

            break #TODO figure out break condition

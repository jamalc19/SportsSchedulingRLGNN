#DEPENDENCIES
import numpy as np
import random
import copy
import os

import torch
import torch.optim as optim
import torch.nn as nn

from collections import namedtuple, deque
from Graph import Graph, creategraph
from s2v_scheduling import Model, structure2vec

#*********************************************************************
# INITIALIZE 
#*********************************************************************

#Training params
GAMMA = 0.999
EPS = 0.333
TRAINING_DELAY = 1
EPISODES = 100
BATCH_SIZE = 128

#Agent params
EMBEDDING_SIZE = 10
CACHE_SIZE = 1000


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
        memory: stores previous instances for model training
        model: nn.module subclass for deep Q
        use_cuda: Boolean for cuda availability
        loss_fn: 
        optimizer: 
    """
    def __init__(self) -> None:
        super().__init__()
        self.memory = deque(maxlen=CACHE_SIZE)
        self.model = Model(EMBEDDING_SIZE)

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.net = self.model.to(device="cuda")   

        self.optimizer = nn.SGD(self.model.parameters(), lr=0.01, momentum=0.9)#TODO tune params
        self.loss_fn = nn.MSELoss()    

    def action(self, graph: Graph) -> int:
        """
        Given a graph, update the current state (struc2vec) and pick the next action (Q-value)
        
        Parameters
            graph (Graph): a graph representation of the current state of the instance

        Returns
            actionID: the nodeID of the selected node
            action: the embedding for the corresponding nodeID
        """
        graph_embeddings = structure2vec(self.model, graph)
        
        q_value_dict = {}
        for nodeID in graph.getActions():
            q_value_dict[nodeID] = self.model.estimateQ(graph_embeddings, nodeID)

        #eps-greedy action
        if random.random() <= EPS: #select random node to add
            actionID = random.choice(list(q_value_dict.keys()))
        else: #select node with highest q
            actionID = max(q_value_dict, key=q_value_dict.get)
        
        action = graph_embeddings[actionID]

        return actionID, action.toList() #int, list


    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (list),
        next_state (list),
        action (list),
        reward ()
        done(bool)
        """
        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor(action).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor(action)
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))


    def recall(self):
        """
        Retrieve a batch of experiences from memory

        Returns
            random 
            state:
            next_state:
            action: 
            reward:
            done:
        """
        batch = random.sample(self.memory, BATCH_SIZE)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))

        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self, y, Q):
        """Backwards pass for model"""
        #Retrive batch from memory
        # state, next_state, action, reward, done = self.recall()
                
        # loss = self.loss_fn(y, Q)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # return loss.item()
        pass


#***********************************************************************
#TRAINING
#***********************************************************************

#Create agent
agent = Agent()

#Training loop
for i in instances:
    
    for e in range(EPISODES):

        graph = copy.deepcopy(i)
        state = np.zeros(EMBEDDING_SIZE)
        done = False
        t = 1 #TODO set training delay
        cumulative_reward = 0 #TODO verify that this is the correct implementation

        #Training
        while not done:

            #Determine which action to take
            node_to_add, action = agent.action(graph)
            
            #Take action, recieve reward
            reward, done = graph.select(node_to_add)
            #TODO implement cumulative_reward += reward for 

            #Determine state t+1
            next_state = np.add(state, action)

            #Cache result
            agent.cache(state, next_state, action, reward, done)

            #Train
            if t >= TRAINING_DELAY:
                agent.learn()
            else:
                t += 1

            #In case of emergencies, break loop
            if t >= 200:
                done = True
            
            #Update state
            state = next_state

            break

#DEPENDENCIES
import math

import numpy as np
import random
import copy
import os
import pickle

import torch
import torch.optim as optim
import torch.nn as nn

from collections import namedtuple, deque
from Graph import Graph
from s2v_scheduling import Model

#*********************************************************************
# INITIALIZE 
#*********************************************************************

#Training params
GAMMA = 0.999
EPS_START = 0.05
EPS_END = 0.001
EPS_DECAY = 1000
TRAINING_DELAY = 1
EPISODES = 1000000
BATCH_SIZE = 128
TARGET_UPDATE = 3
N_STEP_LOOKAHEAD=10

#Agent params
EMBEDDING_SIZE = 64
CACHE_SIZE = 100


#Declare training instances
#instances= os.listdir('PreprocessedInstances/')
instances= ['OnlyHardITC2021_Test1.pkl'] #testing on just one instance for now

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
        self.target_model = Model(EMBEDDING_SIZE)
        self.target_model.load_state_dict(self.model.state_dict())

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.net = self.model.to(device="cuda")   

        self.optimizer =optim.RMSprop(self.model.parameters())#TODO tune params
        self.loss_fn = nn.SmoothL1Loss()
        #self.loss_fn = nn.MSELoss()


    def Q(self,graph,model='model'):
        '''

        :param graph:
        :param model: str in ('model','target')
        :return:
        '''
        network = self.model if model=='model' else self.target_model
        graph_embeddings = network.structure2vec(graph)

        q_value_dict = {}
        for nodeID in graph.getActions():
            q_value_dict[nodeID] = network.q_calc(graph_embeddings, nodeID)
        return q_value_dict, graph_embeddings


    def greedyepsilon(self, q_value_dict,t):
        """
        Given a graph, update the current state (struc2vec) and pick the next action (Q-value)
        
        Parameters
            graph (Graph): a graph representation of the current state of the instance

        Returns
            actionID: the nodeID of the selected node
            action: the embedding for the corresponding nodeID
        """


        #eps-greedy action
        EPS = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * t / EPS_DECAY)
        if random.random() <= EPS: #select random node to add
            actionID = random.choice(list(q_value_dict.keys()))
        else: #select node with highest q
            actionID = max(q_value_dict, key=q_value_dict.get)


        return actionID #int,

    def greedy(self, q_value_dict):
        """
        Given a graph, update the current state (struc2vec) and pick the next action (Q-value)

        Parameters
            graph (Graph): a graph representation of the current state of the instance

        Returns
            actionID: the nodeID of the selected node
            action: the embedding for the corresponding nodeID
        """
        return max(q_value_dict, key=q_value_dict.get)

    def cache(self, instance,partialsolution,action):
        """
        Store the experience to self.memory (replay buffer)
        Transitions are deterministic, so do not need to store rewards or next state.

        Inputs:
        instance (str),
        partialsolution (set),
        action (int)
        """
        self.memory.append((instance,partialsolution,action))


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
        #state, next_state, action, reward, done = map(torch.stack, zip(*batch))

        return batch

    def learn(self, y, Q):
        """Backwards pass for model"""
        #Retrive batch from memory
        # state, next_state, action, reward, done = self.recall()
                
        loss = self.loss_fn(Q, y)
        self.optimizer.zero_grad()
        loss.backward()
        #print(loss.item())
        self.optimizer.step()
        return
        # return loss.item()

    def batch_train(self):
        batch = self.recall()
        self.optimizer.zero_grad()
        for instance, partial_solution, action in batch:
            # get to current state
            graph = pickle.load(open('PreprocessedInstances/' + i, 'rb'))
            for node_id in partial_solution:
                graph.selectnode(node_id)
            #forward pass through the network
            q_value_dict, graph_embeddings = agent.Q(graph)
            #get next state and reward
            reward, done = graph.selectnode(action)

            # compute Q value of next state
            if done:
                nextstateQ = torch.zeros(1)
            elif len(graph.nodedict) + len(graph.solution) < graph.solutionsize:  # RL agent reached an infeasible solution
                nextstateQ = torch.tensor(-1.0)
            else:
                nextstateQ = max(agent.Q(graph, model='target')[0].values())  # get next state Qvalue with target network
            agent.learn(nextstateQ + reward, q_value_dict[node_to_add])

            loss = self.loss_fn(q_value_dict[action], nextstateQ + reward)/BATCH_SIZE

            loss.backward()
        self.optimizer.step()

    def rollout(self, i):
        with torch.no_grad():
            graph = pickle.load(open('PreprocessedInstances/' + i, 'rb'))

            done = False
            cumulative_reward = -graph.costconstant
            while not done:
                q_value_dict, graph_embeddings = agent.Q(graph)
                node_to_add = agent.greedy(q_value_dict)
                action = graph_embeddings[node_to_add]
                # Take action, recieve reward
                reward, done = graph.selectnode(node_to_add)
                cumulative_reward += reward
                if len(graph.nodedict) < graph.solutionsize:  # RL agent reached an infeasible solution
                    print('infeasible')
                    done = True
        return cumulative_reward



#***********************************************************************
#TRAINING
#***********************************************************************
warmstart=False
#warmstart = 'ModelParams/firsttestwarmstart1000'
if __name__=='__main__':

    #Create agent
    agent = Agent()
    if warmstart:
        agent.model.load_state_dict(torch.load(warmstart))

    np.random.seed(0)
    torch.manual_seed(0)
    #Training loop

    for e in range(EPISODES):
        i = instances[np.random.randint(0,len(instances))] #sample the next instance from a uniform distribution
        graph = pickle.load(open('PreprocessedInstances/'+i,'rb'))

        done = False
        t = 1 #TODO set training delay
        cumulative_reward = -graph.costconstant
        #Training
        while not done:
            #Determine which action to take
            q_value_dict, graph_embeddings = agent.Q(graph)
            #node_to_add= agent.greedy(q_value_dict) #node_to_add is the selected nodeid, action is the nodes s2v embedding
            node_to_add = agent.greedyepsilon(q_value_dict,t)  # node_to_add is the selected nodeid, action is the nodes s2v embedding

            #Cache state and action
            agent.cache(i, graph.solution, node_to_add)

            #Take action, recieve reward
            reward, done = graph.selectnode(node_to_add)
            cumulative_reward+=reward

            #Train
            if t >= TRAINING_DELAY:
                #agent.batch_train()
                if done:
                    nextstateQ=torch.zeros(1)
                elif len(graph.nodedict) + len(graph.solution) < graph.solutionsize: #RL agent reached an infeasible solution
                    nextstateQ = torch.tensor(-1.0)
                else:
                    nextstateQ = max(agent.Q(graph,model='target')[0].values()) #get next state Qvalue with target network
                agent.learn(nextstateQ+reward,q_value_dict[node_to_add])

            t += 1

            if t % TARGET_UPDATE == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())

            if len(graph.nodedict) < graph.solutionsize:  # RL agent reached an infeasible solution
                print('infeasible')
                done=True

            #Update state
            #state = next_state
        print(e,cumulative_reward)

        if e% 100 ==0:
            torch.save(agent.model.state_dict(), 'ModelParams/s2visactuallytraining{}'.format(e))

        if e%10==0:#rollout
            cumulative_reward=agent.rollout(i)
            print('Rollout Cumulative Reward:',cumulative_reward)

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

        #self.use_cuda = torch.cuda.is_available()
        # if self.use_cuda:
        #     self.net = self.model.to(device="cuda")   

        self.optimizer = optim.RMSprop(self.model.parameters())#TODO tune params
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
        batch = random.sample(self.memory, min(BATCH_SIZE,len(self.memory)))
        sortedbatch = {}
        for b in batch:
            if b[0] in sortedbatch:
                sortedbatch[b[0]].append(b[1:])
            else:
                sortedbatch[b[0]] = [b[1:]]
        return sortedbatch

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
        for instance in batch: #TODO this only works if graphs aren't dynamically changing besides the selected nodes being marked
            graph = pickle.load(open('PreprocessedInstances/' + instance, 'rb'))
            for partial_solution, action in batch[instance]:
                # get to current state
                for node_id in partial_solution:
                    graph.nodedict[node_id].selected=True
                    #graph.selectnode(node_id) #save for if we do use the dynamically changing graphs
                #forward pass through the network
                q_value_dict, graph_embeddings = self.Q(graph)
                #get next state and reward
                reward, done = graph.selectnode(action)
                # compute Q value of next state
                nsteprewards = 0
                if done:
                    nextstateQ = torch.zeros(1)
                elif len(graph.nodedict) + len(graph.solution) < graph.solutionsize:  # RL agent reached an infeasible solution
                    nextstateQ = torch.tensor(-1.0)
                else:
                    with torch.no_grad():
                        for i in range(N_STEP_LOOKAHEAD):
                            if not done:
                                nstep_q_value_dict, nstep_graph_embeddings=self.Q(graph, model='target')
                                node_to_add = self.greedy(nstep_q_value_dict)
                                reward, done = graph.selectnode(node_to_add)
                                nsteprewards+=reward
                                if len(graph.nodedict) < graph.solutionsize:  # RL agent reached an infeasible solution
                                    print('infeasible')
                                    done = True
                        nextstateQ = max(self.Q(graph, model='target')[0].values())  # get next state Qvalue with target network
                loss = self.loss_fn(q_value_dict[action], nextstateQ +nsteprewards+ reward)/min(BATCH_SIZE,len(self.memory))
                loss.backward()
                #revert graph to base state
                for node_id in graph.solution:
                    graph.nodedict[node_id].selected = False
                graph.solution=set()
        self.optimizer.step()

    def rollout(self, i):
        with torch.no_grad():
            graph = pickle.load(open('PreprocessedInstances/' + i, 'rb'))

            done = False
            cumulative_reward = -graph.costconstant
            while not done:
                q_value_dict, graph_embeddings = self.Q(graph)
                node_to_add = self.greedy(q_value_dict)
                # Take action, recieve reward
                reward, done = graph.selectnode(node_to_add)
                cumulative_reward += reward
                if len(graph.nodedict) < graph.solutionsize:  # RL agent reached an infeasible solution
                    print('infeasible')
                    done = True
        return cumulative_reward

def main():
    #Create agent
    agent = Agent()
    if warmstart:
        agent.model.load_state_dict(torch.load(warmstart))

    np.random.seed(0)
    torch.manual_seed(0)

    #Training loop
    t = 1
    for e in range(EPISODES):
        i = instances[np.random.randint(0,len(instances))] #sample the next instance from a uniform distribution
        graph = pickle.load(open('PreprocessedInstances/'+i, 'rb'))
        done = False
        cumulative_reward = -graph.costconstant
        
        #Training
        while not done:
            #Determine which action to take
            q_value_dict, graph_embeddings = agent.Q(graph)
            node_to_add = agent.greedyepsilon(q_value_dict, t)  # node_to_add is the selected nodeid, action is the nodes s2v embedding

            #Cache state and action
            agent.cache(i, graph.solution.copy(), node_to_add)

            #Take action, recieve reward
            reward, done = graph.selectnode(node_to_add)
            cumulative_reward+=reward

            #Train
            if t >= TRAINING_DELAY:
                if t % OPTIMIZE_FREQUENCY:
                    agent.batch_train()
                if t % TARGET_UPDATE == 0:
                    agent.target_model.load_state_dict(agent.model.state_dict())
            t += 1

            if len(graph.nodedict) < graph.solutionsize:  # RL agent reached an infeasible solution
                print('infeasible')
                done=True

            #Update state
            #state = next_state
        print(e, i, cumulative_reward)

        if (t >= TRAINING_DELAY) and (e % SAVE_FREQUENCY ==0):
            torch.save(agent.model.state_dict(), 'ModelParams/{}{}'.format(RUN_NAME, e))

        if (t >= TRAINING_DELAY) and (e % ROLLOUT_FREQUENCY==0):#rollout
            for i in instances:
                cumulative_reward=agent.rollout(i)
                print('Rollout Cumulative Reward for {}: {}'.format(i, cumulative_reward))

#*********************************************************************
# INITIALIZE
#*********************************************************************

#Training params
EPS_START = 0.1
EPS_END = 0.001
EPS_DECAY = 1000
TRAINING_DELAY = 10
EPISODES = 1000000
BATCH_SIZE = 10
N_STEP_LOOKAHEAD=10
TARGET_UPDATE = 10
OPTIMIZE_FREQUENCY=10

#Agent params
EMBEDDING_SIZE = 64
CACHE_SIZE = 1000

RUN_NAME = 'BatchTrainingFirstAttempt'
ROLLOUT_FREQUENCY =10
SAVE_FREQUENCY = 10

warmstart=False
# warmstart = 'ModelParams/s2visactuallytraining200'

#Use cuda
device = torch.device("cpu")
# use_cuda = torch.cuda.is_available()
# print(f"Using CUDA: {use_cuda}")

#Declare training instances
#instances= os.listdir('PreprocessedInstances/')
#instances=['OnlyHardITC2021_Test1.pkl','OnlyHardITC2021_Test2.pkl','OnlyHardITC2021_Test3.pkl','OnlyHardITC2021_Test4.pkl']#testing on just the small instances for now
instances=['OnlyHardITC2021_Test1.pkl']


if __name__=='__main__':
    main()

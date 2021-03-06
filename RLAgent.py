# DEPENDENCIES
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

from s2v_schedulingNew import Model as NewModel

from GreedyAgent import GreedyAgent


class RLAgent:
    """
    Wrapper class holding the model to train, and a cache of previous steps

    Attributes
        memory: stores previous instances for model training
        model: nn.module subclass for deep
        loss_fn:
        optimizer:
    """

    def __init__(self,CACHE_SIZE=1000,EMBEDDING_SIZE=64,EPS_START=1.0,EPS_END=0.05,EPS_STEP=10000,GAMMA=0.9,N_STEP_LOOKAHEAD=5, S2V_T=4, BATCH_SIZE=64, S2V='Original', IMITATION_LEARNING_EPISODES=50):
        super().__init__()
        self.EPS_START=EPS_START
        self.EPS_END=EPS_END
        self.EPS_STEP=EPS_STEP
        self.N_STEP_LOOKAHEAD=N_STEP_LOOKAHEAD
        self.S2V_T = S2V_T
        self.GAMMA=GAMMA
        self.BATCH_SIZE=BATCH_SIZE
        self.memory = deque(maxlen=CACHE_SIZE)
        if S2V=='Original':
            self.model = Model(EMBEDDING_SIZE)
            self.target_model = Model(EMBEDDING_SIZE)
        else:
            self.model = NewModel(EMBEDDING_SIZE)
            self.target_model = NewModel(EMBEDDING_SIZE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.IMITATION_LEARNING_EPISODES = IMITATION_LEARNING_EPISODES
        self.target_model_heuristic = GreedyAgent()

        self.optimizer = optim.RMSprop(self.model.parameters())
        self.loss_fn = nn.SmoothL1Loss()
        # self.loss_fn = nn.MSELoss()

    def Q(self, graph, model='model', slot=None):
        '''

        :param graph:
        :param model: str in ('model','target')
        :return:
        '''
        network = self.model if model == 'model' else self.target_model
        if slot is not None:
            actionnodeids = graph.getActions(slot)
            q_value_dict = network.structure2vec(graph, self.S2V_T,nodesubset=actionnodeids)
        else:
            actionnodeids = graph.getActions()
            q_value_dict = network.structure2vec(graph, self.S2V_T,nodesubset=actionnodeids)

        return q_value_dict


    def greedyepsilon(self, q_value_dict, t):
        """
        Given a graph, update the current state (struc2vec) and pick the next action (Q-value)

        Parameters
            graph (Graph): a graph representation of the current state of the instance

        Returns
            actionID: the nodeID of the selected node
            action: the embedding for the corresponding nodeID
        """
        # eps-greedy action
        EPS = self.EPS_END + max(0,(self.EPS_START - self.EPS_END) * (self.EPS_STEP- t) / self.EPS_STEP)
        if random.random() <= EPS:  # select random node to add
            actionID = random.choice(list(q_value_dict.keys()))
        else:  # select node with highest q
            actionID = max(q_value_dict, key=q_value_dict.get)

        return actionID  # int,

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

    def greedy_action(self, graph, restricted_action_space=False):
        if restricted_action_space:
            q_value_dict = self.Q(graph, slot = len(graph.solution) // int(len(graph.teams) / 2))
        else:
            q_value_dict = self.Q(graph)
        return max(q_value_dict, key=q_value_dict.get)

    def cache(self, instance, partialsolution, action):
        """
        Store the experience to self.memory (replay buffer)
        Transitions are deterministic, so do not need to store rewards or next state.

        Inputs:
        instance (str),
        partialsolution (set),
        action (int)
        """
        self.memory.append((instance, partialsolution, action))

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
        batch = random.sample(self.memory, min(self.BATCH_SIZE, len(self.memory)))
        sortedbatch = {}
        for b in batch:
            if b[0] in sortedbatch:
                sortedbatch[b[0]].append(b[1:])
            else:
                sortedbatch[b[0]] = [b[1:]]
        return sortedbatch

    def learn(self, y, Q):
        """Backwards pass for model"""
        # Retrive batch from memory
        # state, next_state, action, reward, done = self.recall()

        loss = self.loss_fn(Q, y)
        self.optimizer.zero_grad()
        loss.backward()
        # print(loss.item())
        self.optimizer.step()
        return
        # return loss.item()

    def batch_train(self, episode, restricted_action_space):
        batch = self.recall()
        self.optimizer.zero_grad()
        for instance in batch:  # TODO this only works if graphs aren't dynamically changing besides the selected nodes being marked
            '''
            graph = pickle.load(open('PreprocessedInstances/' + instance, 'rb'))#save for if we do use the dynamically changing graphs
            '''
            for partial_solution, action in batch[instance]:
                graph = pickle.load(open('TrainingInstances4teams/' + instance,
                                         'rb'))  # save for if we do use the dynamically changing graphs
                # get to current state
                for node_id in partial_solution:
                    # graph.nodedict[node_id].selected=True #save for if we use static graphs
                    graph.selectnode(node_id)  # save for if we do use the dynamically changing graphs
                # forward pass through the network
                if restricted_action_space:
                    q_value_dict = self.Q(graph, slot=len(graph.solution)//int(len(graph.teams)/2))
                else:
                    q_value_dict = self.Q(graph)
                # get next state and reward
                reward, done = graph.selectnode(action,restricted_action_space)
                # compute Q value of next state
                nsteprewards = 0
                if done:
                    nextstateQ = torch.zeros(1)
                else:
                    with torch.no_grad():
                        if episode > self.IMITATION_LEARNING_EPISODES:
                            for i in range(self.N_STEP_LOOKAHEAD):
                                if not done:
                                    if restricted_action_space:
                                        nstep_q_value_dict = self.Q(graph, model='target', slot=len(graph.solution) // int(len(graph.teams) / 2))
                                    else:
                                        nstep_q_value_dict = self.Q(graph, model='target')
                                    node_to_add = self.greedy(nstep_q_value_dict)
                                    reward, done = graph.selectnode(node_to_add,restricted_action_space)
                                    nsteprewards += self.GAMMA ** i * reward
                            if not done:
                                if restricted_action_space:
                                    nextstateQ = self.GAMMA ** (i + 1) * max(
                                        self.Q(graph, model='target', slot=len(graph.solution) // int(len(graph.teams) / 2)).values())  # get next state Qvalue with target network
                                else:
                                    nextstateQ = self.GAMMA ** (i + 1) * max(
                                        self.Q(graph, model='target').values())  # get next state Qvalue with target network
                            else:
                                nextstateQ = torch.zeros(1)
                        else:
                            i=0
                            while not done:
                                i+=1
                                node_to_add = self.target_model_heuristic.greedy_action(graph)
                                reward, done = graph.selectnode(node_to_add)
                                nsteprewards += self.GAMMA ** i * reward
                            nextstateQ = torch.zeros(1)
                loss = self.loss_fn(q_value_dict[action], nextstateQ + nsteprewards + reward) / min(self.BATCH_SIZE,
                                                                                                    len(self.memory))
                loss.backward()
                # revert graph to base state
                '''Save for if using dynamically changing graph
                for node_id in graph.solution:
                    graph.nodedict[node_id].selected = False
                graph.solution=set()
                '''
        self.optimizer.step()

    def rollout(self, i,restricted_action_space):
        with torch.no_grad():
            graph = pickle.load(open('TrainingInstances4teams/' + i, 'rb'))

            done = False
            cumulative_reward = -graph.costconstant
            while not done:
                currentslot = len(graph.solution) // int(len(graph.teams) / 2)
                if restricted_action_space:
                    q_value_dict = self.Q(graph, slot=currentslot)
                else:
                    q_value_dict = self.Q(graph)
                node_to_add = self.greedy(q_value_dict)
                # Take action, recieve reward
                reward, done = graph.selectnode(node_to_add,restricted_action_space)
                cumulative_reward += reward
        return cumulative_reward, len(graph.solution), graph.solutionsize
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

from RLAgent import RLAgent

def main():
    # Create agent
    agent = RLAgent(CACHE_SIZE=CACHE_SIZE,EMBEDDING_SIZE=EMBEDDING_SIZE,EPS_START=EPS_START,EPS_END=EPS_END,EPS_STEP=EPS_STEP,
                    GAMMA=GAMMA,N_STEP_LOOKAHEAD=N_STEP_LOOKAHEAD,BATCH_SIZE=BATCH_SIZE)
    if warmstart:
        agent.model.load_state_dict(torch.load(warmstart))

    np.random.seed(0)
    torch.manual_seed(0)
    # Training loop
    t = 1
    Rollouts={}
    for e in range(EPISODES):
        i = instances[np.random.randint(0, len(instances))]  # sample the next instance from a uniform distribution
        graph = pickle.load(open('PreprocessedInstances/' + i, 'rb'))
        done = False
        cumulative_reward = -graph.costconstant
        # Training
        while not done:
            # Determine which action to take
            q_value_dict, graph_embeddings = agent.Q(graph)
            # node_to_add= agent.greedy(q_value_dict) #node_to_add is the selected nodeid, action is the nodes s2v embedding
            node_to_add = agent.greedyepsilon(q_value_dict,
                                              t)  # node_to_add is the selected nodeid, action is the nodes s2v embedding

            # Cache state and action
            agent.cache(i, graph.solution.copy(), node_to_add)

            # Take action, recieve reward
            reward, done, feas = graph.selectnode(node_to_add)

            cumulative_reward += reward

            # Train
            if t >= TRAINING_DELAY:
                if t % OPTIMIZE_FREQUENCY:
                    agent.batch_train()
                if t % TARGET_UPDATE == 0:
                    agent.target_model.load_state_dict(agent.model.state_dict())
            t += 1

            # Update state
            # state = next_state
        print(e, i, cumulative_reward, len(graph.solution), graph.solutionsize)

        if (t >= TRAINING_DELAY) and (e % SAVE_FREQUENCY == 0):
            torch.save(agent.model.state_dict(), 'ModelParams/{}{}'.format(RUN_NAME, e))

        if (t >= TRAINING_DELAY) and (e % ROLLOUT_FREQUENCY == 0):  # rollout
            Rollouts[e]={}
            for i in instances:
                cumulative_reward, solutionlength, fullsolutionsize = agent.rollout(i)
                print('Rollout Cumulative Reward for {}: {}, Partial Solution Length: {}, Target Solution Size: {}'.format(i, cumulative_reward,
                                                                                                 solutionlength,fullsolutionsize))
                Rollouts[e][i] = (cumulative_reward,solutionlength,fullsolutionsize)
            pickle.dump(Rollouts,open('Results/{}{}'.format(RUN_NAME, e),'wb'))


# *********************************************************************
# INITIALIZE
# *********************************************************************

# Training params
EPS_START = 1.0
EPS_END = 0.05
EPS_STEP = 10000
TRAINING_DELAY = 100
EPISODES = 30000
BATCH_SIZE = 64
N_STEP_LOOKAHEAD = 5
TARGET_UPDATE = 10
OPTIMIZE_FREQUENCY = 10
GAMMA = 0.9

# Agent params
EMBEDDING_SIZE = 64
CACHE_SIZE = 1000

# Declare training instances
instances= [inst for inst in os.listdir('PreprocessedInstances/') if 'NoComplexgen_instance' in inst]
#instances = ['OnlyHardITC2021_Test1.pkl', 'OnlyHardITC2021_Test2.pkl', 'OnlyHardITC2021_Test3.pkl','OnlyHardITC2021_Test4.pkl']  # testing on just the small instances for now


# ***********************************************************************
# TRAINING
# ***********************************************************************
warmstart = False
#warmstart = 'ModelParams/128EmbeddingSize10'
RUN_NAME = 'SyntheticInstancesFirstTrain'
ROLLOUT_FREQUENCY = 50
SAVE_FREQUENCY = 10
if __name__ == '__main__':
    main()

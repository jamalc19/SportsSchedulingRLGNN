# DEPENDENCIES
import numpy as np
import os
import pickle

import torch
import torch.optim as optim
import torch.nn as nn

from collections import namedtuple, deque
from Graph import Graph
from s2v_scheduling2 import Model
import csv

from RLAgent2 import RLAgent

def main():

    #Set up outputs
    param_output_list = [
        'RUN_NAME: ' + RUN_NAME + '\n',
        'INSTANCE_SUMMARY ' + INSTANCE_SUMMARY +'\n', 
        '\n***********************AGENT PARAMS*******************\n\n',
        'CACHE_SIZE: ' + str(CACHE_SIZE) + '\n',
        'EMBEDDING_SIZE: ' + str(EMBEDDING_SIZE) + '\n'
        'EPS_START: ' + str(EPS_START) + '\n',
        'EPS_END: ' + str(EPS_END) + '\n',
        'EPS_STEP: ' + str(EPS_STEP) + '\n',
        'GAMMA: ' + str(GAMMA) + '\n',
        'BATCH_SIZE: ' + str(BATCH_SIZE) + '\n',
        'N_STEP_LOOKAHEAD: ' + str(N_STEP_LOOKAHEAD) + '\n',
        'S2V_T: ' + str(S2V_T) + '\n',
        '\n*******************TRAINING PARAMS********************\n\n',
        'EPISODES: ' + str(EPISODES) + '\n',
        'TRAINING_DELAY: ' + str(TRAINING_DELAY) + '\n',
        'TARGET_UPDATE: ' + str(TARGET_UPDATE) + '\n',
        'OPTIMIZE_FREQUENCY: ' + str(OPTIMIZE_FREQUENCY) + '\n',
        'ROLLOUT_FREQUENCY: ' + str(ROLLOUT_FREQUENCY) + '\n',
        'SAVE_FREQUENCY: ' + str(SAVE_FREQUENCY) + '\n'
        ]
    param_output = open('TestResults/{name}.txt'.format(name = RUN_NAME), 'w')
    param_output.writelines(param_output_list)
    param_output.close()

    training_output = open("TestResults/{name}.csv".format(name=RUN_NAME), 'w',newline='')
    training_output_writer=csv.writer(training_output)
    training_output_writer.writerow(['episode','instance','feasible?','cumulative reward','solution length'])

    #Create agent
    agent = RLAgent(CACHE_SIZE=CACHE_SIZE,EMBEDDING_SIZE=EMBEDDING_SIZE,EPS_START=EPS_START,EPS_END=EPS_END,EPS_STEP=EPS_STEP,
                    GAMMA=GAMMA,N_STEP_LOOKAHEAD=N_STEP_LOOKAHEAD, S2V_T = S2V_T, BATCH_SIZE=BATCH_SIZE, IMITATION_LEARNING_EPISODES=IMITATION_LEARNING_EPISODES)
    if warmstart:
        agent.model.load_state_dict(torch.load(warmstart))
        agent.target_model.load_state_dict(agent.model.state_dict())

    #Training loop
    np.random.seed(0)
    torch.manual_seed(0)
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
            q_value_dict = agent.Q(graph)
            # node_to_add= agent.greedy(q_value_dict) #node_to_add is the selected nodeid, action is the nodes s2v embedding
            node_to_add = agent.greedyepsilon(q_value_dict,
                                              t)  # node_to_add is the selected nodeid, action is the nodes s2v embedding

            # Cache state and action
            agent.cache(i, graph.solution.copy(), node_to_add)

            # Take action, recieve reward
            reward, done = graph.selectnode(node_to_add)

            cumulative_reward += reward

            # Train
            if t >= TRAINING_DELAY:
                if t % OPTIMIZE_FREQUENCY:
                    agent.batch_train(e)
                if t % TARGET_UPDATE == 0:
                    agent.target_model.load_state_dict(agent.model.state_dict())
            t += 1
        feasible = len(graph.solution)== graph.solutionsize
        print(e, i, feasible, cumulative_reward, len(graph.solution), graph.solutionsize)
        training_output_writer.writerow([e, i,feasible,cumulative_reward, len(graph.solution), graph.solutionsize])

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
    training_output.close()


# *********************************************************************
# INITIALIZE
# *********************************************************************

RUN_NAME = 'ImitationLearningFirstAttempt'

# Agent params
CACHE_SIZE = 1000
EMBEDDING_SIZE = 16
EPS_START = 1.0
EPS_END = 0.05
EPS_STEP = 1000
GAMMA = 0.9
BATCH_SIZE = 32
N_STEP_LOOKAHEAD = 5
S2V_T = 2

# Training params
EPISODES = 30000
TRAINING_DELAY = 100
TARGET_UPDATE = 20
OPTIMIZE_FREQUENCY = 20
ROLLOUT_FREQUENCY = 25
SAVE_FREQUENCY = 10
IMITATION_LEARNING_EPISODES=50

# Declare training instances
INSTANCE_SUMMARY = 'Preproccessed synthetic instances with no complex constraints'
instances= [inst for inst in os.listdir('PreprocessedInstances/') if 'NoComplexgen_instance' in inst]
#instances= [inst for inst in os.listdir('PreprocessedInstances/')]
#instances = ['OnlyHardITC2021_Test1.pkl', 'OnlyHardITC2021_Test2.pkl', 'OnlyHardITC2021_Test3.pkl','OnlyHardITC2021_Test4.pkl']  # testing on just the small instances for now

warmstart = False
#warmstart = 'ModelParams/128EmbeddingSize10'

# ***********************************************************************
# TRAINING
# ***********************************************************************
if __name__ == '__main__':
    main()

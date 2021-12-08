# DEPENDENCIES
import numpy as np
import os
import pickle

import torch
import torch.optim as optim
import torch.nn as nn

from collections import namedtuple, deque
from Graph import Graph
from s2v_scheduling import Model
import csv

from RLAgent import RLAgent

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
                    GAMMA=GAMMA,N_STEP_LOOKAHEAD=N_STEP_LOOKAHEAD, S2V_T = S2V_T, BATCH_SIZE=BATCH_SIZE)
    if warmstart:
        agent.model.load_state_dict(torch.load(warmstart))

    #Training loop
    np.random.seed(0)
    torch.manual_seed(0)
    t = 1
    Rollouts={}
    for e in range(EPISODES):
        i = instances[np.random.randint(0, len(instances))]  # sample the next instance from a uniform distribution
        graph = pickle.load(open('TrainingInstances4teams/' + i, 'rb'))
        done = False
        cumulative_reward = -graph.costconstant
        # Training
        while not done:
            currentslot = len(graph.solution)//int(len(graph.teams)/2)
            # Determine which action to take
            if RESTRICTED_ACTION_SPACE:
                q_value_dict, graph_embeddings = agent.Q(graph, slot=currentslot)
            else:
                q_value_dict, graph_embeddings = agent.Q(graph)
            # node_to_add= agent.greedy(q_value_dict) #node_to_add is the selected nodeid, action is the nodes s2v embedding
            node_to_add = agent.greedyepsilon(q_value_dict,
                                              t)  # node_to_add is the selected nodeid, action is the nodes s2v embedding
            print(graph.nodedict[node_to_add].slot)
            # Cache state and action
            agent.cache(i, graph.solution.copy(), node_to_add)

            # Take action, recieve reward
            reward, done = graph.selectnode(node_to_add)

            cumulative_reward += reward
            if RESTRICTED_ACTION_SPACE:
                if len(graph.getActions(len(graph.solution)//int(len(graph.teams)/2)))==0:
                    done=True#infeasible

            # Train
            if t >= TRAINING_DELAY:
                if t % OPTIMIZE_FREQUENCY:
                    agent.batch_train(RESTRICTED_ACTION_SPACE)
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
                cumulative_reward, solutionlength, fullsolutionsize = agent.rollout(i,RESTRICTED_ACTION_SPACE)
                print('Rollout Cumulative Reward for {}: {}, Partial Solution Length: {}, Target Solution Size: {}'.format(i, cumulative_reward,
                                                                                                 solutionlength,fullsolutionsize))
                Rollouts[e][i] = (cumulative_reward,solutionlength,fullsolutionsize)
            pickle.dump(Rollouts,open('Results/{}{}'.format(RUN_NAME, e),'wb'))
    training_output.close()


# *********************************************************************
# INITIALIZE
# *********************************************************************



# Agent params
CACHE_SIZE = 1000
EMBEDDING_SIZE = 64
EPS_START = 1.0
EPS_END = 0.05
EPS_STEP = 1000
GAMMA = 0.9
BATCH_SIZE = 16
N_STEP_LOOKAHEAD = 5
S2V_T = 1
RESTRICTED_ACTION_SPACE=True

# Training params
EPISODES = 1000
TRAINING_DELAY = 100
TARGET_UPDATE = 10
OPTIMIZE_FREQUENCY = 10
ROLLOUT_FREQUENCY = 50
SAVE_FREQUENCY = 10

# Declare training instances
INSTANCE_SUMMARY = '4Teams'
RUN_NAME = 'T=4FourTeamInstancesRestrictedActionSpace'
instances= [inst for inst in os.listdir('TrainingInstances4teams/')]
#instances= [inst for inst in os.listdir('PreprocessedInstances/')]
#instances = ['OnlyHardITC2021_Test1.pkl', 'OnlyHardITC2021_Test2.pkl', 'OnlyHardITC2021_Test3.pkl','OnlyHardITC2021_Test4.pkl']  # testing on just the small instances for now

warmstart = False
#warmstart = 'ModelParams/128EmbeddingSize10'

# ***********************************************************************
# TRAINING
# ***********************************************************************
if __name__ == '__main__':
    main()

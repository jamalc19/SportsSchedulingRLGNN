#DEPENDENCIES
import pandas as pd
import numpy as np
import pickle

from Graph import Graph
from s2v_scheduling import Model
from training import EMBEDDING_SIZE

#*************************************
#CLASSES
#*************************************
class GreedyAgent:
    """"
    
    """
    def __init__(self):
        pass

    def greedy_action(self, current_nodeID: int, graph: Graph): 
        """
        :param graph: a graph representation of current state
        :param current_nodeID: the ID# of the most recent node added to the solution
        :return: a nodeID corresponding to the action with the lowest cost (or lowest connectivity if tied for cost)
        """

        possible_actions = []

        #Compute cost and and connectivity of each candidate for addition (possible actions)
        for nodeID in graph.getActions():
            node_connectivity = 0
            cost = 0
            
            for edge in graph.nodedict[nodeID].edges_soft.values():
                node_connectivity += 1
                if edge.node1.id == current_nodeID and edge.node2.id == nodeID: #if edge connects current node and nodeID, add weight of that edge
                    cost += edge.weight

            for edge in graph.nodedict[nodeID].edges_hard.values():
                node_connectivity += 1
                if edge.node1.id == current_nodeID and edge.node2.id == nodeID: #if edge connects current node and nodeID, add cost of that edge
                    cost += edge.weight
            
            cost += graph.nodedict[nodeID].cost
            possible_actions.append([nodeID, cost, node_connectivity])
        
        #sort possible actions by cost (small -> large) first, by connectivity (small -> large) second
        df = pd.DataFrame(possible_actions, columns=['nodeID', 'cost', 'connectivity'])
        df = df.sort_values(['cost', 'connectivity'])

        return df.iloc[0].nodeID




def training():
    agent = GreedyAgent()  

    instances=['OnlyHardITC2021_Test1.pkl','OnlyHardITC2021_Test2.pkl','OnlyHardITC2021_Test3.pkl','OnlyHardITC2021_Test4.pkl']

    for instance in instances:
        graph = pickle.load(open('PreprocessedInstances/'+instance, 'rb'))
        done = False
        cumulative_reward = -graph.costconstant
        added_nodes = []
        
        #Training
        while not done:
            #print("current solution is: ", added_nodes)

            if not added_nodes: #if solution is empty
                node_to_add = agent.greedy_action(None, graph) #no starting node, and the current graph
            else:
                node_to_add = agent.greedy_action(added_nodes[-1], graph) #most recent node added, and the current graph
            
            reward, done = graph.selectnode(node_to_add)
            cumulative_reward+=reward      
            added_nodes.append(node_to_add)
            #print("added node: {n}, reward was {r}, cumulative reward is {c}, solution size is {s}, done is {d}".format(n=node_to_add, r=reward, c=cumulative_reward, s=len(added_nodes), d=done))    
        print("Solved instance {i}, reward was {r}, cumulative reward was {c}, solution size was {s}".format(i=instance, r=reward, c=cumulative_reward, s=len(added_nodes)))

if __name__=='__main__':
    training()
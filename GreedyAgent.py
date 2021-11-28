#DEPENDENCIES
import pickle

from Graph import Graph

#*************************************
#CLASSES
#*************************************
class GreedyAgent:
    """"
    
    """
    def __init__(self):
        pass

    def greedy_action(self, graph: Graph):
        """
        :param graph: a graph representation of current state
        :param current_nodeID: the ID# of the most recent node added to the solution
        :return: a nodeID corresponding to the action with the lowest cost (or lowest connectivity if tied for cost)
        """
        bestnode=None
        bestincurredcost=99999999999999999
        bestpotentialcost=0

        for nodeID in graph.getActions():
            potentialcost = 0
            incurredcost = 0
            node = graph.nodedict[nodeID]
            for edgeset in (node.edges_soft,node.edges_hard):
                for othernode in edgeset:
                    edge = edgeset[othernode]
                    if graph.nodedict[othernode].selected:
                        incurredcost+=edge.weight
                    else:
                        potentialcost+=edge.weight

            if incurredcost< bestincurredcost:
                bestincurredcost=incurredcost
                bestpotentialcost=potentialcost
                bestnode=nodeID
            elif (incurredcost==bestincurredcost) and (potentialcost < bestpotentialcost):
                bestincurredcost=incurredcost
                bestpotentialcost=potentialcost
                bestnode=nodeID
        return bestnode


def solving():
    agent = GreedyAgent()  

    instances=['OnlyHardITC2021_Test1.pkl','OnlyHardITC2021_Test2.pkl','OnlyHardITC2021_Test3.pkl','OnlyHardITC2021_Test4.pkl']

    for instance in instances:
        graph = pickle.load(open('PreprocessedInstances/'+instance, 'rb'))
        done = False
        cumulative_reward = -graph.costconstant

        #Training
        while not done:
            #print("current solution is: ", added_nodes)
            node_to_add = agent.greedy_action( graph)
            
            reward, done = graph.selectnode(node_to_add)
            cumulative_reward+=reward
            #print("added node: {n}, reward was {r}, cumulative reward is {c}, solution size is {s}, done is {d}".format(n=node_to_add, r=reward, c=cumulative_reward, s=len(added_nodes), d=done))    
        print("Solved instance {i}, cumulative reward was {c}, solution size was {s}".format(i=instance, c=cumulative_reward, s=len(graph.solution)))

if __name__=='__main__':
    solving()
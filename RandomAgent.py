from Graph import Graph
import numpy as np

class RandomAgent:
    def __init__(self):
        pass

    def greedy_action(self, graph: Graph,restricted_action_space=None):
        actions = list(graph.getActions())
        return actions[np.random.randint(0,len(actions))]

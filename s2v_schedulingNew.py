import torch
import torch.nn as nn
import copy
import pickle

from torch.cuda.amp import custom_bwd, custom_fwd


class DifferentiableClamp(torch.autograd.Function):
    """
    From https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/3
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)

torch.manual_seed(0)
class Model(nn.Module):
    def __init__(self, p_dim,epsilon=0.0):
        self.p_dim = p_dim
        super(Model, self).__init__()
        self.theta1 = nn.Linear(p_dim, p_dim, bias=False) #non-complex
        self.theta1complex = nn.Linear(p_dim, p_dim, bias=False)  # complex
        self.theta2 = nn.Linear(p_dim, p_dim, bias=False) #non-complex
        self.thetaQ1 = nn.Linear(p_dim, 1, bias=False)
        self.thetaQ2 = nn.Linear(p_dim, p_dim, bias=False)
        self.epsilon=epsilon

    def forward(self, nodeID,graph, IncurredCosts, Ps, StepsRemaining, old_mu):
        innersum = torch.zeros(self.p_dim)
        node = graph.nodedict[nodeID]
        for edgeset in (node.edges_soft, node.edges_hard):
            for othernode in edgeset:
                edge = edgeset[othernode]
                if (othernode in self.node_id_to_tensor_index) and (not graph.nodedict[othernode].selected):
                    othernodeindex= self.node_id_to_tensor_index[othernode]
                    innersum+= Ps[othernodeindex].item()*StepsRemaining*(self.theta2(old_mu[othernodeindex])+ edge.weight)
        complexinnersum = torch.zeros(self.p_dim)
        for edgeset in (node.edges_soft_complex, node.edges_hard_complex):
            for othernode in edgeset:
                edge = edgeset[othernode]
                if not graph.nodedict[othernode].selected:
                    othernodeindex= self.node_id_to_tensor_index[othernode]
                    complexinnersum+= Ps[othernodeindex]*StepsRemaining*(edge.weight)
                else:
                    complexinnersum +=edge.weight
        return IncurredCosts[self.node_id_to_tensor_index[nodeID]] + torch.relu(self.theta1(innersum)) + torch.relu(self.theta1complex(complexinnersum))

    def q_function(self, mu, graph):
        # min q value is known to be negative of the hard constraint cost, so clamp it.
        return DifferentiableClamp.apply(self.thetaQ1(torch.relu(self.thetaQ2(mu))), -graph.hardconstraintcost, None)

    def computeIncurredCosts(self,graph,candidate_nodes,IncurredCosts):
        for nodeID in candidate_nodes:
            node = graph.nodedict[nodeID]
            incurredcost = node.cost
            for edgeset in (node.edges_soft, node.edges_hard):
                for othernode in edgeset:
                    edge = edgeset[othernode]
                    if graph.nodedict[othernode].selected:
                        incurredcost += edge.weight
            IncurredCosts[self.node_id_to_tensor_index[nodeID]] = incurredcost

    def structure2vec(self, graph, t=4,nodesubset=None):  # t is iterations which is rec'd as 4
        if not nodesubset:
            nodesubset= graph.nodedict

        node_num = len(nodesubset)
        self.node_id_to_tensor_index = {node:i for i,node in enumerate(nodesubset)}

        StepsRemaining = graph.solutionsize - len(graph.solution)
        Ps = torch.full((node_num,1),1/node_num)
        IncurredCosts = torch.zeros(node_num,1)
        mu = torch.zeros(node_num, self.p_dim)
        Qs = torch.zeros(node_num,1)
        self.computeIncurredCosts(graph, nodesubset, IncurredCosts)

        for _ in range(t):
            old_mu = mu.clone().detach()
            for nodeID in nodesubset:
                mu[self.node_id_to_tensor_index[nodeID]] = self.forward(nodeID,graph, IncurredCosts, Ps, StepsRemaining, old_mu)
            Qs = self.q_function(mu,graph)
            Ps = torch.softmax(Qs,0).clone().detach()*(1-self.epsilon) + self.epsilon/node_num

        return dict(zip(self.node_id_to_tensor_index.keys(),Qs))



if __name__ == '__main__':
    import numpy as np
    import os
    from Graph import creategraph
    from Graph import Graph
    from NodeAndEdge import Node,Edge

    torch.set_printoptions(threshold=np.nan)

    #graphs = [creategraph('Instances/' + file) for file in os.listdir('Instances/')]
    g=pickle.load(open('PreprocessedInstances/ITC2021_Test1.pkl','rb'))
    p_dim=1
    model=Model(p_dim)

    mapping, Qs = model.structure2vec(g)

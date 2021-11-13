import torch
import torch.nn as nn
import copy


class Model(nn.Module):
    def __init__(self, p_dim):
        super(Model, self).__init__()
        self.theta1 = nn.Linear(p_dim, p_dim, bias=False)
        self.theta2 = nn.Linear(p_dim, p_dim, bias=False)
        self.theta3_hard = nn.Linear(1, p_dim, bias=False)    # h
        self.theta4 = nn.Linear(p_dim, p_dim, bias=False)
        self.theta5_hardcomplex = nn.Linear(1, p_dim, bias=False)  # hc
        self.theta6 = nn.Linear(p_dim, p_dim, bias=False)
        self.theta7_soft = nn.Linear(1, p_dim, bias=False)  # s
        self.theta8 = nn.Linear(p_dim, p_dim, bias=False)
        self.theta9_softcomplex = nn.Linear(1, p_dim, bias=False)  # sc
        self.theta10_node = nn.Linear(3, p_dim, bias=False)  # xi

        self.thetaQ1 = nn.Linear(2*p_dim, 1, bias=False)
        self.thetaQ2 = nn.Linear(p_dim, p_dim, bias=False)
        self.thetaQ3 = nn.Linear(p_dim, p_dim, bias=False)

    def forward(self, xi, mu_N, h, hc, s, sc):
        tmp = self.theta1(torch.sum(mu_N, 0)) + \
              self.theta2(torch.sum(torch.relu(self.theta3_hard(h)), 0)) + \
              self.theta4(torch.sum(torch.relu(self.theta5_hardcomplex(hc)), 0)) + \
              self.theta6(torch.sum(torch.relu(self.theta7_soft(s)), 0)) + \
              self.theta8(torch.sum(torch.relu(self.theta9_softcomplex(sc)), 0))

        mu = torch.relu(tmp + self.theta10_node(xi))

        return mu

    def q_function(self, mu_all, mu_node):
        q_inner = torch.cat((self.thetaQ2(torch.sum(mu_all, 0)), self.thetaQ3(mu_node)), 0)

        q = self.thetaQ1(torch.relu(q_inner))

        return q


def edge_weights(mu_all, node, nd_edges):
    mu_N = []
    weights = []

    for edge in nd_edges:
        node1 = nd_edges[edge].node1.id
        node2 = nd_edges[edge].node1.id

        if int(node1) != int(node):
            edge_to = node1
        else:
            edge_to = node2

        mu_N.append(mu_all[edge_to].unsqueeze(0))
        weights.append(nd_edges[edge].weight)

    return mu_N, weights


def structure2vec(s2v: Model, graph, p_dim=128, t=4):  # t is iterations which is rec'd as 4
    node_list = copy.deepcopy(graph)
    ser_num_list = []

    for node in node_list.nodedict:
        ser_num_list.append(node_list.nodedict[node].id)

    node_num = len(node_list.nodedict)

    mu_all = torch.zeros(node_num, p_dim)
    #x_all = []

    for _ in range(t):
        for node in node_list.nodedict:
            mu_N = []
            h = []
            hc = []
            s = []
            sc = []

            mu_1, weight1 = edge_weights(mu_all, node, node_list.nodedict[node].edges_hard)
            mu_2, weight2 = edge_weights(mu_all, node, node_list.nodedict[node].edges_hard_complex)
            mu_3, weight3 = edge_weights(mu_all, node, node_list.nodedict[node].edges_soft)
            mu_4, weight4 = edge_weights(mu_all, node, node_list.nodedict[node].edges_soft_complex)

            mu_N.extend(mu_1+mu_2+mu_3+mu_4)
            h.extend(weight1)
            hc.extend(weight2)
            s.extend(weight3)
            sc.extend(weight4)

            if len(mu_N) > 0:
                mu_N = torch.cat(mu_N)
            else:
                mu_all[node_list.nodedict[node].id] = torch.zeros(p_dim)
                continue

            h = torch.Tensor(h).unsqueeze(1)
            hc = torch.Tensor(hc).unsqueeze(1)
            s = torch.Tensor(s).unsqueeze(1)
            sc = torch.Tensor(sc).unsqueeze(1)
            xi = [int(node_list.nodedict[node].hometeam),
                  int(node_list.nodedict[node].awayteam),
                  int(node_list.nodedict[node].slot)]

            xi = torch.Tensor(xi)

            mu_all[node_list.nodedict[node].id] = s2v(xi, mu_N, h, hc, s, sc)
            #x_all.append(xi)

        return dict(zip(ser_num_list, mu_all.data))


def q_calc(embedding_dict, node_num, p_dim=128):
    q2v = Model(p_dim)

    mu_all = [embedding_dict[i] for i in embedding_dict]

    mu_all = torch.stack(mu_all, 0)
    mu_node = embedding_dict[node_num]

    q_val = q2v.q_function(mu_all, mu_node)

    return q_val


if __name__ == '__main__':
    import numpy as np
    import os
    from Graph import creategraph

    torch.set_printoptions(threshold=np.nan)

    graphs = [creategraph('Instances/' + file) for file in os.listdir('Instances/')]
    node_embedding = structure2vec(graphs[-1])

    print(node_embedding)

    q_calculation1 = q_calc(node_embedding, 1)

    print(q_calculation1)

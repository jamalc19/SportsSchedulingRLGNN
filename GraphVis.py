import networkx as nx
import Graph
import NodeAndEdge
from pyvis.network import Network
import pickle
import os

if __name__ == '__main__':
    g = Graph.creategraph('../GenInstances/gen_instance_28.xml', hardconstraintcost=10000)
    # g = pickle.load(open('PreprocessedInstances/NoComplexgen_instance_41.pkl', 'rb'))
    # g = pickle.load(open('PreprocessedInstances/'+'OnlyHardITC2021_Test4.pkl', 'rb'))
    nxg = nx.Graph()
    dicto = g.nodedict

    for nodeID, node in g.nodedict.items():
        nxg.add_node(nodeID)
        for edgeset in (node.edges_soft,node.edges_hard):
            for othernodeID in edgeset:
                nxg.add_node(othernodeID)
                nxg.add_edge(nodeID, othernodeID)

  
    net = Network('1000px', '1000px')
    net.from_nx(nxg)
    # net.barnes_hut()
    net.show_buttons()
    net.show('gen4.html')
